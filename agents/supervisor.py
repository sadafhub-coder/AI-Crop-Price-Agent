"""
agents/supervisor.py
--------------------
Supervisor Agent — the brain that routes a question to the right specialist(s)
and merges their output into one farmer-friendly answer.

Flow:
    question -> QueryParserAgent -> intent + entities
             -> one or more of MarketPriceAgent / TrendAgent /
                PredictionAgent / RecommendationAgent
             -> a single formatted reply (+ structured `data` for the UI)

Every failure path produces a helpful message: unknown crop lists the valid
crops, an empty question asks for one, a missing model falls back to statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.market_agent import MarketPriceAgent
from agents.prediction_agent import DISCLAIMER as PREDICTION_DISCLAIMER
from agents.prediction_agent import PredictionAgent
from agents.query_parser import QueryParserAgent
from agents.recommendation_agent import RecommendationAgent
from agents.trend_agent import TrendAgent
from tools.market_api import MarketAPI, get_market_api
from utils.gemini import GeminiClient, get_gemini_client
from utils.helpers import (
    err,
    format_currency,
    format_pct,
    humanize_days,
    ok,
    trend_emoji,
)

HELP_TEXT = """🤖 I can help you with crop prices in the mandis I have data for.

Try asking me:
• What is the current price of wheat?
• Predict wheat price for the next 7 days in Aurangabad.
• Show me the price trend of rice.
• Should I sell my onion now or wait?
• Compare wheat and rice prices.
• Which market has the best price for tomatoes?
• Which crop has the highest price currently?"""


class SupervisorAgent:
    """Coordinates every other agent and produces the final chat answer."""

    name = "supervisor"

    def __init__(
        self,
        api: Optional[MarketAPI] = None,
        gemini: Optional[GeminiClient] = None,
        use_gemini: bool = True,
    ):
        self.api = api or get_market_api()
        self.gemini = gemini or get_gemini_client()
        self.use_gemini = use_gemini

        self.market_agent = MarketPriceAgent(self.api, self.gemini)
        self.trend_agent = TrendAgent(self.api, self.gemini)
        self.prediction_agent = PredictionAgent(self.api, self.gemini)
        self.recommendation_agent = RecommendationAgent(
            self.api, self.trend_agent, self.prediction_agent, self.gemini
        )
        self.parser = QueryParserAgent(
            known_crops=self.api.crops(),
            known_markets=self.api.markets(),
            gemini=self.gemini,
            use_gemini=use_gemini,
        )

    # ---------------------------------------------------------------- public
    def handle(self, query: str) -> Dict[str, Any]:
        """
        Main entry point used by the chatbot.

        Returns: {ok, reply, intent, agents_used, data, parsed}
        `reply` is ready to print; `data` carries the structured results so the
        Streamlit UI can chart the same numbers the text describes.
        """
        parsed = self.parser.parse(query)
        if not parsed["ok"]:
            return {
                "ok": False,
                "reply": parsed["error"],
                "intent": "invalid",
                "agents_used": ["query_parser"],
                "data": {},
                "parsed": parsed,
            }

        intent = parsed.get("intent", "unknown")
        try:
            router = {
                "current_price": self._handle_current_price,
                "prediction": self._handle_prediction,
                "trend": self._handle_trend,
                "recommendation": self._handle_recommendation,
                "compare_markets": self._handle_compare_markets,
                "compare_crops": self._handle_compare_crops,
                "highest_price": self._handle_highest_price,
                "market_info": self._handle_market_info,
                "help": self._handle_help,
            }
            handler = router.get(intent, self._handle_unknown)
            result = handler(parsed)
        except Exception as exc:  # last-resort guard: the chat never dies
            result = {
                "ok": False,
                "reply": (
                    "Something went wrong while looking that up. Please try rephrasing, "
                    f"for example: 'price of wheat in Pune'. (Details: {exc})"
                ),
                "agents_used": [],
                "data": {},
            }

        result.setdefault("intent", intent)
        result["parsed"] = parsed
        if parsed.get("notes"):
            result["reply"] = result["reply"] + "\n\nℹ️ " + " ".join(parsed["notes"])
        return result

    # ------------------------------------------------------------- handlers
    def _handle_current_price(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        crop = parsed.get("crop")
        if not crop:
            return self._need_crop()
        price = self.market_agent.get_current_price(crop, parsed.get("market"))
        if not price["ok"]:
            return self._failure(price, ["market_agent"])

        trend = self.trend_agent.analyze(price["crop"], price["market"])
        lines = [
            f"🌾 {price['crop']} — {price['market']} ({price['state']})",
            "",
            f"Current Price: {format_currency(price['modal_price'])}",
        ]
        if price.get("price_range"):
            lines.append(f"Price Range Today: {price['price_range']}")
        lines.append(f"As of: {price['date']}")
        if trend["ok"]:
            lines += [
                "",
                f"Current Trend: {trend_emoji(trend['direction'])} "
                f"{trend['direction'].title()} ({format_pct(trend['change_pct'])} in 30 days)",
            ]
        return {
            "ok": True,
            "reply": "\n".join(lines),
            "agents_used": ["query_parser", "market_agent", "trend_agent"],
            "data": {"price": price, "trend": trend if trend["ok"] else None},
        }

    def _handle_prediction(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        crop = parsed.get("crop")
        if not crop:
            return self._need_crop()
        days = parsed.get("days") or 7
        prediction = self.prediction_agent.predict(crop, parsed.get("market"), days=days)
        if not prediction["ok"]:
            return self._failure(prediction, ["prediction_agent"])

        market = prediction["market"]
        price = self.market_agent.get_current_price(prediction["crop"], market)
        trend = self.trend_agent.analyze(prediction["crop"], market)
        recommendation = self.recommendation_agent.recommend(
            prediction["crop"], market, days=days, price=price, trend=trend, prediction=prediction
        )

        lines = [f"🌾 {prediction['crop']} Price Prediction — {market}", ""]
        if price["ok"]:
            lines.append(f"Current Price: {format_currency(price['modal_price'])}")
        if trend["ok"]:
            lines.append(
                f"Current Trend: {trend_emoji(trend['direction'])} {trend['direction'].title()}"
            )
        lines += ["", f"Predicted Price (next {humanize_days(prediction['horizon_days'])}):"]
        for point in prediction["checkpoints"]:
            lines.append(
                f"  Day {point['day']} ({point['date']}): "
                f"{format_currency(point['predicted_price'], None)}"
            )
        lines += [
            "",
            "Insight:",
            f"  {self._insight_line(prediction)}",
        ]
        if recommendation["ok"]:
            lines += ["", "Recommendation:", f"  {self._recommendation_text(recommendation)}"]
        lines += ["", PREDICTION_DISCLAIMER]

        return {
            "ok": True,
            "reply": "\n".join(lines),
            "agents_used": [
                "query_parser",
                "prediction_agent",
                "market_agent",
                "trend_agent",
                "recommendation_agent",
            ],
            "data": {
                "price": price if price["ok"] else None,
                "trend": trend if trend["ok"] else None,
                "prediction": prediction,
                "recommendation": recommendation if recommendation["ok"] else None,
            },
        }

    def _handle_trend(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        crop = parsed.get("crop")
        if not crop:
            return self._need_crop()
        window = self._window_from_dates(parsed)
        trend = self.trend_agent.analyze(crop, parsed.get("market"), window=window)
        if not trend["ok"]:
            return self._failure(trend, ["trend_agent"])

        lines = [
            f"📊 {trend['crop']} Price Trend — {trend['market']}",
            "",
            f"Trend: {trend_emoji(trend['direction'])} {trend['direction'].title()} "
            f"({format_pct(trend['change_pct'])} over {trend['window_days']} days)",
            f"Price on {trend['start_date']}: {format_currency(trend['start_price'], None)}",
            f"Price on {trend['end_date']}: {format_currency(trend['end_price'])}",
            "",
            "Insights:",
        ]
        lines += [f"  • {insight}" for insight in trend["insights"]]
        return {
            "ok": True,
            "reply": "\n".join(lines),
            "agents_used": ["query_parser", "trend_agent"],
            "data": {"trend": trend},
        }

    def _handle_recommendation(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        crop = parsed.get("crop")
        if not crop:
            return self._need_crop()
        days = parsed.get("days") or 7
        recommendation = self.recommendation_agent.recommend(
            crop, parsed.get("market"), days=days
        )
        if not recommendation["ok"]:
            return self._failure(recommendation, ["recommendation_agent"])

        price = recommendation["price"]
        trend = recommendation.get("trend") or {}
        prediction = recommendation.get("prediction") or {}
        lines = [
            f"🧑‍🌾 Should you sell {recommendation['crop']} in {recommendation['market']}?",
            "",
            f"Current Price: {format_currency(price['modal_price'])}",
        ]
        if trend:
            lines.append(
                f"Current Trend: {trend_emoji(trend.get('direction'))} "
                f"{str(trend.get('direction', '')).title()} "
                f"({format_pct(trend.get('change_pct'))} in 30 days)"
            )
        if prediction:
            lines.append(
                f"Forecast ({humanize_days(prediction['horizon_days'])}): "
                f"{format_currency(prediction['predicted_final_price'])} "
                f"({format_pct(prediction['expected_change_pct'])})"
            )
        lines += [
            "",
            f"Suggestion: {recommendation['action_label']}",
            "",
            self._recommendation_text(recommendation),
            "",
            "Why:",
        ]
        lines += [f"  • {reason}" for reason in recommendation["reasons"]]
        lines += ["", recommendation["disclaimer"]]
        return {
            "ok": True,
            "reply": "\n".join(lines),
            "agents_used": [
                "query_parser",
                "market_agent",
                "trend_agent",
                "prediction_agent",
                "recommendation_agent",
            ],
            "data": {
                "price": price,
                "trend": trend or None,
                "prediction": prediction or None,
                "recommendation": recommendation,
            },
        }

    def _handle_compare_markets(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        crop = parsed.get("crop")
        if not crop:
            return self._need_crop()
        comparison = self.market_agent.get_market_comparison(crop)
        if not comparison["ok"]:
            return self._failure(comparison, ["market_agent"])

        lines = [f"🏪 {comparison['crop']} — Market Comparison", ""]
        lines.append(f"{'Market':<14}{'Price':>14}   Trend")
        lines.append("-" * 40)
        for row in comparison["markets"]:
            lines.append(
                f"{row['market']:<14}{format_currency(row['modal_price'], None):>14}   "
                f"{trend_emoji(row.get('trend'))} {str(row.get('trend') or '').title()}"
            )
        lines += [
            "",
            f"✅ Best price: {comparison['best_market']} at "
            f"{format_currency(comparison['best_price'])}.",
            f"Lowest: {comparison['lowest_market']} at "
            f"{format_currency(comparison['lowest_price'])} "
            f"(gap of {format_currency(comparison['price_gap'], None)}).",
            "",
            "Check transport cost before moving your produce to another mandi.",
        ]
        return {
            "ok": True,
            "reply": "\n".join(lines),
            "agents_used": ["query_parser", "market_agent"],
            "data": {"comparison": comparison},
        }

    def _handle_compare_crops(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        crops = parsed.get("crops") or []
        if len(crops) < 2:
            return self._need_crop("Name two crops to compare, e.g. 'compare wheat and rice'.")
        comparison = self.api.compare_crops(crops, parsed.get("market"))
        if not comparison["ok"]:
            return self._failure(comparison, ["market_agent"])

        lines = ["⚖️ Crop Price Comparison", ""]
        lines.append(f"{'Crop':<12}{'Market':<14}{'Price':>14}   Trend")
        lines.append("-" * 52)
        for row in comparison["rows"]:
            lines.append(
                f"{row['crop']:<12}{row['market']:<14}"
                f"{format_currency(row['modal_price'], None):>14}   "
                f"{trend_emoji(row.get('trend'))} {str(row.get('trend') or '').title()}"
            )
        highest, lowest = comparison["highest"], comparison["lowest"]
        lines += [
            "",
            f"Highest: {highest['crop']} at {format_currency(highest['modal_price'])} "
            f"({highest['market']}).",
            f"Lowest: {lowest['crop']} at {format_currency(lowest['modal_price'])} "
            f"({lowest['market']}).",
            "",
            "Remember that prices per quintal are not directly comparable as income — "
            "yield per acre and input cost differ by crop.",
        ]
        return {
            "ok": True,
            "reply": "\n".join(lines),
            "agents_used": ["query_parser", "market_agent", "trend_agent"],
            "data": {"crop_comparison": comparison},
        }

    def _handle_highest_price(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        result = self.market_agent.highest_priced_crop(limit=10)
        if not result["ok"]:
            return self._failure(result, ["market_agent"])
        lines = ["🏆 Highest Priced Crops Right Now", ""]
        lines.append(f"{'Crop':<12}{'Avg Price':>13}{'Best Mandi':>14}")
        lines.append("-" * 40)
        for row in result["rows"]:
            lines.append(
                f"{row['crop']:<12}{format_currency(row['average_price'], None):>13}"
                f"{row['best_market']:>14}"
            )
        lines += ["", result["summary"]]
        return {
            "ok": True,
            "reply": "\n".join(lines),
            "agents_used": ["query_parser", "market_agent"],
            "data": {"top_crops": result},
        }

    def _handle_market_info(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        crop = parsed.get("crop")
        crops = self.api.crops()
        markets = self.api.markets(crop) if crop else self.api.markets()
        lines = ["📋 What I have data for", ""]
        lines.append("Crops: " + ", ".join(crops))
        lines.append(("Markets for " + crop if crop else "Markets") + ": " + ", ".join(markets))
        return {
            "ok": True,
            "reply": "\n".join(lines),
            "agents_used": ["query_parser", "market_agent"],
            "data": {"crops": crops, "markets": markets},
        }

    def _handle_help(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "reply": HELP_TEXT, "agents_used": ["query_parser"], "data": {}}

    def _handle_unknown(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unrecognised question: if a crop was detected, default to the most useful
        answer (price + trend). Otherwise show what I can do.
        """
        if parsed.get("crop"):
            return self._handle_current_price(parsed)
        crops = ", ".join(self.api.crops()[:6])
        return {
            "ok": True,
            "reply": (
                "I didn't catch which crop you mean.\n\n"
                f"I have prices for: {crops} and more.\n\n" + HELP_TEXT
            ),
            "agents_used": ["query_parser"],
            "data": {},
        }

    # -------------------------------------------------------------- helpers
    def _need_crop(self, message: Optional[str] = None) -> Dict[str, Any]:
        """
        No crop detected: answer helpfully rather than with an error.

        The chatbot should always leave the user with a next step, so this lists
        what data exists and shows example questions.
        """
        crops = ", ".join(self.api.crops()[:8])
        lead = message or "I didn't catch which crop you mean."
        return {
            "ok": True,
            "reply": f"{lead}\n\nI have prices for: {crops} and more.\n\n{HELP_TEXT}",
            "agents_used": ["query_parser"],
            "data": {},
        }

    @staticmethod
    def _failure(result: Dict[str, Any], agents: List[str]) -> Dict[str, Any]:
        return {
            "ok": False,
            "reply": result.get("error", "I could not find that information."),
            "agents_used": ["query_parser"] + agents,
            "data": {},
            "code": result.get("code"),
        }

    @staticmethod
    def _window_from_dates(parsed: Dict[str, Any]) -> int:
        """Convert a parsed date range into a look-back window in days."""
        import datetime as _dt

        date_from, date_to = parsed.get("date_from"), parsed.get("date_to")
        if date_from and date_to:
            try:
                start = _dt.date.fromisoformat(str(date_from)[:10])
                end = _dt.date.fromisoformat(str(date_to)[:10])
                span = (end - start).days
                if span >= 3:
                    return min(span, 365)
            except ValueError:
                pass
        return int(parsed.get("days") or 30)

    def _insight_line(self, prediction: Dict[str, Any]) -> str:
        direction = prediction.get("predicted_direction", "stable")
        change = prediction.get("expected_change_pct") or 0
        size = "moderate" if abs(change) >= 2 else "slight"
        if direction == "increasing":
            return (
                f"The model predicts a {size} upward move of {format_pct(change)} over the "
                f"next {humanize_days(prediction['horizon_days'])}."
            )
        if direction == "decreasing":
            return (
                f"The model predicts a {size} downward move of {format_pct(change)} over the "
                f"next {humanize_days(prediction['horizon_days'])}."
            )
        return (
            f"The model predicts broadly stable prices ({format_pct(change)}) over the next "
            f"{humanize_days(prediction['horizon_days'])}."
        )

    def _recommendation_text(self, recommendation: Dict[str, Any]) -> str:
        """Gemini-polished wording when available, deterministic text otherwise."""
        if self.use_gemini and self.gemini.is_available():
            return self.recommendation_agent.explain(recommendation)
        return recommendation.get("recommendation", "")

    # ------------------------------------------------------------- dashboard
    def dashboard_payload(
        self, crop: str, market: Optional[str] = None, days: int = 7
    ) -> Dict[str, Any]:
        """
        Everything the Streamlit dashboard needs in one call:
        price, trend, forecast, market comparison, recommendation, history.
        """
        try:
            resolved = self.api.resolve(crop, market)
        except Exception as exc:
            return err(str(exc), code=getattr(exc, "code", "error"))

        crop_name, market_name = resolved["crop"], resolved["market"]
        price = self.market_agent.get_current_price(crop_name, market_name)
        trend = self.trend_agent.analyze(crop_name, market_name)
        prediction = self.prediction_agent.predict(crop_name, market_name, days=days)
        history = self.api.get_history(crop_name, market_name, days=365)
        comparison = self.market_agent.get_market_comparison(crop_name)
        recommendation = self.recommendation_agent.recommend(
            crop_name, market_name, days=days, price=price, trend=trend, prediction=prediction
        )
        return ok(
            crop=crop_name,
            market=market_name,
            price=price if price["ok"] else None,
            trend=trend if trend["ok"] else None,
            prediction=prediction if prediction["ok"] else None,
            history=history if history["ok"] else None,
            comparison=comparison if comparison["ok"] else None,
            recommendation=recommendation if recommendation["ok"] else None,
            errors=[
                r["error"]
                for r in (price, trend, prediction, history, comparison, recommendation)
                if not r.get("ok")
            ],
        )

    def health(self) -> Dict[str, Any]:
        """Data + model + Gemini status for the sidebar."""
        status = self.api.health()
        status["gemini"] = self.gemini.status()
        return status
