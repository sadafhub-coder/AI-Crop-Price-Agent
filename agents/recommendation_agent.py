"""
agents/recommendation_agent.py
------------------------------
Recommendation Agent — combines price + trend + forecast into a plain-language
"sell now / wait / watch" suggestion.

Hard rule: it never guarantees an outcome. Every recommendation is framed as a
consideration, carries the reasoning that produced it, and ends with a
disclaimer. Scoring is deterministic (auditable); Gemini only rewrites the
wording, and if Gemini is unavailable the template text is used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.prediction_agent import PredictionAgent
from agents.trend_agent import TrendAgent
from tools.market_api import MarketAPI, get_market_api
from utils.gemini import GeminiClient, get_gemini_client, load_prompt
from utils.helpers import (
    TREND_DOWN,
    TREND_UP,
    err,
    format_currency,
    format_pct,
    ok,
)

DISCLAIMER = (
    "⚠️ This is guidance based on historical price patterns, not a guaranteed "
    "outcome. Please also consider your storage cost, crop quality, loan dues and "
    "cash needs before deciding."
)

ACTIONS = {
    "wait": "Consider waiting",
    "sell": "Consider selling soon",
    "watch": "Monitor the market",
}


class RecommendationAgent:
    """Turns market facts into an understandable, non-guaranteed suggestion."""

    name = "recommendation_agent"

    def __init__(
        self,
        api: Optional[MarketAPI] = None,
        trend_agent: Optional[TrendAgent] = None,
        prediction_agent: Optional[PredictionAgent] = None,
        gemini: Optional[GeminiClient] = None,
    ):
        self.api = api or get_market_api()
        self.trend_agent = trend_agent or TrendAgent(self.api)
        self.prediction_agent = prediction_agent or PredictionAgent(self.api)
        self.gemini = gemini or get_gemini_client()

    # ---------------------------------------------------------------- public
    def recommend(
        self,
        crop: str,
        market: Optional[str] = None,
        days: int = 7,
        price: Optional[Dict[str, Any]] = None,
        trend: Optional[Dict[str, Any]] = None,
        prediction: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a recommendation. Pre-computed price/trend/prediction results can be
        passed in (the supervisor does this to avoid duplicate work).
        """
        if not crop:
            return err("Which crop do you want advice on?", code="missing_crop")

        price = price or self.api.get_price(crop, market)
        if not price["ok"]:
            return price
        market = market or price["market"]

        trend = trend or self.trend_agent.analyze(crop, market)
        prediction = prediction or self.prediction_agent.predict(crop, market, days=days)

        score, reasons = self._score(price, trend, prediction)
        action = "wait" if score >= 1.5 else "sell" if score <= -1.5 else "watch"
        best_market = self._best_market_hint(crop, market)
        if best_market:
            reasons.append(best_market["reason"])

        text = self._template_text(action, price, trend, prediction, best_market)
        return ok(
            crop=price["crop"],
            market=market,
            action=action,
            action_label=ACTIONS[action],
            score=round(score, 2),
            reasons=reasons,
            recommendation=text,
            best_market=best_market["market"] if best_market else None,
            best_market_price=best_market["price"] if best_market else None,
            price=price,
            trend=trend if trend.get("ok") else None,
            prediction=prediction if prediction.get("ok") else None,
            disclaimer=DISCLAIMER,
            agent=self.name,
        )

    # --------------------------------------------------------------- scoring
    @staticmethod
    def _score(
        price: Dict[str, Any], trend: Dict[str, Any], prediction: Dict[str, Any]
    ) -> tuple:
        """
        Positive score -> waiting looks reasonable; negative -> selling sooner does.

        Signals: forecast direction/size (strongest), recent trend, position of
        today's price inside the recent range, and volatility (a penalty on
        waiting, because a volatile market can reverse quickly).
        """
        score = 0.0
        reasons: List[str] = []

        if prediction.get("ok"):
            change = prediction.get("expected_change_pct") or 0.0
            if change >= 3:
                score += 2.0
                reasons.append(
                    f"The model expects prices about {format_pct(change)} higher over the "
                    f"next {prediction['horizon_days']} days."
                )
            elif change >= 1:
                score += 1.0
                reasons.append(f"A mild rise of {format_pct(change)} is expected.")
            elif change <= -3:
                score -= 2.0
                reasons.append(
                    f"The model expects prices about {format_pct(change)} lower over the "
                    f"next {prediction['horizon_days']} days."
                )
            elif change <= -1:
                score -= 1.0
                reasons.append(f"A mild fall of {format_pct(change)} is expected.")
            else:
                reasons.append("The forecast is broadly flat.")
            if prediction.get("method") == "trend_fallback":
                score *= 0.6  # weaker evidence, so soften its influence
                reasons.append("Forecast confidence is lower (statistical fallback used).")

        if trend.get("ok"):
            direction = trend.get("direction")
            change = trend.get("change_pct") or 0.0
            if direction == TREND_UP:
                score += 1.0
                reasons.append(f"Recent trend is rising ({format_pct(change)} in 30 days).")
            elif direction == TREND_DOWN:
                score -= 1.0
                reasons.append(f"Recent trend is falling ({format_pct(change)} in 30 days).")
            else:
                reasons.append("Recent prices have been broadly stable.")

            current = price.get("modal_price")
            high = trend.get("highest_price")
            low = trend.get("lowest_price")
            if current and high and low and high > low:
                position = (current - low) / (high - low)
                if position >= 0.85:
                    score -= 1.0
                    reasons.append("Today's price is near the top of its recent range.")
                elif position <= 0.2:
                    score += 0.5
                    reasons.append("Today's price is near the bottom of its recent range.")

            volatility = trend.get("volatility_pct") or 0
            if volatility >= 10:
                score -= 0.5
                reasons.append(
                    f"The market is volatile ({volatility:.1f}% swing), so waiting carries risk."
                )
        return score, reasons

    def _best_market_hint(self, crop: str, market: Optional[str]) -> Optional[Dict[str, Any]]:
        """Flag a materially better mandi (>2% premium) for the same crop."""
        comparison = self.api.compare_markets(crop)
        if not comparison["ok"]:
            return None
        best = comparison["markets"][0]
        if not market or best["market"] == market:
            return None
        current = next(
            (row for row in comparison["markets"] if row["market"] == market), None
        )
        if not current or not current["modal_price"] or not best["modal_price"]:
            return None
        premium = (best["modal_price"] - current["modal_price"]) / current["modal_price"] * 100
        if premium < 2:
            return None
        return {
            "market": best["market"],
            "price": best["modal_price"],
            "premium_pct": round(premium, 2),
            "reason": (
                f"{best['market']} is currently paying {format_currency(best['modal_price'])}, "
                f"about {premium:.1f}% more than {market} — worth checking transport cost against that gap."
            ),
        }

    # -------------------------------------------------------------- narration
    def _template_text(
        self,
        action: str,
        price: Dict[str, Any],
        trend: Dict[str, Any],
        prediction: Dict[str, Any],
        best_market: Optional[Dict[str, Any]],
    ) -> str:
        current = format_currency(price.get("modal_price"))
        direction = trend.get("direction") if trend.get("ok") else "unclear"
        if action == "wait":
            body = (
                f"Prices for {price['crop']} in {price['market']} are {direction} and the "
                f"forecast points higher, so if you are not under immediate selling pressure, "
                f"monitoring the market for a few more days may be worth considering."
            )
        elif action == "sell":
            body = (
                f"Prices for {price['crop']} in {price['market']} are {direction} and the "
                f"forecast points lower, so selling sooner rather than holding may be worth "
                f"considering — especially if storage is costly for you."
            )
        else:
            # "watch" can mean genuinely flat OR conflicting signals (e.g. a
            # falling recent trend against a flat forecast) — say which it is.
            forecast_change = (
                prediction.get("expected_change_pct") if prediction.get("ok") else None
            )
            if direction in (TREND_UP, TREND_DOWN) and forecast_change is not None:
                body = (
                    f"Recent prices for {price['crop']} in {price['market']} are {direction}, but "
                    f"the forecast for the coming days is close to flat "
                    f"({format_pct(forecast_change)}), so the signals do not point clearly either "
                    f"way. Watching the market for a few days before deciding may be reasonable."
                )
            else:
                body = (
                    f"Prices for {price['crop']} in {price['market']} look broadly steady, so there "
                    f"is no clear advantage either way right now. Watching the market for a few days "
                    f"before deciding may be reasonable."
                )
        parts = [f"Current price: {current}.", body]
        if best_market:
            parts.append(best_market["reason"])
        return " ".join(parts)

    def explain(self, result: Dict[str, Any]) -> str:
        """Gemini rewrite of the recommendation; template text is the fallback."""
        if not result.get("ok"):
            return result.get("error", "Recommendation unavailable.")
        fallback = result.get("recommendation", "")
        trend = result.get("trend") or {}
        prediction = result.get("prediction") or {}
        prompt = load_prompt(
            "recommendation_prompt",
            crop=result.get("crop", ""),
            market=result.get("market", ""),
            current_price=(result.get("price") or {}).get("modal_price", ""),
            trend_direction=trend.get("direction", "unknown"),
            trend_change_pct=trend.get("change_pct", ""),
            volatility_pct=trend.get("volatility_pct", ""),
            horizon_days=prediction.get("horizon_days", ""),
            predicted_price=prediction.get("predicted_final_price", ""),
            predicted_change_pct=prediction.get("expected_change_pct", ""),
            action=result.get("action", ""),
            reasons="; ".join(result.get("reasons", [])),
            best_market=result.get("best_market") or "none",
        )
        if not prompt:
            return fallback
        response = self.gemini.generate(prompt, temperature=0.4, max_output_tokens=450)
        return response.text.strip() if response.ok else fallback
