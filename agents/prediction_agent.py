"""
agents/prediction_agent.py
--------------------------
Prediction Agent — the ML-facing agent.

It validates the requested horizon (7 / 15 / 30 / 90 days, and anything in
between), asks MarketAPI for a forecast, and formats it the way a farmer reads
it: a few checkpoint days rather than 90 rows of numbers — always with the
"estimate, not a guarantee" framing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.market_api import MarketAPI, get_market_api
from utils.gemini import GeminiClient, get_gemini_client, load_prompt
from utils.helpers import (
    VALID_HORIZONS,
    classify_trend,
    err,
    format_currency,
    format_pct,
    ok,
    humanize_days,
)

DISCLAIMER = (
    "⚠️ Prediction is based on historical data and is not guaranteed."
)


class PredictionAgent:
    """Future price estimates for a crop+market."""

    name = "prediction_agent"

    def __init__(self, api: Optional[MarketAPI] = None, gemini: Optional[GeminiClient] = None):
        self.api = api or get_market_api()
        self.gemini = gemini or get_gemini_client()

    # ---------------------------------------------------------------- public
    def predict(
        self, crop: str, market: Optional[str] = None, days: int = 7
    ) -> Dict[str, Any]:
        """Forecast `days` ahead. Returns ok=False with a friendly message on failure."""
        if not crop:
            return err("Which crop should I predict?", code="missing_crop")

        result = self.api.get_prediction(crop, market, days=days)
        if not result["ok"]:
            return result

        predictions: List[Dict[str, Any]] = result["predictions"]
        checkpoints = self.checkpoints(predictions)
        change_pct = result.get("expected_change_pct")
        direction = classify_trend(change_pct, flat_band=1.0)

        summary = (
            f"Over the next {humanize_days(result['horizon_days'])}, {result['crop']} in "
            f"{result['market']} is estimated to move from "
            f"{format_currency(result['last_actual_price'])} to about "
            f"{format_currency(result['predicted_final_price'])} "
            f"({format_pct(change_pct)})."
        )

        # `result` already carries a disclaimer from the predictor; the agent's
        # wording wins, so merge instead of passing duplicate keywords.
        payload = dict(result)
        payload.update(
            {
                "checkpoints": checkpoints,
                "predicted_direction": direction,
                "summary": summary,
                "confidence_note": self._confidence_note(result),
                "disclaimer": DISCLAIMER,
                "agent": self.name,
            }
        )
        return ok(**payload)

    def predict_multi_horizon(
        self, crop: str, market: Optional[str] = None, horizons: tuple = VALID_HORIZONS
    ) -> Dict[str, Any]:
        """One row per supported horizon — the "how far out?" view."""
        rows: List[Dict[str, Any]] = []
        problems: List[str] = []
        for horizon in horizons:
            result = self.predict(crop, market, days=horizon)
            if result["ok"]:
                rows.append(
                    {
                        "horizon_days": horizon,
                        "predicted_price": result["predicted_final_price"],
                        "expected_change_pct": result["expected_change_pct"],
                        "direction": result["predicted_direction"],
                        "method": result["method"],
                    }
                )
            else:
                problems.append(result["error"])
        if not rows:
            return err(problems[0] if problems else "No forecast available.", code="no_data")
        return ok(crop=crop, market=market, horizons=rows, problems=problems, agent=self.name)

    # ------------------------------------------------------------- utilities
    @staticmethod
    def checkpoints(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Pick readable checkpoint days (1, 3, 7, 15, 30, 60, 90 where they exist)
        plus the final day, so a 90-day forecast stays a 6-line answer.
        """
        if not predictions:
            return []
        wanted = [1, 3, 7, 15, 30, 60, 90]
        by_day = {p["day"]: p for p in predictions}
        picked = [by_day[d] for d in wanted if d in by_day]
        last = predictions[-1]
        if last not in picked:
            picked.append(last)
        return picked

    @staticmethod
    def _confidence_note(result: Dict[str, Any]) -> str:
        if result.get("method") == "trend_fallback":
            return (
                "The trained model was not available, so this is a simple trend "
                "estimate from recent prices."
            )
        mae = result.get("model_mae")
        horizon = result.get("horizon_days", 0)
        base = "Estimates come from a Random Forest model trained on historical mandi prices."
        if mae:
            base += f" Typical error on unseen data is about {format_currency(mae, None)} per quintal."
        if horizon and horizon > 30:
            base += " Accuracy falls the further out the forecast goes."
        return base

    def format_reply(self, result: Dict[str, Any]) -> str:
        """The farmer-facing block used in the chat answer."""
        if not result.get("ok"):
            return result.get("error", "Prediction unavailable.")
        lines = [
            f"🌾 {result['crop']} Price Prediction — {result['market']}",
            "",
            f"Current Price: {format_currency(result['last_actual_price'])} "
            f"(as of {result['last_actual_date']})",
            f"Predicted Price ({humanize_days(result['horizon_days'])}):",
        ]
        for point in result["checkpoints"]:
            lines.append(
                f"  Day {point['day']} ({point['date']}): "
                f"{format_currency(point['predicted_price'], None)}"
            )
        lines += [
            "",
            f"Expected change: {format_pct(result['expected_change_pct'])} "
            f"({format_currency(result['expected_change'], None)}/quintal)",
            "",
            result["confidence_note"],
            DISCLAIMER,
        ]
        return "\n".join(lines)

    def explain(self, result: Dict[str, Any]) -> str:
        """Gemini narrative for the forecast, with a deterministic fallback."""
        if not result.get("ok"):
            return result.get("error", "Prediction unavailable.")
        fallback = f"{result.get('summary', '')} {result.get('confidence_note', '')}".strip()
        checkpoint_text = "; ".join(
            f"day {p['day']} ≈ ₹{p['predicted_price']:.0f}" for p in result.get("checkpoints", [])
        )
        prompt = load_prompt(
            "prediction_prompt",
            crop=result.get("crop", ""),
            market=result.get("market", ""),
            horizon_days=result.get("horizon_days", ""),
            current_price=result.get("last_actual_price", ""),
            predicted_price=result.get("predicted_final_price", ""),
            change_pct=result.get("expected_change_pct", ""),
            direction=result.get("predicted_direction", ""),
            checkpoints=checkpoint_text,
            method=result.get("method", ""),
            model_mae=result.get("model_mae", ""),
        )
        if not prompt:
            return fallback
        response = self.gemini.generate(prompt, temperature=0.35, max_output_tokens=400)
        return response.text.strip() if response.ok else fallback
