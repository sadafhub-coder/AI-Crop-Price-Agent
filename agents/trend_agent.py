"""
agents/trend_agent.py
---------------------
Trend Agent — reads the recent history of a crop+market and says whether prices
are rising, falling or flat, by how much, and what that implies.

All maths lives in tools.market_tools.compute_trend_stats; this agent turns the
numbers into insight lines a farmer can act on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.market_api import MarketAPI, get_market_api
from utils.gemini import GeminiClient, get_gemini_client, load_prompt
from utils.helpers import (
    TREND_DOWN,
    TREND_FLAT,
    TREND_UP,
    err,
    format_currency,
    format_pct,
    ok,
    trend_emoji,
)


class TrendAgent:
    """Historical trend analysis and plain-language insights."""

    name = "trend_agent"

    def __init__(self, api: Optional[MarketAPI] = None, gemini: Optional[GeminiClient] = None):
        self.api = api or get_market_api()
        self.gemini = gemini or get_gemini_client()

    # ---------------------------------------------------------------- public
    def analyze(
        self, crop: str, market: Optional[str] = None, window: int = 30
    ) -> Dict[str, Any]:
        """Trend stats + insights for one crop+market over `window` days."""
        if not crop:
            return err("Which crop's trend should I check?", code="missing_crop")

        result = self.api.get_trend(crop, market, window=window)
        if not result["ok"]:
            return result

        insights = self._build_insights(result)
        summary = (
            f"{result['crop']} in {result['market']} is "
            f"{trend_emoji(result['direction'])} {result['direction']} — "
            f"{format_pct(result['change_pct'])} over the last {result['window_days']} days "
            f"({format_currency(result['start_price'], None)} → "
            f"{format_currency(result['end_price'])})."
        )
        return ok(**result, insights=insights, summary=summary, agent=self.name)

    def compare_windows(
        self, crop: str, market: Optional[str] = None, windows: tuple = (7, 30, 90)
    ) -> Dict[str, Any]:
        """Trend over several look-back windows — short vs long term picture."""
        rows: List[Dict[str, Any]] = []
        for window in windows:
            result = self.api.get_trend(crop, market, window=window)
            if result["ok"]:
                rows.append(
                    {
                        "window_days": window,
                        "direction": result["direction"],
                        "change_pct": result["change_pct"],
                        "average_price": result["average_price"],
                        "volatility_pct": result["volatility_pct"],
                    }
                )
        if not rows:
            return err(f"Not enough history to analyse {crop}.", code="no_data")
        return ok(crop=crop, market=market, windows=rows, agent=self.name)

    # ------------------------------------------------------------- narration
    def _build_insights(self, stats: Dict[str, Any]) -> List[str]:
        """Deterministic insight lines — always available, Gemini or not."""
        insights: List[str] = []
        direction = stats.get("direction")
        change = stats.get("change_pct")
        volatility = stats.get("volatility_pct") or 0

        if direction == TREND_UP:
            insights.append(
                f"Prices have climbed {format_pct(change)} in {stats['window_days']} days "
                f"(about {format_pct(stats.get('slope_pct_per_week'))} per week)."
            )
        elif direction == TREND_DOWN:
            insights.append(
                f"Prices have slipped {format_pct(change)} in {stats['window_days']} days "
                f"(about {format_pct(stats.get('slope_pct_per_week'))} per week)."
            )
        else:
            insights.append(
                f"Prices have been broadly flat ({format_pct(change)}) over the last "
                f"{stats['window_days']} days."
            )

        week = stats.get("week_over_week_pct")
        month = stats.get("month_over_month_pct")
        if week is not None:
            insights.append(f"Week-on-week change: {format_pct(week)}.")
        if month is not None:
            insights.append(f"Month-on-month change: {format_pct(month)}.")

        if volatility >= 8:
            insights.append(
                f"This market is volatile right now (price swing of about {volatility:.1f}%), "
                "so day-to-day rates can move sharply."
            )
        elif volatility <= 3:
            insights.append(f"Prices are steady (swing of only about {volatility:.1f}%).")

        insights.append(
            f"Recent range: {format_currency(stats.get('lowest_price'), None)} – "
            f"{format_currency(stats.get('highest_price'))} (average "
            f"{format_currency(stats.get('average_price'))})."
        )

        current = stats.get("end_price")
        average = stats.get("average_price")
        if current and average:
            if current > average * 1.03:
                insights.append("Today's rate is above the recent average.")
            elif current < average * 0.97:
                insights.append("Today's rate is below the recent average.")
        return insights

    def explain(self, trend_result: Dict[str, Any]) -> str:
        """Gemini-written trend narrative with a deterministic fallback."""
        if not trend_result.get("ok"):
            return trend_result.get("error", "Trend unavailable.")
        fallback = " ".join(
            [trend_result.get("summary", "")] + list(trend_result.get("insights", []))
        ).strip()
        prompt = load_prompt(
            "trend_prompt",
            crop=trend_result.get("crop", ""),
            market=trend_result.get("market", ""),
            window_days=trend_result.get("window_days", ""),
            direction=trend_result.get("direction", ""),
            change_pct=trend_result.get("change_pct", ""),
            start_price=trend_result.get("start_price", ""),
            end_price=trend_result.get("end_price", ""),
            average_price=trend_result.get("average_price", ""),
            highest_price=trend_result.get("highest_price", ""),
            lowest_price=trend_result.get("lowest_price", ""),
            volatility_pct=trend_result.get("volatility_pct", ""),
            week_over_week_pct=trend_result.get("week_over_week_pct", ""),
            month_over_month_pct=trend_result.get("month_over_month_pct", ""),
        )
        if not prompt:
            return fallback
        response = self.gemini.generate(prompt, temperature=0.35, max_output_tokens=400)
        return response.text.strip() if response.ok else fallback

    @staticmethod
    def direction_label(direction: Optional[str]) -> str:
        return {
            TREND_UP: "📈 Increasing",
            TREND_DOWN: "📉 Decreasing",
            TREND_FLAT: "➡️ Stable",
        }.get((direction or "").lower(), "➡️ Stable")
