"""
agents/market_agent.py
----------------------
Market Price Agent — answers "what is the price right now?".

Wraps MarketAPI with farmer-friendly text, multi-crop / multi-market support and
graceful handling of missing data (never a stack trace, always a next step).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from tools.market_api import MarketAPI, get_market_api
from utils.gemini import GeminiClient, get_gemini_client, load_prompt
from utils.helpers import err, format_currency, format_date, ok


class MarketPriceAgent:
    """Current/latest prices for one or many crops and markets."""

    name = "market_agent"

    def __init__(self, api: Optional[MarketAPI] = None, gemini: Optional[GeminiClient] = None):
        self.api = api or get_market_api()
        self.gemini = gemini or get_gemini_client()

    # ---------------------------------------------------------------- single
    def get_current_price(self, crop: str, market: Optional[str] = None) -> Dict[str, Any]:
        """Latest price for a crop in one market (or the best-covered market)."""
        if not crop:
            return err(
                "Tell me which crop you mean, for example: 'price of wheat in Pune'.",
                code="missing_crop",
            )
        result = self.api.get_price(crop, market)
        if not result["ok"]:
            return result

        spread = None
        if result.get("price_min") and result.get("price_max"):
            spread = f"{format_currency(result['price_min'], None)} – {format_currency(result['price_max'], None)}"

        summary = (
            f"{result['crop']} in {result['market']} ({result['state']}) is "
            f"{format_currency(result['modal_price'])} as of {format_date(result['date'])}."
        )
        if result.get("market_defaulted"):
            summary += f" (No market given, so I used {result['market']}, the best-covered mandi.)"

        return ok(
            **result,
            price_range=spread,
            summary=summary,
            agent=self.name,
        )

    # ------------------------------------------------------------ many crops
    def get_prices(
        self, crops: Sequence[str], market: Optional[str] = None
    ) -> Dict[str, Any]:
        """Latest price for several crops (used by comparison queries)."""
        if not crops:
            return err("No crops given to price.", code="missing_crop")
        rows: List[Dict[str, Any]] = []
        problems: List[str] = []
        for crop in crops:
            single = self.get_current_price(crop, market)
            if single["ok"]:
                rows.append(single)
            else:
                problems.append(single["error"])
        if not rows:
            return err(problems[0] if problems else "No prices found.", code="no_data")
        return ok(rows=rows, problems=problems, agent=self.name)

    # ---------------------------------------------------------- all markets
    def get_market_comparison(
        self, crop: str, markets: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        """Latest price for a crop across markets, best price first."""
        if not crop:
            return err("Which crop should I compare across markets?", code="missing_crop")
        result = self.api.compare_markets(crop, markets)
        if not result["ok"]:
            return result
        rows = result["markets"]
        best, worst = rows[0], rows[-1]
        gap = (best["modal_price"] or 0) - (worst["modal_price"] or 0)
        summary = (
            f"Best price for {result['crop']} right now: {format_currency(best['modal_price'])} "
            f"in {best['market']}. Lowest: {format_currency(worst['modal_price'])} in "
            f"{worst['market']} — a gap of {format_currency(gap, None)} per quintal."
        )
        return ok(**result, price_gap=round(gap, 2), summary=summary, agent=self.name)

    # ----------------------------------------------------------- top earners
    def highest_priced_crop(self, limit: int = 5) -> Dict[str, Any]:
        """Which crops are fetching the most money right now."""
        result = self.api.highest_priced_crops(limit=limit)
        if not result["ok"]:
            return result
        rows = result["rows"]
        top = rows[0]
        summary = (
            f"{top['crop']} currently has the highest price at about "
            f"{format_currency(top['average_price'])} (best mandi: {top['best_market']} at "
            f"{format_currency(top['best_price'])})."
        )
        return ok(rows=rows, top=top, summary=summary, agent=self.name)

    # ------------------------------------------------------------- narration
    def explain(self, price_result: Dict[str, Any]) -> str:
        """
        Optional Gemini rewrite of the price summary in simple language.
        Falls back to the deterministic summary whenever Gemini is unavailable.
        """
        if not price_result.get("ok"):
            return price_result.get("error", "Price unavailable.")
        fallback = price_result.get("summary", "")
        prompt = load_prompt(
            "market_prompt",
            crop=price_result.get("crop", ""),
            market=price_result.get("market", ""),
            state=price_result.get("state", ""),
            date=price_result.get("date", ""),
            modal_price=price_result.get("modal_price", ""),
            price_min=price_result.get("price_min", ""),
            price_max=price_result.get("price_max", ""),
            arrival_quantity=price_result.get("arrival_quantity", ""),
        )
        if not prompt:
            return fallback
        response = self.gemini.generate(prompt, temperature=0.3, max_output_tokens=300)
        return response.text.strip() if response.ok else fallback

    # ------------------------------------------------------------- discovery
    def available_crops(self) -> List[str]:
        return self.api.crops()

    def available_markets(self, crop: Optional[str] = None) -> List[str]:
        return self.api.markets(crop)
