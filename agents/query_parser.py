"""
agents/query_parser.py
----------------------
Query Parser Agent — turns free-text farmer questions into structured intent.

Two-stage design:
  1. Gemini is asked for strict JSON (best natural-language coverage).
  2. A deterministic regex/keyword parser runs as a fallback AND as a repair
     layer — it fills anything Gemini left out and always runs when Gemini is
     unavailable, so the chatbot works with no API key at all.

Output shape (always the same keys):
    {
      ok, intent, crop, crops, market, days, date_from, date_to,
      confidence, source, raw_query, notes
    }
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from typing import Any, Dict, List, Optional

from utils.gemini import GeminiClient, get_gemini_client, load_prompt
from utils.helpers import EmptyQueryError, VALID_HORIZONS, err_from_exception, ok

INTENTS = (
    "current_price",
    "prediction",
    "trend",
    "recommendation",
    "compare_markets",
    "compare_crops",
    "highest_price",
    "market_info",
    "help",
    "unknown",
)

# Keyword → intent, checked in priority order (first match wins).
INTENT_KEYWORDS: List[tuple] = [
    ("recommendation", ("should i sell", "sell now", "sell or wait", "hold or sell",
                        "should i hold", "shall i sell", "worth selling", "advice",
                        "recommend", "kya karu", "best time to sell")),
    ("prediction", ("predict", "forecast", "future", "next week", "next month",
                    "next 7", "next 15", "next 30", "next 90", "will be", "expected price",
                    "how much will", "tomorrow", "coming days", "outlook")),
    ("compare_markets", ("which market", "best market", "compare market", "across markets",
                         "highest market", "market wise", "mandi comparison", "where should i sell")),
    ("compare_crops", ("compare", "versus", " vs ", "vs.", "difference between")),
    ("highest_price", ("highest price", "most expensive", "top crop", "costliest",
                       "which crop has the highest", "best paying crop", "lowest price",
                       "cheapest crop")),
    ("trend", ("trend", "trending", "going up", "going down", "rising", "falling",
               "history", "historical", "last month", "past month", "past week",
               "over time", "chart", "movement", "pattern")),
    ("current_price", ("current price", "today", "price of", "rate of", "what is the price",
                       "how much is", "latest price", "market price", "bhav", "rate")),
    ("market_info", ("which markets", "list markets", "available markets", "which crops",
                     "list crops", "what crops", "available crops")),
    ("help", ("help", "what can you do", "how do you work", "capabilities")),
]

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


class QueryParserAgent:
    """Extracts crop, market, dates, horizon and intent from a user question."""

    def __init__(
        self,
        known_crops: Optional[List[str]] = None,
        known_markets: Optional[List[str]] = None,
        gemini: Optional[GeminiClient] = None,
        use_gemini: bool = True,
    ):
        self.known_crops = list(known_crops or [])
        self.known_markets = list(known_markets or [])
        self.gemini = gemini or get_gemini_client()
        self.use_gemini = use_gemini

    # ---------------------------------------------------------------- public
    def parse(self, query: str) -> Dict[str, Any]:
        """Parse a question. Never raises; returns ok=False for empty input."""
        try:
            if query is None or not str(query).strip():
                raise EmptyQueryError(
                    "Please type a question, for example: 'What is the price of wheat today?'"
                )
            text = str(query).strip()

            rule_based = self._parse_rule_based(text)
            parsed = dict(rule_based)
            source = "rules"

            if self.use_gemini and self.gemini.is_available():
                ai = self._parse_with_gemini(text)
                if ai:
                    source = "gemini"
                    # Gemini wins on intent/entities it is confident about;
                    # rule output fills every gap.
                    for key, value in ai.items():
                        if value not in (None, "", [], {}):
                            parsed[key] = value
                    for key, value in rule_based.items():
                        if parsed.get(key) in (None, "", []):
                            parsed[key] = value

            parsed = self._postprocess(parsed, text)
            return ok(source=source, raw_query=text, **parsed)
        except Exception as exc:
            return err_from_exception(exc, raw_query=query)

    # ------------------------------------------------------------ rule-based
    def _parse_rule_based(self, text: str) -> Dict[str, Any]:
        lowered = f" {text.lower().strip()} "

        crops = self._find_all(lowered, self.known_crops, self._crop_alias_pairs())
        market = self._find_first(lowered, self.known_markets, self._market_alias_pairs())
        days = self._extract_days(lowered)
        intent = self._extract_intent(lowered, crops, days)
        date_from, date_to = self._extract_date_range(lowered)

        return {
            "intent": intent,
            "crop": crops[0] if crops else None,
            "crops": crops,
            "market": market,
            "days": days,
            "date_from": date_from,
            "date_to": date_to,
            "confidence": 0.55 if crops else 0.3,
            "notes": [],
        }

    def _crop_alias_pairs(self) -> Dict[str, str]:
        from tools.market_tools import CROP_ALIASES

        return CROP_ALIASES

    def _market_alias_pairs(self) -> Dict[str, str]:
        from tools.market_tools import MARKET_ALIASES

        return MARKET_ALIASES

    @staticmethod
    def _find_all(lowered: str, options: List[str], aliases: Dict[str, str]) -> List[str]:
        """
        All known names mentioned, in the order they appear in the question.

        Matching is whole-word (plural tolerant): a plain substring search would
        find "Rice" inside "price" and answer the wrong question.
        """
        from tools.market_tools import find_word

        hits: List[tuple] = []
        for option in options:
            position = find_word(lowered, option)
            if position >= 0:
                hits.append((position, option))
        for alias, canonical in aliases.items():
            if canonical not in [h[1] for h in hits]:
                position = find_word(lowered, alias)
                if position >= 0 and (not options or canonical in options):
                    hits.append((position, canonical))
        hits.sort(key=lambda h: h[0])
        seen: List[str] = []
        for _, name in hits:
            if name not in seen:
                seen.append(name)
        return seen

    def _find_first(self, lowered: str, options: List[str], aliases: Dict[str, str]) -> Optional[str]:
        found = self._find_all(lowered, options, aliases)
        return found[0] if found else None

    @staticmethod
    def _extract_days(lowered: str) -> Optional[int]:
        """Pull a prediction horizon out of the sentence."""
        match = re.search(r"(?:next|coming|after|in|for)\s+(\d{1,3})\s*(day|days|din)", lowered)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d{1,3})\s*(?:day|days|din)", lowered)
        if match:
            return int(match.group(1))
        match = re.search(r"(?:next|coming|in)\s+(\d{1,2})\s*(week|weeks|month|months)", lowered)
        if match:
            count = int(match.group(1))
            return count * (7 if "week" in match.group(2) else 30)
        if "next week" in lowered or "coming week" in lowered or "agle hafte" in lowered:
            return 7
        if "next fortnight" in lowered or "two weeks" in lowered or "next 2 weeks" in lowered:
            return 15
        if "next month" in lowered or "agle mahine" in lowered or "coming month" in lowered:
            return 30
        if "next quarter" in lowered or "three months" in lowered or "3 months" in lowered:
            return 90
        if "tomorrow" in lowered or "kal" in lowered:
            return 7  # shortest supported view; day 1 answers "tomorrow"
        return None

    @staticmethod
    def _extract_intent(lowered: str, crops: List[str], days: Optional[int]) -> str:
        for intent, keywords in INTENT_KEYWORDS:
            for keyword in keywords:
                if keyword in lowered:
                    if intent == "compare_crops" and len(crops) < 2:
                        # "compare wheat across markets" is a market comparison
                        continue
                    return intent
        if days:
            return "prediction"
        if len(crops) >= 2:
            return "compare_crops"
        if crops:
            return "current_price"
        return "unknown"

    @staticmethod
    def _extract_date_range(lowered: str) -> tuple:
        """Recognise 'last N days/weeks/months' and explicit ISO dates."""
        today = _dt.date.today()
        iso = re.findall(r"\d{4}-\d{2}-\d{2}", lowered)
        if len(iso) >= 2:
            return iso[0], iso[1]
        if len(iso) == 1:
            return iso[0], None

        match = re.search(r"(?:last|past|previous)\s+(\d{1,3})\s*(day|days|week|weeks|month|months)", lowered)
        if match:
            count = int(match.group(1))
            unit = match.group(2)
            span = count * (1 if "day" in unit else 7 if "week" in unit else 30)
            return (today - _dt.timedelta(days=span)).isoformat(), today.isoformat()
        if "last week" in lowered or "past week" in lowered:
            return (today - _dt.timedelta(days=7)).isoformat(), today.isoformat()
        if "last month" in lowered or "past month" in lowered:
            return (today - _dt.timedelta(days=30)).isoformat(), today.isoformat()
        if "last year" in lowered or "past year" in lowered:
            return (today - _dt.timedelta(days=365)).isoformat(), today.isoformat()

        for name, number in MONTHS.items():
            if re.search(rf"\b{name}\b", lowered):
                year = today.year if number <= today.month else today.year - 1
                start = _dt.date(year, number, 1)
                end_month = start.replace(day=28) + _dt.timedelta(days=4)
                end = end_month - _dt.timedelta(days=end_month.day)
                return start.isoformat(), end.isoformat()
        return None, None

    # ---------------------------------------------------------------- Gemini
    def _parse_with_gemini(self, text: str) -> Optional[Dict[str, Any]]:
        prompt = load_prompt(
            "query_parser_prompt",
            query=text,
            crops=", ".join(self.known_crops) or "unknown",
            markets=", ".join(self.known_markets) or "unknown",
            intents=", ".join(INTENTS),
            today=_dt.date.today().isoformat(),
        )
        if not prompt:  # prompt file missing -> rules only
            return None
        response = self.gemini.generate_json(prompt, temperature=0.0, max_output_tokens=400)
        if not response.ok:
            return None
        try:
            data = json.loads(response.text)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None

        crops = data.get("crops") or ([data.get("crop")] if data.get("crop") else [])
        crops = [str(c).strip().title() for c in crops if c]
        parsed = {
            "intent": str(data.get("intent") or "").strip().lower() or None,
            "crop": crops[0] if crops else None,
            "crops": crops,
            "market": (str(data.get("market")).strip().title() if data.get("market") else None),
            "days": data.get("days") or data.get("prediction_days"),
            "date_from": data.get("date_from"),
            "date_to": data.get("date_to"),
            "confidence": data.get("confidence") or 0.8,
            "notes": [],
        }
        if parsed["intent"] not in INTENTS:
            parsed["intent"] = None
        return parsed

    # --------------------------------------------------------- normalisation
    def _postprocess(self, parsed: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Canonicalise names against the known lists and sanitise the horizon."""
        from tools.market_tools import CROP_ALIASES, MARKET_ALIASES, _resolve

        notes: List[str] = list(parsed.get("notes") or [])

        crops: List[str] = []
        for crop in parsed.get("crops") or []:
            match = _resolve(crop, self.known_crops, CROP_ALIASES) if self.known_crops else crop
            if match and match not in crops:
                crops.append(match)
        parsed["crops"] = crops
        parsed["crop"] = crops[0] if crops else None

        if parsed.get("market"):
            match = (
                _resolve(parsed["market"], self.known_markets, MARKET_ALIASES)
                if self.known_markets
                else parsed["market"]
            )
            if not match:
                notes.append(f"Market '{parsed['market']}' is not in the dataset.")
            parsed["market"] = match

        days = parsed.get("days")
        try:
            days = int(days) if days is not None else None
        except (TypeError, ValueError):
            days = None
        if days is not None:
            if days < 1:
                notes.append("Prediction period must be at least 1 day; using 7 days.")
                days = 7
            elif days > max(VALID_HORIZONS):
                notes.append(
                    f"Predictions go up to {max(VALID_HORIZONS)} days; using "
                    f"{max(VALID_HORIZONS)} days instead of {days}."
                )
                days = max(VALID_HORIZONS)
        parsed["days"] = days

        intent = parsed.get("intent") or "unknown"
        if intent == "prediction" and days is None:
            parsed["days"] = 7
        if intent in ("compare_crops",) and len(crops) < 2:
            intent = "compare_markets" if parsed.get("crop") else "unknown"
        parsed["intent"] = intent if intent in INTENTS else "unknown"
        parsed["notes"] = notes
        return parsed
