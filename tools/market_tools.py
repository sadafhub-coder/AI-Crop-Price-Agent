"""
tools/market_tools.py
---------------------
Shared, side-effect-free helpers that operate on the price table: name
resolution (fuzzy matching so "wheet"/"WHEAT " still work), series slicing and
trend/statistics maths.

Agents call these instead of touching pandas directly, which keeps the agents
readable and the maths tested in one place.
"""

from __future__ import annotations

import datetime as _dt
import re
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.helpers import (
    InvalidCropError,
    InvalidMarketError,
    DataUnavailableError,
    classify_trend,
    pct_change,
    safe_float,
)

#: Common spellings / regional names mapped onto canonical crop names.
CROP_ALIASES: Dict[str, str] = {
    "gehu": "Wheat",
    "gehun": "Wheat",
    "atta": "Wheat",
    "wheet": "Wheat",
    "paddy": "Rice",
    "dhan": "Rice",
    "chawal": "Rice",
    "basmati": "Rice",
    "corn": "Maize",
    "makka": "Maize",
    "bhutta": "Maize",
    "kapas": "Cotton",
    "soya": "Soybean",
    "soyabean": "Soybean",
    "pyaz": "Onion",
    "kanda": "Onion",
    "onions": "Onion",
    "tamatar": "Tomato",
    "tomatoes": "Tomato",
    "tomatos": "Tomato",
    "aloo": "Potato",
    "potatoes": "Potato",
    "ganna": "Sugarcane",
    "dal": "Pulses",
    "daal": "Pulses",
    "tur": "Pulses",
    "arhar": "Pulses",
    "chana": "Pulses",
    "gram": "Pulses",
    "lentil": "Pulses",
    "pulse": "Pulses",
}

MARKET_ALIASES: Dict[str, str] = {
    "bangalore": "Bengaluru",
    "bengaluru city": "Bengaluru",
    "poona": "Pune",
    "aurangabad city": "Aurangabad",
    "chhatrapati sambhajinagar": "Aurangabad",
    "nasik": "Nashik",
    "indore mandi": "Indore",
    "karnal mandi": "Karnal",
}


# --------------------------------------------------------------------------
# Name resolution
# --------------------------------------------------------------------------


def list_crops(df: pd.DataFrame) -> List[str]:
    return sorted(df["crop"].dropna().unique().tolist())


def list_markets(df: pd.DataFrame, crop: Optional[str] = None) -> List[str]:
    frame = df
    if crop:
        frame = df[df["crop"] == crop]
    return sorted(frame["market"].dropna().unique().tolist())


def word_pattern(word: str) -> str:
    """
    Whole-word regex for a crop/market name, tolerating a simple plural.

    Word boundaries matter: a naive substring search finds "Rice" inside the word
    "price", which silently mis-parses "price of wheat" as a rice query.
    """
    return rf"\b{re.escape(word.lower())}(?:e?s)?\b"


def contains_word(haystack: str, word: str) -> bool:
    return re.search(word_pattern(word), haystack.lower()) is not None


def find_word(haystack: str, word: str) -> int:
    """Position of a whole-word match, or -1."""
    match = re.search(word_pattern(word), haystack.lower())
    return match.start() if match else -1


def _resolve(
    value: str, options: Sequence[str], aliases: Dict[str, str]
) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lookup = {opt.lower(): opt for opt in options}
    key = text.lower()
    if key in lookup:
        return lookup[key]
    alias = aliases.get(key)
    if alias and alias.lower() in lookup:
        return lookup[alias.lower()]
    # Whole-word match inside a phrase ("wheat price" -> Wheat)
    for opt_lower, opt in lookup.items():
        if contains_word(key, opt_lower):
            return opt
    for alias_key, canonical in aliases.items():
        if contains_word(key, alias_key) and canonical.lower() in lookup:
            return lookup[canonical.lower()]
    close = get_close_matches(key, list(lookup.keys()), n=1, cutoff=0.75)
    return lookup[close[0]] if close else None


def resolve_crop(df: pd.DataFrame, crop: str) -> str:
    """Canonical crop name or InvalidCropError with the valid options."""
    options = list_crops(df)
    match = _resolve(crop, options, CROP_ALIASES)
    if not match:
        raise InvalidCropError(
            f"I don't have price data for '{crop}'. Available crops: {', '.join(options)}."
        )
    return match


def resolve_market(df: pd.DataFrame, market: str, crop: Optional[str] = None) -> str:
    """Canonical market name or InvalidMarketError with the valid options."""
    options = list_markets(df, crop)
    match = _resolve(market, options, MARKET_ALIASES)
    if not match:
        scope = f" for {crop}" if crop else ""
        raise InvalidMarketError(
            f"I don't have data for market '{market}'{scope}. "
            f"Available markets{scope}: {', '.join(options)}."
        )
    return match


def default_market(df: pd.DataFrame, crop: str) -> str:
    """
    Pick a sensible market when the user names none: the one with the most
    recent records for that crop (i.e. the best-covered mandi).
    """
    frame = df[df["crop"] == crop]
    if frame.empty:
        raise DataUnavailableError(f"No records found for {crop}.")
    counts = frame.groupby("market")["date"].agg(["max", "count"])
    counts = counts.sort_values(["max", "count"], ascending=False)
    return str(counts.index[0])


# --------------------------------------------------------------------------
# Series access
# --------------------------------------------------------------------------


def get_series(
    df: pd.DataFrame,
    crop: str,
    market: Optional[str] = None,
    days: Optional[int] = None,
    date_from: Optional[Any] = None,
    date_to: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Slice one crop (+optional market) into a date-sorted price series.

    Raises DataUnavailableError when the slice is empty — callers turn that into
    a friendly message instead of a stack trace.
    """
    frame = df[df["crop"] == crop]
    if market:
        frame = frame[frame["market"] == market]
    if frame.empty:
        where = f" in {market}" if market else ""
        raise DataUnavailableError(f"No price history available for {crop}{where}.")

    frame = frame.sort_values("date")
    if date_from is not None:
        frame = frame[frame["date"] >= pd.Timestamp(date_from)]
    if date_to is not None:
        frame = frame[frame["date"] <= pd.Timestamp(date_to)]
    if days:
        cutoff = frame["date"].max() - pd.Timedelta(days=int(days))
        frame = frame[frame["date"] >= cutoff]
    if frame.empty:
        raise DataUnavailableError(
            f"No {crop} records in that date range" + (f" for {market}." if market else ".")
        )
    return frame.reset_index(drop=True)


def latest_record(df: pd.DataFrame, crop: str, market: Optional[str] = None) -> Dict[str, Any]:
    """The most recent mandi record as a plain dict."""
    series = get_series(df, crop, market)
    row = series.iloc[-1]
    return {
        "crop": row["crop"],
        "market": row["market"],
        "state": row["state"],
        "date": pd.Timestamp(row["date"]).date().isoformat(),
        "modal_price": safe_float(row["modal_price"]),
        "price_min": safe_float(row["price_min"]),
        "price_max": safe_float(row["price_max"]),
        "arrival_quantity": safe_float(row["arrival_quantity"]),
    }


def price_history_records(series: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame slice -> JSON-friendly list of records."""
    out = []
    for _, row in series.iterrows():
        out.append(
            {
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "modal_price": safe_float(row["modal_price"]),
                "price_min": safe_float(row["price_min"]),
                "price_max": safe_float(row["price_max"]),
                "arrival_quantity": safe_float(row["arrival_quantity"]),
                "market": row["market"],
            }
        )
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def compute_trend_stats(series: pd.DataFrame, window: int = 30) -> Dict[str, Any]:
    """
    Trend maths over the last `window` days of a price series.

    Combines two views so a single noisy day cannot flip the verdict:
      * endpoint change (first vs last price in the window)
      * least-squares slope over the window, expressed as % per week
    """
    if series is None or series.empty:
        raise DataUnavailableError("Not enough history to analyse a trend.")

    frame = series.sort_values("date").copy()
    cutoff = frame["date"].max() - pd.Timedelta(days=int(window))
    recent = frame[frame["date"] >= cutoff]
    if len(recent) < 3:
        recent = frame.tail(max(3, min(len(frame), 7)))

    prices = recent["modal_price"].astype(float).to_numpy()
    dates = pd.to_datetime(recent["date"])
    first_price, last_price = float(prices[0]), float(prices[-1])
    change = pct_change(first_price, last_price)

    # Slope in ₹/day -> % per week, robust to uneven gaps (mandi holidays).
    x = (dates - dates.min()).dt.days.to_numpy(dtype=float)
    slope_per_day = float(np.polyfit(x, prices, 1)[0]) if len(set(x.tolist())) > 1 else 0.0
    mean_price = float(np.mean(prices)) or 1.0
    slope_pct_week = slope_per_day * 7 / mean_price * 100

    blended = np.mean([v for v in (change, slope_pct_week) if v is not None])
    direction = classify_trend(float(blended))

    return {
        "window_days": int(window),
        "points": int(len(recent)),
        "start_date": dates.min().date().isoformat(),
        "end_date": dates.max().date().isoformat(),
        "start_price": round(first_price, 2),
        "end_price": round(last_price, 2),
        "change_pct": round(change, 2) if change is not None else None,
        "slope_per_day": round(slope_per_day, 2),
        "slope_pct_per_week": round(slope_pct_week, 2),
        "direction": direction,
        "average_price": round(mean_price, 2),
        "highest_price": round(float(np.max(prices)), 2),
        "lowest_price": round(float(np.min(prices)), 2),
        "volatility_pct": round(float(np.std(prices) / mean_price * 100), 2),
        "week_over_week_pct": _window_change(frame, 7),
        "month_over_month_pct": _window_change(frame, 30),
    }


def _window_change(frame: pd.DataFrame, days: int) -> Optional[float]:
    """% change between the latest price and the price ~`days` ago."""
    if frame.empty:
        return None
    latest_date = frame["date"].max()
    target = latest_date - pd.Timedelta(days=days)
    past = frame[frame["date"] <= target]
    if past.empty:
        return None
    change = pct_change(
        float(past.iloc[-1]["modal_price"]), float(frame.iloc[-1]["modal_price"])
    )
    return round(change, 2) if change is not None else None


def compare_markets_table(
    df: pd.DataFrame, crop: str, markets: Optional[Sequence[str]] = None
) -> List[Dict[str, Any]]:
    """Latest price per market for one crop, sorted best price first."""
    frame = df[df["crop"] == crop]
    if frame.empty:
        raise DataUnavailableError(f"No price data available for {crop}.")
    if markets:
        frame = frame[frame["market"].isin(list(markets))]
        if frame.empty:
            raise DataUnavailableError(f"No {crop} data for the requested markets.")

    rows: List[Dict[str, Any]] = []
    for market, group in frame.groupby("market"):
        group = group.sort_values("date")
        last = group.iloc[-1]
        stats = compute_trend_stats(group, window=30) if len(group) >= 3 else {}
        rows.append(
            {
                "market": market,
                "state": last["state"],
                "date": pd.Timestamp(last["date"]).date().isoformat(),
                "modal_price": safe_float(last["modal_price"]),
                "price_min": safe_float(last["price_min"]),
                "price_max": safe_float(last["price_max"]),
                "arrival_quantity": safe_float(last["arrival_quantity"]),
                "trend": stats.get("direction"),
                "change_30d_pct": stats.get("change_pct"),
            }
        )
    return sorted(rows, key=lambda r: (r["modal_price"] or 0), reverse=True)


def top_crops_by_price(df: pd.DataFrame, limit: int = 10) -> List[Dict[str, Any]]:
    """Latest average price per crop across markets, highest first."""
    rows: List[Dict[str, Any]] = []
    for crop, group in df.groupby("crop"):
        latest_date = group["date"].max()
        recent = group[group["date"] >= latest_date - pd.Timedelta(days=7)]
        best = recent.sort_values("modal_price", ascending=False).iloc[0]
        rows.append(
            {
                "crop": crop,
                "average_price": round(float(recent["modal_price"].mean()), 2),
                "best_price": safe_float(best["modal_price"]),
                "best_market": best["market"],
                "date": pd.Timestamp(latest_date).date().isoformat(),
            }
        )
    rows.sort(key=lambda r: r["average_price"], reverse=True)
    return rows[:limit]


def next_business_dates(start: _dt.date, days: int) -> List[_dt.date]:
    """Calendar days ahead of `start` (mandis trade most days, Sundays vary)."""
    return [start + _dt.timedelta(days=i) for i in range(1, int(days) + 1)]
