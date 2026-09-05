"""
utils/helpers.py
----------------
Small, dependency-light helpers shared by every layer of the project:
error types, uniform result envelopes, currency/date formatting and
tiny numeric utilities.

Keeping these here means the agents never duplicate formatting logic and
every module speaks the same "result dict" language.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DEFAULT_UNIT = "quintal"
VALID_HORIZONS = (7, 15, 30, 90)

TREND_UP = "increasing"
TREND_DOWN = "decreasing"
TREND_FLAT = "stable"


# --------------------------------------------------------------------------
# Error types
# --------------------------------------------------------------------------


class CropPriceError(Exception):
    """Base class for every expected (user-recoverable) failure."""

    code = "error"


class InvalidCropError(CropPriceError):
    code = "invalid_crop"


class InvalidMarketError(CropPriceError):
    code = "invalid_market"


class DataUnavailableError(CropPriceError):
    code = "no_data"


class InvalidPeriodError(CropPriceError):
    code = "invalid_period"


class ModelUnavailableError(CropPriceError):
    code = "model_unavailable"


class EmptyQueryError(CropPriceError):
    code = "empty_query"


# --------------------------------------------------------------------------
# Uniform result envelopes
# --------------------------------------------------------------------------


def ok(**payload: Any) -> Dict[str, Any]:
    """Successful result envelope."""
    return {"ok": True, **payload}


def err(message: str, code: str = "error", **extra: Any) -> Dict[str, Any]:
    """Failed result envelope. Never raises, never crashes a caller."""
    return {"ok": False, "error": message, "code": code, **extra}


def err_from_exception(exc: Exception, **extra: Any) -> Dict[str, Any]:
    """Convert any exception into a result envelope."""
    code = getattr(exc, "code", "internal_error")
    message = str(exc) or exc.__class__.__name__
    return err(message, code=code, **extra)


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def format_number(value: Any, decimals: int = 0) -> str:
    """Format a number with Indian-style thousand separators (2,450 / 1,20,500)."""
    number = safe_float(value)
    if number is None:
        return "N/A"
    negative = number < 0
    number = abs(number)
    whole = int(number)
    frac = number - whole
    digits = str(whole)
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        digits = ",".join(parts + [tail])
    out = digits
    if decimals > 0:
        out = f"{digits}.{str(round(frac, decimals))[2:][:decimals].ljust(decimals, '0')}"
    return f"-{out}" if negative else out


def format_currency(value: Any, unit: Optional[str] = DEFAULT_UNIT) -> str:
    """₹2,450/quintal — the format farmers see in mandi boards."""
    number = safe_float(value)
    if number is None:
        return "N/A"
    text = f"₹{format_number(round(number))}"
    return f"{text}/{unit}" if unit else text


def format_pct(value: Any, decimals: int = 2) -> str:
    number = safe_float(value)
    if number is None:
        return "N/A"
    sign = "+" if number > 0 else ""
    return f"{sign}{number:.{decimals}f}%"


def format_date(value: Any) -> str:
    """Render any date-ish value as 05 Sep 2026."""
    if value is None:
        return "N/A"
    if isinstance(value, str):
        try:
            value = _dt.datetime.fromisoformat(value)
        except ValueError:
            return value
    if isinstance(value, _dt.datetime):
        value = value.date()
    if isinstance(value, _dt.date):
        return value.strftime("%d %b %Y")
    return str(value)


def to_iso(value: Any) -> str:
    """Normalise any date-ish value to YYYY-MM-DD."""
    if isinstance(value, str):
        return value[:10]
    if isinstance(value, _dt.datetime):
        return value.date().isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    return str(value)


def trend_emoji(direction: Optional[str]) -> str:
    return {
        TREND_UP: "📈",
        TREND_DOWN: "📉",
        TREND_FLAT: "➡️",
    }.get((direction or "").lower(), "➡️")


def title_case(text: Optional[str]) -> str:
    return " ".join(w.capitalize() for w in str(text or "").split())


def humanize_days(days: int) -> str:
    if days % 30 == 0 and days >= 30:
        months = days // 30
        return f"{months} month{'s' if months > 1 else ''}"
    if days % 7 == 0 and days >= 7:
        weeks = days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''}"
    return f"{days} day{'s' if days != 1 else ''}"


# --------------------------------------------------------------------------
# Numeric utilities
# --------------------------------------------------------------------------


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """float() that never raises and rejects NaN/inf."""
    try:
        if value is None:
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def pct_change(old: Any, new: Any) -> Optional[float]:
    """Percentage change from `old` to `new`; None when it is undefined."""
    a = safe_float(old)
    b = safe_float(new)
    if a is None or b is None or a == 0:
        return None
    return (b - a) / abs(a) * 100.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def classify_trend(change_pct: Optional[float], flat_band: float = 1.5) -> str:
    """Map a % change onto increasing / decreasing / stable."""
    if change_pct is None:
        return TREND_FLAT
    if change_pct > flat_band:
        return TREND_UP
    if change_pct < -flat_band:
        return TREND_DOWN
    return TREND_FLAT


def coerce_horizon(days: Any, allowed: tuple = VALID_HORIZONS) -> int:
    """
    Validate a prediction horizon.

    Accepts anything int-like between 1 and max(allowed). Values that are not
    one of the canonical horizons are snapped up to the nearest allowed one so
    "predict 10 days" still works instead of failing.
    """
    number = safe_float(days)
    if number is None:
        raise InvalidPeriodError(
            f"Prediction period must be a number of days (allowed: {', '.join(map(str, allowed))})."
        )
    number = int(round(number))
    if number < 1:
        raise InvalidPeriodError("Prediction period must be at least 1 day.")
    if number > max(allowed):
        raise InvalidPeriodError(
            f"Predictions are supported up to {max(allowed)} days ahead, not {number}."
        )
    return number
