"""
tests/test_trend.py
-------------------
Trend Agent + trend maths: direction detection on synthetic rising/falling/flat
series, percentage change, volatility, insights and error handling.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
import pytest

from agents.trend_agent import TrendAgent
from tools.market_tools import compute_trend_stats
from utils.helpers import (
    TREND_DOWN,
    TREND_FLAT,
    TREND_UP,
    DataUnavailableError,
    classify_trend,
    pct_change,
)


def _series(prices, start="2026-01-01"):
    """Build a minimal crop+market price frame from a list of prices."""
    dates = pd.date_range(start=start, periods=len(prices), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "crop": "Wheat",
            "market": "Pune",
            "state": "Maharashtra",
            "modal_price": prices,
            "price_min": [p * 0.97 for p in prices],
            "price_max": [p * 1.03 for p in prices],
            "arrival_quantity": [100] * len(prices),
        }
    )


# ------------------------------------------------------------------ pure maths
def test_pct_change_basics():
    assert pct_change(100, 110) == pytest.approx(10.0)
    assert pct_change(100, 90) == pytest.approx(-10.0)
    assert pct_change(0, 90) is None
    assert pct_change(None, 90) is None


@pytest.mark.parametrize(
    "change,expected",
    [(5.0, TREND_UP), (-5.0, TREND_DOWN), (0.4, TREND_FLAT), (None, TREND_FLAT)],
)
def test_classify_trend(change, expected):
    assert classify_trend(change) == expected


def test_rising_series_is_detected():
    stats = compute_trend_stats(_series(list(np.linspace(2000, 2400, 40))), window=30)
    assert stats["direction"] == TREND_UP
    assert stats["change_pct"] > 0
    assert stats["slope_per_day"] > 0


def test_falling_series_is_detected():
    stats = compute_trend_stats(_series(list(np.linspace(2400, 2000, 40))), window=30)
    assert stats["direction"] == TREND_DOWN
    assert stats["change_pct"] < 0
    assert stats["slope_per_day"] < 0


def test_flat_series_is_detected():
    prices = [2200 + (1 if i % 2 else -1) for i in range(40)]
    stats = compute_trend_stats(_series(prices), window=30)
    assert stats["direction"] == TREND_FLAT
    assert abs(stats["change_pct"]) < 1.5


def test_stats_report_range_and_volatility():
    stats = compute_trend_stats(_series([2000, 2100, 2500, 2300, 2200] * 8), window=30)
    assert stats["highest_price"] >= stats["average_price"] >= stats["lowest_price"]
    assert stats["volatility_pct"] > 0
    assert stats["points"] > 0


def test_short_series_still_works():
    """Only 4 points: must not crash, just analyse what exists."""
    stats = compute_trend_stats(_series([2000, 2050, 2100, 2150]), window=30)
    assert stats["direction"] in (TREND_UP, TREND_FLAT)


def test_empty_series_raises():
    with pytest.raises(DataUnavailableError):
        compute_trend_stats(_series([]).iloc[0:0], window=30)


# ----------------------------------------------------------------- TrendAgent
def test_agent_analyze_real_data(api, sample_crop, sample_market):
    result = TrendAgent(api).analyze(sample_crop, sample_market)
    assert result["ok"] is True
    assert result["direction"] in (TREND_UP, TREND_DOWN, TREND_FLAT)
    assert isinstance(result["insights"], list) and result["insights"]
    assert "%" in result["summary"]


def test_agent_analyze_window_is_respected(api, sample_crop, sample_market):
    short = TrendAgent(api).analyze(sample_crop, sample_market, window=7)
    long = TrendAgent(api).analyze(sample_crop, sample_market, window=90)
    assert short["window_days"] == 7 and long["window_days"] == 90
    assert long["points"] >= short["points"]


def test_agent_requires_crop(api):
    result = TrendAgent(api).analyze("")
    assert result["ok"] is False and result["code"] == "missing_crop"


def test_agent_invalid_crop_is_graceful(api):
    result = TrendAgent(api).analyze("moonrock")
    assert result["ok"] is False and result["code"] == "invalid_crop"


def test_agent_compare_windows(api, sample_crop, sample_market):
    result = TrendAgent(api).compare_windows(sample_crop, sample_market)
    assert result["ok"] is True and len(result["windows"]) == 3


def test_direction_label_has_emoji():
    assert "📈" in TrendAgent.direction_label(TREND_UP)
    assert "📉" in TrendAgent.direction_label(TREND_DOWN)
    assert "➡️" in TrendAgent.direction_label(TREND_FLAT)


def test_explain_falls_back_without_gemini(api, sample_crop, sample_market):
    agent = TrendAgent(api)
    trend = agent.analyze(sample_crop, sample_market)
    text = agent.explain(trend)
    assert trend["summary"] in text  # deterministic fallback path
