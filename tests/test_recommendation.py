"""
tests/test_recommendation.py
----------------------------
Recommendation Agent: scoring logic (rising forecast -> wait, falling -> sell),
the "never guarantee" contract, best-market hints and error handling.
"""

from __future__ import annotations

import pytest

from agents.recommendation_agent import ACTIONS, RecommendationAgent

FORBIDDEN_PHRASES = [
    "guaranteed",
    "will definitely",
    "you will get",
    "assured price",
    "certain profit",
    "risk-free",
]


def _price(value=2450.0, crop="Wheat", market="Pune"):
    return {
        "ok": True,
        "crop": crop,
        "market": market,
        "state": "Maharashtra",
        "date": "2026-09-01",
        "modal_price": value,
        "price_min": value * 0.97,
        "price_max": value * 1.03,
        "arrival_quantity": 120.0,
    }


def _trend(direction="increasing", change=6.0, volatility=4.0, low=2300, high=2500):
    return {
        "ok": True,
        "crop": "Wheat",
        "market": "Pune",
        "direction": direction,
        "change_pct": change,
        "volatility_pct": volatility,
        "lowest_price": low,
        "highest_price": high,
        "average_price": (low + high) / 2,
        "window_days": 30,
        "insights": [],
    }


def _prediction(change=5.0, horizon=7, method="random_forest", price=2570.0):
    return {
        "ok": True,
        "crop": "Wheat",
        "market": "Pune",
        "horizon_days": horizon,
        "predicted_final_price": price,
        "expected_change_pct": change,
        "expected_change": price - 2450,
        "method": method,
        "predictions": [],
        "checkpoints": [],
    }


# ------------------------------------------------------------------- scoring
def test_rising_forecast_suggests_waiting(api):
    result = RecommendationAgent(api).recommend(
        "Wheat", "Pune",
        price=_price(2350),  # near the bottom of the range
        trend=_trend("increasing", 6.0),
        prediction=_prediction(6.0),
    )
    assert result["ok"] is True
    assert result["action"] == "wait"
    assert result["score"] > 0


def test_falling_forecast_suggests_selling(api):
    result = RecommendationAgent(api).recommend(
        "Wheat", "Pune",
        price=_price(2490),
        trend=_trend("decreasing", -6.0),
        prediction=_prediction(-6.0, price=2300),
    )
    assert result["action"] == "sell"
    assert result["score"] < 0


def test_flat_market_suggests_watching(api):
    result = RecommendationAgent(api).recommend(
        "Wheat", "Pune",
        price=_price(2400),
        trend=_trend("stable", 0.3, volatility=2.0),
        prediction=_prediction(0.2, price=2455),
    )
    assert result["action"] == "watch"


def test_fallback_forecast_is_weighted_down(api):
    agent = RecommendationAgent(api)
    strong = agent.recommend(
        "Wheat", "Pune", price=_price(), trend=_trend(), prediction=_prediction(6.0)
    )
    weak = agent.recommend(
        "Wheat", "Pune", price=_price(), trend=_trend(),
        prediction=_prediction(6.0, method="trend_fallback"),
    )
    assert weak["score"] < strong["score"]


def test_volatility_penalises_waiting(api):
    agent = RecommendationAgent(api)
    calm = agent.recommend(
        "Wheat", "Pune", price=_price(), trend=_trend(volatility=2.0), prediction=_prediction(4.0)
    )
    wild = agent.recommend(
        "Wheat", "Pune", price=_price(), trend=_trend(volatility=15.0), prediction=_prediction(4.0)
    )
    assert wild["score"] < calm["score"]


# --------------------------------------------------------------- safety rules
def test_recommendation_never_guarantees(api, sample_crop, sample_market):
    result = RecommendationAgent(api).recommend(sample_crop, sample_market, days=7)
    text = (result["recommendation"] + " " + " ".join(result["reasons"])).lower()
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in text
    assert "not a guaranteed outcome" in result["disclaimer"].lower()


def test_recommendation_uses_considered_language(api, sample_crop, sample_market):
    result = RecommendationAgent(api).recommend(sample_crop, sample_market, days=7)
    text = result["recommendation"].lower()
    assert any(word in text for word in ("consider", "may", "could", "watching"))


def test_action_labels_are_known(api, sample_crop, sample_market):
    result = RecommendationAgent(api).recommend(sample_crop, sample_market, days=7)
    assert result["action"] in ACTIONS
    assert result["action_label"] == ACTIONS[result["action"]]


# ------------------------------------------------------------- real data path
def test_recommend_on_real_data_has_reasons(api, sample_crop, sample_market):
    result = RecommendationAgent(api).recommend(sample_crop, sample_market, days=15)
    assert result["ok"] is True
    assert len(result["reasons"]) >= 2
    assert result["price"]["ok"] is True
    assert result["prediction"] is not None


def test_best_market_hint_when_another_mandi_pays_more(api, sample_crop):
    """Pick the cheapest mandi: the agent should flag a better-paying one."""
    comparison = api.compare_markets(sample_crop)
    cheapest = comparison["markets"][-1]["market"]
    result = RecommendationAgent(api).recommend(sample_crop, cheapest, days=7)
    assert result["ok"] is True
    if result["best_market"]:
        assert result["best_market"] != cheapest
        assert result["best_market_price"] >= result["price"]["modal_price"]


def test_missing_crop_is_graceful(api):
    result = RecommendationAgent(api).recommend("")
    assert result["ok"] is False and result["code"] == "missing_crop"


def test_invalid_crop_is_graceful(api):
    result = RecommendationAgent(api).recommend("kryptonite")
    assert result["ok"] is False and result["code"] == "invalid_crop"


def test_explain_falls_back_without_gemini(api, sample_crop, sample_market):
    agent = RecommendationAgent(api)
    result = agent.recommend(sample_crop, sample_market, days=7)
    assert agent.explain(result) == result["recommendation"]
