"""
tests/test_prediction.py
------------------------
Prediction Agent + ML pipeline: feature engineering with no leakage,
chronological splitting, horizon validation, recursive forecasting and the
statistical fallback when no model is available.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
import pytest

from agents.prediction_agent import PredictionAgent
from models.predictor import PricePredictor
from utils.helpers import InvalidPeriodError, VALID_HORIZONS, coerce_horizon
from utils.preprocessing import (
    FEATURE_COLUMNS,
    LAG_FEATURES,
    TARGET,
    build_feature_frame,
    make_feature_row,
    season_from_month,
    time_series_split,
)


# ------------------------------------------------------------ horizon validation
@pytest.mark.parametrize("days", list(VALID_HORIZONS))
def test_canonical_horizons_accepted(days):
    assert coerce_horizon(days) == days


def test_non_canonical_horizon_accepted():
    assert coerce_horizon(10) == 10


@pytest.mark.parametrize("bad", [0, -5, 400, "soon", None])
def test_invalid_horizons_rejected(bad):
    with pytest.raises(InvalidPeriodError):
        coerce_horizon(bad)


# ------------------------------------------------------------ feature pipeline
def test_feature_frame_builds_expected_columns(data_frame):
    features, encoders = build_feature_frame(data_frame.head(4000), fit=True)
    assert not features.empty
    for column in FEATURE_COLUMNS:
        assert column in features.columns
    assert set(encoders) == {"crop", "market", "state", "season"}
    assert features[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_lag_features_contain_no_leakage():
    """lag_1 for day D must equal the price of day D-1, never day D."""
    prices = list(np.linspace(2000, 2200, 120))
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=120, freq="D"),
            "crop": "Wheat",
            "market": "Pune",
            "state": "Maharashtra",
            "modal_price": prices,
            "price_min": [p * 0.97 for p in prices],
            "price_max": [p * 1.03 for p in prices],
            "arrival_quantity": 100,
        }
    )
    features, _ = build_feature_frame(frame, fit=True)
    row = features.iloc[10]
    original = frame.set_index("date")["modal_price"]
    previous_day = pd.Timestamp(row["date"]) - pd.Timedelta(days=1)
    assert row["lag_1"] == pytest.approx(original.loc[previous_day])
    assert row["lag_1"] != pytest.approx(row[TARGET])


def test_time_series_split_is_chronological(data_frame):
    features, _ = build_feature_frame(data_frame, fit=True)
    train, test, cutoff = time_series_split(features, test_size=0.2)
    assert len(train) > 0 and len(test) > 0
    assert pd.to_datetime(train["date"]).max() <= cutoff
    assert pd.to_datetime(test["date"]).min() > cutoff


def test_season_mapping():
    assert season_from_month(7) == "Kharif"
    assert season_from_month(12) == "Rabi"
    assert season_from_month(4) == "Zaid"


def test_make_feature_row_uses_history_tail():
    history = list(np.linspace(2000, 2100, 40))
    row = make_feature_row(history, _dt.date(2026, 6, 15), {"crop_code": 1, "market_code": 2})
    assert row["lag_1"] == pytest.approx(history[-1])
    assert row["lag_7"] == pytest.approx(history[-7])
    assert row["roll_mean_7"] == pytest.approx(float(np.mean(history[-7:])))
    assert row["month"] == 6 and row["crop_code"] == 1
    for column in LAG_FEATURES:
        assert column in row


# ---------------------------------------------------------------- predictions
def test_prediction_returns_requested_number_of_days(api, sample_crop, sample_market):
    result = PredictionAgent(api).predict(sample_crop, sample_market, days=7)
    assert result["ok"] is True
    assert len(result["predictions"]) == 7
    assert result["horizon_days"] == 7


@pytest.mark.parametrize("days", [7, 15, 30, 90])
def test_all_supported_horizons_work(api, sample_crop, sample_market, days):
    result = PredictionAgent(api).predict(sample_crop, sample_market, days=days)
    assert result["ok"] is True and len(result["predictions"]) == days


def test_prediction_dates_are_future_and_sequential(api, sample_crop, sample_market):
    result = PredictionAgent(api).predict(sample_crop, sample_market, days=15)
    last_actual = _dt.date.fromisoformat(result["last_actual_date"])
    dates = [_dt.date.fromisoformat(p["date"]) for p in result["predictions"]]
    assert dates[0] == last_actual + _dt.timedelta(days=1)
    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates)


def test_predicted_prices_are_plausible(api, sample_crop, sample_market):
    result = PredictionAgent(api).predict(sample_crop, sample_market, days=30)
    current = result["last_actual_price"]
    for point in result["predictions"]:
        assert 0 < point["predicted_price"] < current * 3
        assert point["lower_bound"] <= point["predicted_price"] <= point["upper_bound"]


def test_prediction_carries_disclaimer(api, sample_crop, sample_market):
    result = PredictionAgent(api).predict(sample_crop, sample_market, days=7)
    assert "not guaranteed" in result["disclaimer"].lower()
    assert "⚠️" in PredictionAgent(api).format_reply(result)


def test_checkpoints_are_readable_for_long_horizons(api, sample_crop, sample_market):
    result = PredictionAgent(api).predict(sample_crop, sample_market, days=90)
    days = [point["day"] for point in result["checkpoints"]]
    assert len(days) <= 8 and days[0] == 1 and days[-1] == 90


def test_invalid_period_is_graceful(api, sample_crop, sample_market):
    result = PredictionAgent(api).predict(sample_crop, sample_market, days=500)
    assert result["ok"] is False and result["code"] == "invalid_period"


def test_invalid_crop_is_graceful(api):
    result = PredictionAgent(api).predict("stardust", days=7)
    assert result["ok"] is False and result["code"] == "invalid_crop"


def test_missing_crop_is_graceful(api):
    result = PredictionAgent(api).predict("", days=7)
    assert result["ok"] is False and result["code"] == "missing_crop"


def test_multi_horizon_covers_every_horizon(api, sample_crop, sample_market):
    result = PredictionAgent(api).predict_multi_horizon(sample_crop, sample_market)
    assert result["ok"] is True
    assert [row["horizon_days"] for row in result["horizons"]] == list(VALID_HORIZONS)


# ------------------------------------------------------------------- fallback
def test_missing_model_falls_back_not_crashes(api, sample_crop, sample_market, tmp_path):
    """Point the predictor at a non-existent model file: must still forecast."""
    from tools.market_api import MarketAPI

    broken = MarketAPI(source=api.source, predictor=PricePredictor(tmp_path / "nope.joblib"))
    result = broken.get_prediction(sample_crop, sample_market, days=7)
    assert result["ok"] is True
    assert result["method"] == "trend_fallback"
    assert len(result["predictions"]) == 7
    assert "not a guaranteed price" in result["disclaimer"]


def test_predictor_info_reports_availability(api):
    info = api.predictor.info()
    assert "available" in info
    if info["available"]:
        assert info["metrics"]["test"]["r2"] is not None
        assert info["model_type"] == "RandomForestRegressor"


def test_trained_model_quality_if_present(api):
    """When a model exists, it must beat a naive constant predictor (R² > 0.5)."""
    info = api.predictor.info()
    if not info.get("available"):
        pytest.skip("model not trained in this environment")
    assert info["metrics"]["test"]["r2"] > 0.5
