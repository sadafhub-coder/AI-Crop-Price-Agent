"""
models/predictor.py
-------------------
Inference side of the ML pipeline.

`PricePredictor` loads the saved bundle (model + encoders + metrics) and walks
the forecast forward one day at a time, feeding each prediction back in as the
next day's lag — the only correct way to forecast multiple steps with a model
trained on lag features.

If the model file is missing or unreadable the predictor degrades to a
transparent statistical fallback (damped linear trend on recent prices) and
says so in `method`, so the UI never shows an empty screen.
"""

from __future__ import annotations

import datetime as _dt
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.helpers import (
    DataUnavailableError,
    ModelUnavailableError,
    coerce_horizon,
)
from utils.preprocessing import (
    CATEGORICAL_COLS,
    CategoryEncoder,
    FEATURE_COLUMNS,
    feature_row_to_array,
    make_feature_row,
    season_from_month,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "saved_models" / "crop_price_model.joblib"

MIN_HISTORY_POINTS = 35  # need enough rows for lag_30 + rolling_30


class PricePredictor:
    """Loads the trained model once and produces future price paths."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self._bundle: Optional[Dict[str, Any]] = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ load
    def load(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """Load (and cache) the model bundle. Returns None on failure."""
        if force:
            self._bundle, self._load_error = None, None
        if self._bundle is not None:
            return self._bundle
        if self._load_error is not None:
            return None
        with self._lock:
            if self._bundle is not None:
                return self._bundle
            if not self.model_path.exists():
                self._load_error = (
                    "No trained model found. Run: python -m models.train_model"
                )
                return None
            try:
                import joblib

                bundle = joblib.load(self.model_path)
                if "model" not in bundle:
                    raise ValueError("model bundle is missing the estimator")
                bundle["encoders"] = {
                    name: CategoryEncoder(mapping)
                    for name, mapping in (bundle.get("encoders") or {}).items()
                }
                self._bundle = bundle
                return bundle
            except Exception as exc:
                self._load_error = f"Could not load the trained model: {exc}"
                return None

    def is_available(self) -> bool:
        return self.load() is not None

    def info(self) -> Dict[str, Any]:
        """Model card for the sidebar: metrics, training window, features."""
        bundle = self.load()
        if bundle is None:
            return {
                "available": False,
                "reason": self._load_error,
                "model_type": None,
                "metrics": {},
            }
        return {
            "available": True,
            "model_type": type(bundle["model"]).__name__,
            "trained_at": bundle.get("trained_at"),
            "train_cutoff": bundle.get("train_cutoff"),
            "metrics": bundle.get("metrics", {}),
            "params": bundle.get("params", {}),
            "n_features": len(bundle.get("feature_columns", FEATURE_COLUMNS)),
            "feature_importance": bundle.get("feature_importance", {}),
            "data": bundle.get("data", {}),
            "path": str(self.model_path),
        }

    @property
    def test_mae(self) -> Optional[float]:
        bundle = self.load()
        if not bundle:
            return None
        return (bundle.get("metrics", {}).get("test") or {}).get("mae")

    # --------------------------------------------------------------- predict
    def predict_future(
        self,
        history: pd.DataFrame,
        days: int = 7,
        crop: Optional[str] = None,
        market: Optional[str] = None,
        state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Forecast `days` calendar days beyond the last date in `history`.

        `history` must be a single crop+market series with columns date and
        modal_price (as produced by tools.market_tools.get_series).
        """
        horizon = coerce_horizon(days)

        if history is None or history.empty:
            raise DataUnavailableError("No historical prices available to forecast from.")

        frame = history.sort_values("date").reset_index(drop=True)
        crop = crop or str(frame.iloc[-1].get("crop", ""))
        market = market or str(frame.iloc[-1].get("market", ""))
        state = state or str(frame.iloc[-1].get("state", ""))

        prices = [float(v) for v in frame["modal_price"].astype(float).tolist()]
        last_date = pd.Timestamp(frame["date"].max()).date()
        last_price = prices[-1]

        bundle = self.load()
        if bundle is None or len(prices) < MIN_HISTORY_POINTS:
            reason = (
                self._load_error
                if bundle is None
                else f"only {len(prices)} historical points (need {MIN_HISTORY_POINTS})"
            )
            return self._fallback_forecast(
                prices, last_date, horizon, crop, market, state, reason
            )

        model = bundle["model"]
        encoders: Dict[str, CategoryEncoder] = bundle["encoders"]
        columns = bundle.get("feature_columns", FEATURE_COLUMNS)

        base_codes = {
            "crop_code": self._code(encoders, "crop", crop),
            "market_code": self._code(encoders, "market", market),
            "state_code": self._code(encoders, "state", state),
        }

        working = list(prices)
        predictions: List[Dict[str, Any]] = []
        mae = self.test_mae or max(1.0, last_price * 0.02)

        try:
            for step in range(1, horizon + 1):
                target_date = last_date + _dt.timedelta(days=step)
                codes = dict(base_codes)
                codes["season_code"] = self._code(
                    encoders, "season", season_from_month(target_date.month)
                )
                row = make_feature_row(working, target_date, codes)
                vector = pd.DataFrame(
                    feature_row_to_array(row), columns=list(FEATURE_COLUMNS)
                )
                if list(columns) != list(FEATURE_COLUMNS):
                    vector = vector.reindex(columns=list(columns), fill_value=0.0)
                predicted = float(model.predict(vector)[0])

                # Guard rail: mandi prices do not move more than ~15% a day.
                predicted = float(
                    np.clip(predicted, working[-1] * 0.85, working[-1] * 1.15)
                )
                working.append(predicted)

                # Uncertainty widens with the horizon (√t), anchored on test MAE.
                band = mae * np.sqrt(step)
                predictions.append(
                    {
                        "day": step,
                        "date": target_date.isoformat(),
                        "predicted_price": round(predicted, 2),
                        "lower_bound": round(max(0.0, predicted - band), 2),
                        "upper_bound": round(predicted + band, 2),
                    }
                )
        except Exception as exc:  # model failure -> never crash the app
            return self._fallback_forecast(
                prices, last_date, horizon, crop, market, state, f"model error: {exc}"
            )

        final_price = predictions[-1]["predicted_price"]
        return {
            "crop": crop,
            "market": market,
            "state": state,
            "method": "random_forest",
            "model_type": type(model).__name__,
            "horizon_days": horizon,
            "last_actual_date": last_date.isoformat(),
            "last_actual_price": round(last_price, 2),
            "predictions": predictions,
            "predicted_final_price": final_price,
            "expected_change": round(final_price - last_price, 2),
            "expected_change_pct": round((final_price - last_price) / last_price * 100, 2),
            "model_mae": round(float(mae), 2),
            "disclaimer": (
                "Predictions are statistical estimates from historical price "
                "patterns, not guaranteed prices."
            ),
        }

    # -------------------------------------------------------------- fallback
    @staticmethod
    def _fallback_forecast(
        prices: List[float],
        last_date: _dt.date,
        horizon: int,
        crop: str,
        market: str,
        state: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Damped linear-trend forecast used when the ML model cannot run.

        Honest and clearly labelled: `method="trend_fallback"`.
        """
        if not prices:
            raise DataUnavailableError("No historical prices available to forecast from.")
        recent = np.asarray(prices[-30:], dtype=float)
        last_price = float(recent[-1])
        if len(recent) >= 3:
            x = np.arange(len(recent), dtype=float)
            slope = float(np.polyfit(x, recent, 1)[0])
            noise = float(np.std(recent))
        else:
            slope, noise = 0.0, max(1.0, last_price * 0.02)

        predictions: List[Dict[str, Any]] = []
        for step in range(1, horizon + 1):
            damping = 0.85 ** step  # trends fade; do not extrapolate forever
            predicted = last_price + slope * step * damping
            predicted = float(np.clip(predicted, last_price * 0.6, last_price * 1.6))
            band = max(noise, last_price * 0.02) * np.sqrt(step)
            predictions.append(
                {
                    "day": step,
                    "date": (last_date + _dt.timedelta(days=step)).isoformat(),
                    "predicted_price": round(predicted, 2),
                    "lower_bound": round(max(0.0, predicted - band), 2),
                    "upper_bound": round(predicted + band, 2),
                }
            )

        final_price = predictions[-1]["predicted_price"]
        return {
            "crop": crop,
            "market": market,
            "state": state,
            "method": "trend_fallback",
            "model_type": "DampedLinearTrend",
            "fallback_reason": reason,
            "horizon_days": horizon,
            "last_actual_date": last_date.isoformat(),
            "last_actual_price": round(last_price, 2),
            "predictions": predictions,
            "predicted_final_price": final_price,
            "expected_change": round(final_price - last_price, 2),
            "expected_change_pct": round((final_price - last_price) / last_price * 100, 2),
            "model_mae": None,
            "disclaimer": (
                "The trained model was unavailable, so this is a simple trend "
                "estimate from recent prices — not a guaranteed price."
            ),
        }

    @staticmethod
    def _code(encoders: Dict[str, CategoryEncoder], column: str, value: Any) -> int:
        encoder = encoders.get(column)
        if encoder is None:
            return CategoryEncoder.UNKNOWN
        return encoder.transform_one(value)

    def require(self) -> Dict[str, Any]:
        """Load or raise — used by tests that must assert on a real model."""
        bundle = self.load()
        if bundle is None:
            raise ModelUnavailableError(self._load_error or "Model unavailable.")
        return bundle


_default_predictor: Optional[PricePredictor] = None


def get_predictor() -> PricePredictor:
    """Process-wide predictor singleton (the model is loaded at most once)."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = PricePredictor()
    return _default_predictor
