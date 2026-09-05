"""
models/train_model.py
---------------------
The complete training pipeline for the crop-price model.

Steps (all real, all runnable):
  1. load the dataset (CSV or SQLite, generated if missing)
  2. clean it and handle missing values
  3. build time, lag and rolling-average features
  4. encode categoricals with an unknown-tolerant encoder
  5. split chronologically (never randomly — that would leak the future)
  6. train a RandomForestRegressor
  7. evaluate with MAE / RMSE / R² on the held-out future window
  8. save model + encoders + metrics to models/saved_models/

Run it with:  python -m models.train_model
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allows `python models/train_model.py` too
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402

from tools.data_loader import load_data  # noqa: E402
from utils.preprocessing import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET,
    build_feature_frame,
    time_series_split,
)

SAVED_MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
MODEL_PATH = SAVED_MODELS_DIR / "crop_price_model.joblib"
METRICS_PATH = SAVED_MODELS_DIR / "metrics.json"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE without relying on sklearn's `squared=` kwarg (removed in 1.6)."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def evaluate(model: RandomForestRegressor, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    if len(X) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}
    pred = model.predict(X)
    mape = float(np.mean(np.abs((y.to_numpy() - pred) / np.maximum(y.to_numpy(), 1))) * 100)
    return {
        "mae": round(float(mean_absolute_error(y, pred)), 2),
        "rmse": round(rmse(y, pred), 2),
        "r2": round(float(r2_score(y, pred)), 4),
        "mape": round(mape, 2),
        "n": int(len(X)),
    }


def train(
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 2,
    test_size: float = 0.2,
    source: str = "auto",
    model_path: Path = MODEL_PATH,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the whole pipeline and persist the model bundle. Returns a report."""

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log("1/8  Loading dataset ...")
    raw = load_data(source=source, use_cache=False)
    log(f"     {len(raw):,} cleaned rows | {raw.groupby(['crop', 'market']).ngroups} series")

    log("2/8  Building features (time + lag + rolling) ...")
    features, encoders = build_feature_frame(raw, fit=True)
    if features.empty:
        raise SystemExit("No usable feature rows — dataset too short to train on.")
    log(f"     {len(features):,} training rows x {len(FEATURE_COLUMNS)} features")

    log("3/8  Chronological train/test split (no random shuffling) ...")
    train_df, test_df, cutoff = time_series_split(features, test_size=test_size)
    log(
        f"     train {len(train_df):,} rows (to {cutoff.date() if cutoff is not None else 'n/a'}) | "
        f"test {len(test_df):,} rows (future window)"
    )

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET]

    log(f"4/8  Training RandomForestRegressor(n_estimators={n_estimators}) ...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    log("5/8  Evaluating ...")
    train_metrics = evaluate(model, X_train, y_train)
    test_metrics = evaluate(model, X_test, y_test)
    log(
        f"     TEST  MAE {test_metrics['mae']}  RMSE {test_metrics['rmse']}  "
        f"R2 {test_metrics['r2']}  MAPE {test_metrics['mape']}%"
    )

    log("6/8  Ranking feature importance ...")
    importance = (
        pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
        .sort_values(ascending=False)
        .round(4)
    )
    log("     top: " + ", ".join(f"{k}={v}" for k, v in importance.head(6).items()))

    log("7/8  Saving model bundle ...")
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "encoders": {name: enc.to_dict() for name, enc in encoders.items()},
        "feature_columns": list(FEATURE_COLUMNS),
        "target": TARGET,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_cutoff": str(cutoff.date()) if cutoff is not None else None,
        "metrics": {"train": train_metrics, "test": test_metrics},
        "feature_importance": importance.to_dict(),
        "data": {
            "rows": int(len(raw)),
            "feature_rows": int(len(features)),
            "crops": sorted(raw["crop"].unique().tolist()),
            "markets": sorted(raw["market"].unique().tolist()),
            "date_from": str(pd.to_datetime(raw["date"]).min().date()),
            "date_to": str(pd.to_datetime(raw["date"]).max().date()),
        },
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "test_size": test_size,
        },
        "environment": {"python": platform.python_version()},
    }
    joblib.dump(bundle, model_path, compress=3)

    report = {
        "model_path": str(model_path),
        "metrics": bundle["metrics"],
        "trained_at": bundle["trained_at"],
        "train_cutoff": bundle["train_cutoff"],
        "rows": bundle["data"]["feature_rows"],
        "top_features": importance.head(10).to_dict(),
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log(f"8/8  Done. Model -> {model_path.name}, metrics -> {METRICS_PATH.name}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the crop price prediction model.")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--source", choices=["auto", "csv", "sqlite"], default="auto")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report = train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        test_size=args.test_size,
        source=args.source,
        verbose=not args.quiet,
    )
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
