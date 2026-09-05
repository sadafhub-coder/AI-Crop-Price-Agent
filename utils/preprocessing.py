"""
utils/preprocessing.py
----------------------
Data cleaning and feature engineering shared by training and inference.

Both paths MUST build features the same way, otherwise the model sees a
different world at predict time. That is why the feature order lives here in a
single constant (`FEATURE_COLUMNS`) and both the training frame builder and the
single-row builder derive from it.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TARGET = "modal_price"
GROUP_COLS = ["crop", "market"]
CATEGORICAL_COLS = ["crop", "market", "state", "season"]

LAGS: Tuple[int, ...] = (1, 3, 7, 14, 30)
ROLL_WINDOWS: Tuple[int, ...] = (7, 14, 30)

TIME_FEATURES = ["month", "day", "dayofweek", "dayofyear", "weekofyear", "quarter"]
LAG_FEATURES = [f"lag_{lag}" for lag in LAGS]
ROLL_FEATURES = [f"roll_mean_{w}" for w in ROLL_WINDOWS] + ["roll_std_7"]
CAT_FEATURES = [f"{col}_code" for col in CATEGORICAL_COLS]

#: Canonical model input order — never reorder without retraining.
FEATURE_COLUMNS: List[str] = TIME_FEATURES + CAT_FEATURES + LAG_FEATURES + ROLL_FEATURES

REQUIRED_COLUMNS = [
    "date",
    "crop",
    "market",
    "state",
    "arrival_quantity",
    "price_min",
    "price_max",
    "modal_price",
]

# Indian cropping seasons — a genuinely predictive feature for mandi prices.
SEASONS = {
    "Kharif": (6, 7, 8, 9, 10),
    "Rabi": (11, 12, 1, 2, 3),
    "Zaid": (4, 5),
}


def season_from_month(month: int) -> str:
    for name, months in SEASONS.items():
        if month in months:
            return name
    return "Unknown"


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a raw price table trustworthy:
      * parse dates, drop unparseable rows
      * normalise text columns (strip + title case)
      * coerce numerics, repair impossible min/max ordering
      * interpolate short gaps in modal_price inside each crop+market series
      * drop exact duplicates, keeping the latest record per key
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    for col in missing:
        out[col] = np.nan

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    # Blank / literal-NaN text becomes real missing data, so the dropna below
    # actually removes unusable rows instead of keeping empty crop names.
    for col in ("crop", "market", "state"):
        out[col] = (
            out[col]
            .astype(str)
            .str.strip()
            .str.title()
            .replace({"Nan": np.nan, "None": np.nan, "": np.nan})
        )
    out = out.dropna(subset=["crop", "market"])

    for col in ("arrival_quantity", "price_min", "price_max", "modal_price"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Impossible values -> missing
    for col in ("price_min", "price_max", "modal_price"):
        out.loc[out[col] <= 0, col] = np.nan

    # Fill modal price from min/max when only those exist, and vice versa
    mid = out[["price_min", "price_max"]].mean(axis=1)
    out["modal_price"] = out["modal_price"].fillna(mid)
    out["price_min"] = out["price_min"].fillna(out["modal_price"] * 0.95)
    out["price_max"] = out["price_max"].fillna(out["modal_price"] * 1.05)

    # Swap inverted min/max
    inverted = out["price_min"] > out["price_max"]
    out.loc[inverted, ["price_min", "price_max"]] = out.loc[
        inverted, ["price_max", "price_min"]
    ].values

    out["arrival_quantity"] = out["arrival_quantity"].fillna(0).clip(lower=0)

    out = out.sort_values(["crop", "market", "date"])
    out = out.drop_duplicates(subset=["crop", "market", "date"], keep="last")

    # Interpolate short holes per series, then drop what is still unknown
    out["modal_price"] = out.groupby(GROUP_COLS, group_keys=False)["modal_price"].apply(
        lambda s: s.interpolate(limit=5, limit_direction="both")
    )
    out = out.dropna(subset=["modal_price"])

    out["season"] = out["date"].dt.month.map(season_from_month)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Categorical encoding
# --------------------------------------------------------------------------


class CategoryEncoder:
    """
    Dictionary-based label encoder that tolerates unseen categories.

    sklearn's LabelEncoder raises on unknown values, which would crash the app
    the first time a user picks a market added after training. This returns -1
    instead, which tree models handle as "its own branch".
    """

    UNKNOWN = -1

    def __init__(self, mapping: Optional[Dict[str, int]] = None):
        self.mapping: Dict[str, int] = dict(mapping or {})

    def fit(self, values: Iterable[Any]) -> "CategoryEncoder":
        uniques = sorted({self._key(v) for v in values if self._key(v)})
        self.mapping = {value: index for index, value in enumerate(uniques)}
        return self

    def transform_one(self, value: Any) -> int:
        return self.mapping.get(self._key(value), self.UNKNOWN)

    def transform(self, values: Iterable[Any]) -> np.ndarray:
        return np.array([self.transform_one(v) for v in values], dtype=int)

    @property
    def classes_(self) -> List[str]:
        return list(self.mapping.keys())

    @staticmethod
    def _key(value: Any) -> str:
        return str(value).strip().title() if value is not None else ""

    def to_dict(self) -> Dict[str, int]:
        return dict(self.mapping)


def fit_encoders(df: pd.DataFrame) -> Dict[str, CategoryEncoder]:
    return {col: CategoryEncoder().fit(df[col]) for col in CATEGORICAL_COLS}


def apply_encoders(
    df: pd.DataFrame, encoders: Dict[str, CategoryEncoder]
) -> pd.DataFrame:
    out = df.copy()
    for col in CATEGORICAL_COLS:
        encoder = encoders.get(col) or CategoryEncoder()
        out[f"{col}_code"] = encoder.transform(out[col])
    return out


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    out["year"] = dates.dt.year
    out["month"] = dates.dt.month
    out["day"] = dates.dt.day
    out["dayofweek"] = dates.dt.dayofweek
    out["dayofyear"] = dates.dt.dayofyear
    out["weekofyear"] = dates.dt.isocalendar().week.astype(int)
    out["quarter"] = dates.dt.quarter
    out["season"] = out["month"].map(season_from_month)
    return out


def add_lag_features(df: pd.DataFrame, lags: Sequence[int] = LAGS) -> pd.DataFrame:
    out = df.sort_values(GROUP_COLS + ["date"]).copy()
    grouped = out.groupby(GROUP_COLS)[TARGET]
    for lag in lags:
        out[f"lag_{lag}"] = grouped.shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame, windows: Sequence[int] = ROLL_WINDOWS
) -> pd.DataFrame:
    """
    Rolling statistics computed on *shifted* prices only.

    The shift(1) is what prevents target leakage: the value for day D never
    contains day D's own price.
    """
    out = df.sort_values(GROUP_COLS + ["date"]).copy()
    shifted = out.groupby(GROUP_COLS)[TARGET].shift(1)
    out["_shifted"] = shifted
    for window in windows:
        out[f"roll_mean_{window}"] = (
            out.groupby(GROUP_COLS)["_shifted"]
            .rolling(window, min_periods=max(2, window // 3))
            .mean()
            .reset_index(level=list(range(len(GROUP_COLS))), drop=True)
        )
    out["roll_std_7"] = (
        out.groupby(GROUP_COLS)["_shifted"]
        .rolling(7, min_periods=3)
        .std()
        .reset_index(level=list(range(len(GROUP_COLS))), drop=True)
    )
    return out.drop(columns=["_shifted"])


def build_feature_frame(
    df: pd.DataFrame,
    encoders: Optional[Dict[str, CategoryEncoder]] = None,
    fit: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, CategoryEncoder]]:
    """
    Full feature pipeline: clean -> time features -> lags -> rollings -> encode.

    Rows whose lag/rolling history is incomplete (the warm-up of every series)
    are dropped, because imputing them would teach the model fake history.
    """
    clean = clean_dataframe(df)
    if clean.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + [TARGET, "date"]), (encoders or {})

    featured = add_rolling_features(add_lag_features(add_time_features(clean)))

    if fit or not encoders:
        encoders = fit_encoders(featured)
    featured = apply_encoders(featured, encoders)

    featured = featured.dropna(subset=LAG_FEATURES + ROLL_FEATURES)
    keep = ["date"] + GROUP_COLS + ["state", "season", TARGET] + FEATURE_COLUMNS
    keep = [c for c in dict.fromkeys(keep) if c in featured.columns]
    return featured[keep].reset_index(drop=True), encoders


def time_series_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Timestamp]]:
    """
    Chronological split — NEVER random for forecasting.

    The cut is made on the date axis (not on row count) so no future date leaks
    into training for any crop/market series.
    """
    if df.empty:
        return df, df, None
    dates = np.sort(pd.to_datetime(df["date"]).unique())
    if len(dates) < 5:
        return df, df.iloc[0:0], None
    cutoff_index = max(1, int(len(dates) * (1 - test_size)) - 1)
    cutoff = pd.Timestamp(dates[cutoff_index])
    train = df[pd.to_datetime(df["date"]) <= cutoff]
    test = df[pd.to_datetime(df["date"]) > cutoff]
    return train.reset_index(drop=True), test.reset_index(drop=True), cutoff


# --------------------------------------------------------------------------
# Single-row features (recursive forecasting)
# --------------------------------------------------------------------------


def make_feature_row(
    history: Sequence[float],
    target_date: _dt.date,
    codes: Dict[str, int],
) -> Dict[str, float]:
    """
    Build ONE model input row from a price history ending the day before
    `target_date`.

    `history` is ordered oldest -> newest and must already include any prices
    predicted earlier in the same recursive walk. `codes` holds the encoded
    crop/market/state/season values.
    """
    series = [float(v) for v in history if v is not None and not np.isnan(float(v))]
    if not series:
        raise ValueError("Cannot build features from an empty price history.")

    stamp = pd.Timestamp(target_date)
    row: Dict[str, float] = {
        "month": stamp.month,
        "day": stamp.day,
        "dayofweek": stamp.dayofweek,
        "dayofyear": stamp.dayofyear,
        "weekofyear": int(stamp.isocalendar().week),
        "quarter": stamp.quarter,
    }
    for col in CATEGORICAL_COLS:
        row[f"{col}_code"] = codes.get(f"{col}_code", codes.get(col, CategoryEncoder.UNKNOWN))

    for lag in LAGS:
        row[f"lag_{lag}"] = series[-lag] if len(series) >= lag else series[0]

    arr = np.asarray(series, dtype=float)
    for window in ROLL_WINDOWS:
        window_slice = arr[-window:] if len(arr) >= 1 else arr
        row[f"roll_mean_{window}"] = float(np.mean(window_slice))
    tail7 = arr[-7:]
    row["roll_std_7"] = float(np.std(tail7, ddof=1)) if len(tail7) > 1 else 0.0
    return row


def feature_row_to_array(row: Dict[str, float]) -> np.ndarray:
    """Order a feature dict into the model's expected column order."""
    return np.array([[float(row.get(col, 0.0)) for col in FEATURE_COLUMNS]], dtype=float)
