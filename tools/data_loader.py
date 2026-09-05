"""
tools/data_loader.py
--------------------
The single door to historical price data.

Responsibilities:
  * generate a realistic sample dataset when none exists (so the project runs
    on a fresh clone with zero setup);
  * load from CSV or SQLite;
  * validate the schema and hand back a cleaned DataFrame;
  * cache in-process so Streamlit reruns stay fast.

Swap in a real data source by pointing CROP_DATA_CSV / CROP_DATA_SQLITE at it,
or by using tools.market_api.RestMarketDataSource.
"""

from __future__ import annotations

import datetime as _dt
import os
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allows `python tools/data_loader.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import DataUnavailableError  # noqa: E402
from utils.preprocessing import REQUIRED_COLUMNS, clean_dataframe  # noqa: E402

DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "crop_prices.csv"
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "crop_prices.db"
SQLITE_TABLE = "crop_prices"

# --------------------------------------------------------------------------
# Sample-data definition: crop -> (base ₹/quintal, annual volatility, seasonality
# amplitude, yearly drift, harvest months where prices dip)
# --------------------------------------------------------------------------
CROP_PROFILE: Dict[str, dict] = {
    "Wheat":     {"base": 2450, "vol": 0.010, "season_amp": 0.06, "drift": 0.05, "harvest": (3, 4)},
    "Rice":      {"base": 3150, "vol": 0.009, "season_amp": 0.05, "drift": 0.04, "harvest": (10, 11)},
    "Maize":     {"base": 2100, "vol": 0.013, "season_amp": 0.08, "drift": 0.06, "harvest": (9, 10)},
    "Cotton":    {"base": 7200, "vol": 0.014, "season_amp": 0.07, "drift": 0.03, "harvest": (11, 12)},
    "Soybean":   {"base": 4600, "vol": 0.016, "season_amp": 0.09, "drift": 0.02, "harvest": (10, 11)},
    "Onion":     {"base": 1800, "vol": 0.034, "season_amp": 0.28, "drift": 0.08, "harvest": (4, 5)},
    "Tomato":    {"base": 1600, "vol": 0.045, "season_amp": 0.35, "drift": 0.07, "harvest": (2, 3)},
    "Sugarcane": {"base": 340,  "vol": 0.005, "season_amp": 0.03, "drift": 0.03, "harvest": (12, 1)},
    "Potato":    {"base": 1250, "vol": 0.026, "season_amp": 0.22, "drift": 0.05, "harvest": (2, 3)},
    "Pulses":    {"base": 6100, "vol": 0.012, "season_amp": 0.07, "drift": 0.06, "harvest": (3, 4)},
}

#: market -> state, plus a per-market price multiplier (freight/demand premium)
MARKET_PROFILE: Dict[str, Tuple[str, float]] = {
    "Aurangabad": ("Maharashtra", 1.00),
    "Pune":       ("Maharashtra", 1.045),
    "Nashik":     ("Maharashtra", 1.020),
    "Nagpur":     ("Maharashtra", 0.985),
    "Indore":     ("Madhya Pradesh", 0.975),
    "Bhopal":     ("Madhya Pradesh", 0.990),
    "Ludhiana":   ("Punjab", 1.035),
    "Karnal":     ("Haryana", 1.010),
    "Jaipur":     ("Rajasthan", 0.995),
    "Bengaluru":  ("Karnataka", 1.060),
}

#: Which markets trade which crop (keeps the sample data plausible).
CROP_MARKETS: Dict[str, List[str]] = {
    "Wheat":     ["Aurangabad", "Indore", "Ludhiana", "Karnal", "Bhopal"],
    "Rice":      ["Karnal", "Ludhiana", "Nagpur", "Bengaluru"],
    "Maize":     ["Nashik", "Indore", "Bengaluru", "Nagpur"],
    "Cotton":    ["Aurangabad", "Nagpur", "Jaipur", "Indore"],
    "Soybean":   ["Indore", "Bhopal", "Nagpur", "Aurangabad"],
    "Onion":     ["Nashik", "Pune", "Bengaluru", "Indore"],
    "Tomato":    ["Bengaluru", "Pune", "Nashik", "Jaipur"],
    "Sugarcane": ["Pune", "Karnal", "Aurangabad", "Bhopal"],
    "Potato":    ["Bhopal", "Jaipur", "Pune", "Karnal"],
    "Pulses":    ["Jaipur", "Indore", "Aurangabad", "Bengaluru"],
}

_CACHE: Dict[str, pd.DataFrame] = {}
_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Sample dataset generation
# --------------------------------------------------------------------------


def generate_sample_dataset(
    path: Optional[Path] = None,
    start: Optional[_dt.date] = None,
    end: Optional[_dt.date] = None,
    seed: int = 42,
    write: bool = True,
) -> pd.DataFrame:
    """
    Build ~20 months of daily mandi records for 10 crops across 10 markets.

    The generator is deterministic (seeded) and models the things that actually
    move mandi prices: a slow yearly drift, a seasonal harvest dip, a random
    walk, weekly arrival cycles and occasional supply shocks for perishables.
    """
    path = Path(path or DEFAULT_CSV_PATH)
    end = end or _dt.date.today()
    start = start or (end - _dt.timedelta(days=600))
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start, end=end, freq="D")
    day_index = np.arange(len(dates))
    rows = []

    for crop, profile in CROP_PROFILE.items():
        for market in CROP_MARKETS[crop]:
            state, market_mult = MARKET_PROFILE[market]
            base = profile["base"] * market_mult

            # Persistent random walk (mean-reverting) around the seasonal path.
            shocks = rng.normal(0.0, profile["vol"], len(dates))
            walk = np.zeros(len(dates))
            for i in range(1, len(dates)):
                walk[i] = 0.94 * walk[i - 1] + shocks[i]

            # Perishables get occasional supply shocks (onion/tomato/potato).
            if profile["vol"] > 0.02:
                for shock_day in rng.choice(len(dates), size=6, replace=False):
                    magnitude = rng.uniform(0.12, 0.30) * rng.choice([-1, 1])
                    decay = np.exp(-np.arange(len(dates) - shock_day) / 12.0)
                    walk[shock_day:] += magnitude * decay

            seasonal = profile["season_amp"] * np.sin(
                2 * np.pi * (dates.dayofyear.to_numpy() - 30) / 365.25
            )
            harvest_dip = np.where(
                np.isin(dates.month.to_numpy(), profile["harvest"]),
                -profile["season_amp"] * 0.9,
                0.0,
            )
            drift = profile["drift"] * (day_index / 365.25)

            modal = base * (1 + seasonal + harvest_dip + drift + walk)
            modal = np.clip(modal, base * 0.55, base * 1.9)

            spread = rng.uniform(0.03, 0.07, len(dates))
            price_min = modal * (1 - spread)
            price_max = modal * (1 + spread)

            # Arrivals: weekly cycle, heavier at harvest, inverse to price.
            weekly = 1 + 0.25 * np.sin(2 * np.pi * dates.dayofweek.to_numpy() / 7)
            harvest_push = np.where(
                np.isin(dates.month.to_numpy(), profile["harvest"]), 1.8, 1.0
            )
            arrivals = (
                rng.uniform(80, 260, len(dates))
                * weekly
                * harvest_push
                * (base / np.maximum(modal, 1))
            )

            frame = pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "crop": crop,
                    "market": market,
                    "state": state,
                    "arrival_quantity": np.round(arrivals, 1),
                    "price_min": np.round(price_min, 0),
                    "price_max": np.round(price_max, 0),
                    "modal_price": np.round(modal, 0),
                }
            )
            # Mandis are closed on some Sundays and holidays -> realistic gaps.
            closed = (dates.dayofweek.to_numpy() == 6) & (
                rng.random(len(dates)) < 0.55
            )
            frame = frame[~closed]
            rows.append(frame)

    dataset = pd.concat(rows, ignore_index=True)
    dataset = dataset.sort_values(["date", "crop", "market"]).reset_index(drop=True)

    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(path, index=False)
    return dataset


# --------------------------------------------------------------------------
# Loading / validation
# --------------------------------------------------------------------------


def validate_schema(df: pd.DataFrame) -> None:
    """Raise DataUnavailableError when the table cannot be used at all."""
    if df is None or df.empty:
        raise DataUnavailableError("The price dataset is empty.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataUnavailableError(
            "Dataset is missing required column(s): " + ", ".join(missing)
        )


def csv_path() -> Path:
    return Path(os.getenv("CROP_DATA_CSV", str(DEFAULT_CSV_PATH)))


def sqlite_path() -> Path:
    return Path(os.getenv("CROP_DATA_SQLITE", str(DEFAULT_SQLITE_PATH)))


def load_csv(path: Optional[Path] = None, auto_generate: bool = True) -> pd.DataFrame:
    path = Path(path or csv_path())
    if not path.exists():
        if not auto_generate:
            raise DataUnavailableError(f"Price data file not found: {path.name}")
        generate_sample_dataset(path)
    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        raise DataUnavailableError(f"Could not read the price data file: {exc}") from exc
    validate_schema(raw)
    return raw


def load_sqlite(path: Optional[Path] = None, table: str = SQLITE_TABLE) -> pd.DataFrame:
    path = Path(path or sqlite_path())
    if not path.exists():
        raise DataUnavailableError(f"SQLite database not found: {path.name}")
    try:
        with sqlite3.connect(str(path)) as conn:
            raw = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception as exc:
        raise DataUnavailableError(f"Could not read SQLite data: {exc}") from exc
    validate_schema(raw)
    return raw


def export_to_sqlite(
    df: Optional[pd.DataFrame] = None,
    path: Optional[Path] = None,
    table: str = SQLITE_TABLE,
) -> Path:
    """Persist the dataset to SQLite (handy once the CSV gets large)."""
    df = df if df is not None else load_data()
    path = Path(path or sqlite_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)
    return path


def load_data(
    source: str = "auto",
    path: Optional[Path] = None,
    use_cache: bool = True,
    auto_generate: bool = True,
) -> pd.DataFrame:
    """
    Load + clean historical prices.

    source: "auto" (SQLite if present, else CSV), "csv" or "sqlite".
    Returns a cleaned DataFrame with a real `date` dtype and a `season` column.
    """
    cache_key = f"{source}:{path or ''}"
    if use_cache:
        with _LOCK:
            cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()

    if source == "sqlite":
        raw = load_sqlite(path)
    elif source == "csv":
        raw = load_csv(path, auto_generate=auto_generate)
    else:  # auto
        if path is None and sqlite_path().exists():
            raw = load_sqlite()
        else:
            raw = load_csv(path, auto_generate=auto_generate)

    clean = clean_dataframe(raw)
    if clean.empty:
        raise DataUnavailableError("No usable price rows after cleaning the dataset.")

    with _LOCK:
        _CACHE[cache_key] = clean.copy()
    return clean


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def dataset_summary(df: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    """Small stats block for the README / sidebar / tests."""
    df = df if df is not None else load_data()
    return {
        "rows": int(len(df)),
        "crops": sorted(df["crop"].unique().tolist()),
        "markets": sorted(df["market"].unique().tolist()),
        "states": sorted(df["state"].unique().tolist()),
        "date_from": str(pd.to_datetime(df["date"]).min().date()),
        "date_to": str(pd.to_datetime(df["date"]).max().date()),
        "series": int(df.groupby(["crop", "market"]).ngroups),
    }


if __name__ == "__main__":  # `python tools/data_loader.py` regenerates the CSV
    frame = generate_sample_dataset()
    print(f"Wrote {len(frame):,} rows to {DEFAULT_CSV_PATH}")
    print(dataset_summary(clean_dataframe(frame)))
