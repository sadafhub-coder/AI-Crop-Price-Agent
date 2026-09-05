"""
tools/market_api.py
-------------------
The data façade every agent talks to.

`MarketAPI` exposes exactly four verbs — get_price / get_history / get_trend /
get_prediction (plus a few conveniences) — and hides *where* the data comes
from behind a pluggable data source:

    LocalDatasetSource  -> the bundled CSV/SQLite history (default)
    RestMarketDataSource -> any real market-price REST API (e.g. data.gov.in
                            Agmarknet resource), configured purely via env vars

Because the agents only know the façade, swapping in the real API is a
one-line change and needs no edits inside the agents.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from models.predictor import PricePredictor, get_predictor
from tools import market_tools as mt
from tools.data_loader import load_data
from utils.helpers import (
    DataUnavailableError,
    coerce_horizon,
    err_from_exception,
    ok,
)


# ==========================================================================
# Data sources
# ==========================================================================


class BaseMarketDataSource:
    """Interface for anything that can hand back a price table."""

    name = "base"

    def frame(self, refresh: bool = False) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    def health(self) -> Dict[str, Any]:  # pragma: no cover
        return {"source": self.name, "ok": False}


class LocalDatasetSource(BaseMarketDataSource):
    """Reads the bundled historical dataset (CSV, or SQLite when present)."""

    name = "local_dataset"

    def __init__(self, source: str = "auto", path: Optional[str] = None):
        self.source = source
        self.path = path
        self._frame: Optional[pd.DataFrame] = None

    def frame(self, refresh: bool = False) -> pd.DataFrame:
        if self._frame is None or refresh:
            self._frame = load_data(
                source=self.source, path=self.path, use_cache=not refresh
            )
        return self._frame

    def health(self) -> Dict[str, Any]:
        try:
            frame = self.frame()
            return {
                "source": self.name,
                "ok": True,
                "rows": int(len(frame)),
                "latest_date": str(pd.to_datetime(frame["date"]).max().date()),
            }
        except Exception as exc:
            return {"source": self.name, "ok": False, "error": str(exc)}


class RestMarketDataSource(BaseMarketDataSource):
    """
    Live REST source with automatic fallback to the local dataset.

    Configure via .env (defaults target the Government of India Agmarknet
    resource on data.gov.in, which returns the same columns we model):

        MARKET_API_URL=https://api.data.gov.in/resource/<resource-id>
        MARKET_API_KEY=<your api key>
        MARKET_API_RECORD_LIMIT=5000
        MARKET_API_FIELD_MAP={"arrival_date":"date","commodity":"crop", ...}

    Any transport/parse failure is caught and the local dataset is used instead,
    so enabling the API can never take the app down.
    """

    name = "rest_api"

    #: Agmarknet-style response fields -> our schema.
    DEFAULT_FIELD_MAP = {
        "arrival_date": "date",
        "commodity": "crop",
        "market": "market",
        "state": "state",
        "arrivals_in_qtl": "arrival_quantity",
        "min_price": "price_min",
        "max_price": "price_max",
        "modal_price": "modal_price",
    }

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        limit: Optional[int] = None,
        field_map: Optional[Dict[str, str]] = None,
        fallback: Optional[BaseMarketDataSource] = None,
        timeout: float = 15.0,
    ):
        self.url = url or os.getenv("MARKET_API_URL", "")
        self.api_key = api_key or os.getenv("MARKET_API_KEY", "")
        self.limit = int(limit or os.getenv("MARKET_API_RECORD_LIMIT", "5000"))
        self.field_map = field_map or self._field_map_from_env()
        self.fallback = fallback or LocalDatasetSource()
        self.timeout = timeout
        self._frame: Optional[pd.DataFrame] = None
        self.last_error: Optional[str] = None
        self.using_fallback = False

    def _field_map_from_env(self) -> Dict[str, str]:
        import json

        raw = os.getenv("MARKET_API_FIELD_MAP")
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return dict(self.DEFAULT_FIELD_MAP)

    def configured(self) -> bool:
        return bool(self.url)

    def frame(self, refresh: bool = False) -> pd.DataFrame:
        if self._frame is not None and not refresh:
            return self._frame
        if self.configured():
            try:
                self._frame = self._fetch_remote()
                self.using_fallback = False
                self.last_error = None
                return self._frame
            except Exception as exc:
                self.last_error = str(exc)
        else:
            self.last_error = "MARKET_API_URL is not configured."
        self.using_fallback = True
        self._frame = self.fallback.frame(refresh=refresh)
        return self._frame

    def _fetch_remote(self) -> pd.DataFrame:
        import requests  # imported lazily: only needed for the live path

        params: Dict[str, Any] = {"format": "json", "limit": self.limit}
        if self.api_key:
            params["api-key"] = self.api_key
        response = requests.get(self.url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        records = (
            payload.get("records")
            if isinstance(payload, dict)
            else payload if isinstance(payload, list) else None
        )
        if not records:
            raise DataUnavailableError("Market API returned no records.")
        frame = pd.DataFrame(records).rename(columns=self.field_map)
        from utils.preprocessing import clean_dataframe

        clean = clean_dataframe(frame)
        if clean.empty:
            raise DataUnavailableError("Market API records could not be parsed.")
        return clean

    def health(self) -> Dict[str, Any]:
        frame = self.frame()
        return {
            "source": self.name,
            "ok": not self.using_fallback,
            "configured": self.configured(),
            "using_fallback": self.using_fallback,
            "error": self.last_error,
            "rows": int(len(frame)),
        }


def build_default_source() -> BaseMarketDataSource:
    """REST source when MARKET_API_URL is set, otherwise the local dataset."""
    if os.getenv("MARKET_API_URL"):
        return RestMarketDataSource()
    return LocalDatasetSource()


# ==========================================================================
# The façade
# ==========================================================================


class MarketAPI:
    """
    Clean access layer used by every agent.

        api = MarketAPI()
        api.get_price("wheat", "Aurangabad")
        api.get_history("wheat", "Aurangabad", days=90)
        api.get_trend("wheat", "Aurangabad")
        api.get_prediction("wheat", "Aurangabad", days=7)

    Every method returns a result dict with `ok: True/False` — agents branch on
    that instead of handling exceptions.
    """

    def __init__(
        self,
        source: Optional[BaseMarketDataSource] = None,
        predictor: Optional[PricePredictor] = None,
    ):
        self.source = source or build_default_source()
        self.predictor = predictor or get_predictor()

    # -------------------------------------------------------------- plumbing
    def frame(self, refresh: bool = False) -> pd.DataFrame:
        return self.source.frame(refresh=refresh)

    def refresh(self) -> None:
        self.source.frame(refresh=True)

    def health(self) -> Dict[str, Any]:
        data_health = self.source.health()
        return {
            "data": data_health,
            "model": self.predictor.info(),
        }

    def crops(self) -> List[str]:
        try:
            return mt.list_crops(self.frame())
        except Exception:
            return []

    def markets(self, crop: Optional[str] = None) -> List[str]:
        try:
            frame = self.frame()
            resolved = mt.resolve_crop(frame, crop) if crop else None
            return mt.list_markets(frame, resolved)
        except Exception:
            return []

    def resolve(self, crop: str, market: Optional[str] = None) -> Dict[str, str]:
        """Canonicalise crop/market names; picks a default market when absent."""
        frame = self.frame()
        crop_name = mt.resolve_crop(frame, crop)
        market_name = (
            mt.resolve_market(frame, market, crop_name)
            if market
            else mt.default_market(frame, crop_name)
        )
        return {"crop": crop_name, "market": market_name}

    # ------------------------------------------------------------- get_price
    def get_price(self, crop: str, market: Optional[str] = None) -> Dict[str, Any]:
        """Latest available price for a crop (in one market, or the best-covered)."""
        try:
            frame = self.frame()
            crop_name = mt.resolve_crop(frame, crop)
            market_name = (
                mt.resolve_market(frame, market, crop_name)
                if market
                else mt.default_market(frame, crop_name)
            )
            record = mt.latest_record(frame, crop_name, market_name)
            return ok(**record, market_defaulted=market is None)
        except Exception as exc:
            return err_from_exception(exc, crop=crop, market=market)

    def get_prices_all_markets(self, crop: str) -> Dict[str, Any]:
        """Latest price in every market trading this crop (best first)."""
        try:
            frame = self.frame()
            crop_name = mt.resolve_crop(frame, crop)
            rows = mt.compare_markets_table(frame, crop_name)
            best, worst = rows[0], rows[-1]
            return ok(
                crop=crop_name,
                markets=rows,
                best_market=best["market"],
                best_price=best["modal_price"],
                lowest_market=worst["market"],
                lowest_price=worst["modal_price"],
            )
        except Exception as exc:
            return err_from_exception(exc, crop=crop)

    # ----------------------------------------------------------- get_history
    def get_history(
        self,
        crop: str,
        market: Optional[str] = None,
        days: Optional[int] = 180,
        date_from: Optional[Any] = None,
        date_to: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Historical price series as records (+ the raw frame for charts)."""
        try:
            frame = self.frame()
            crop_name = mt.resolve_crop(frame, crop)
            market_name = (
                mt.resolve_market(frame, market, crop_name)
                if market
                else mt.default_market(frame, crop_name)
            )
            series = mt.get_series(
                frame,
                crop_name,
                market_name,
                days=days,
                date_from=date_from,
                date_to=date_to,
            )
            return ok(
                crop=crop_name,
                market=market_name,
                state=str(series.iloc[-1]["state"]),
                points=int(len(series)),
                date_from=str(series["date"].min().date()),
                date_to=str(series["date"].max().date()),
                history=mt.price_history_records(series),
                frame=series,
            )
        except Exception as exc:
            return err_from_exception(exc, crop=crop, market=market)

    # ------------------------------------------------------------- get_trend
    def get_trend(
        self, crop: str, market: Optional[str] = None, window: int = 30
    ) -> Dict[str, Any]:
        """Direction, % change and volatility over the last `window` days."""
        try:
            frame = self.frame()
            crop_name = mt.resolve_crop(frame, crop)
            market_name = (
                mt.resolve_market(frame, market, crop_name)
                if market
                else mt.default_market(frame, crop_name)
            )
            series = mt.get_series(frame, crop_name, market_name)
            stats = mt.compute_trend_stats(series, window=window)
            return ok(crop=crop_name, market=market_name, **stats)
        except Exception as exc:
            return err_from_exception(exc, crop=crop, market=market)

    # -------------------------------------------------------- get_prediction
    def get_prediction(
        self, crop: str, market: Optional[str] = None, days: int = 7
    ) -> Dict[str, Any]:
        """Forecast the next `days` days for a crop+market."""
        try:
            horizon = coerce_horizon(days)
            frame = self.frame()
            crop_name = mt.resolve_crop(frame, crop)
            market_name = (
                mt.resolve_market(frame, market, crop_name)
                if market
                else mt.default_market(frame, crop_name)
            )
            series = mt.get_series(frame, crop_name, market_name)
            result = self.predictor.predict_future(
                series,
                days=horizon,
                crop=crop_name,
                market=market_name,
                state=str(series.iloc[-1]["state"]),
            )
            return ok(**result)
        except Exception as exc:
            return err_from_exception(exc, crop=crop, market=market, days=days)

    # ---------------------------------------------------------- conveniences
    def compare_markets(
        self, crop: str, markets: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        try:
            frame = self.frame()
            crop_name = mt.resolve_crop(frame, crop)
            resolved = (
                [mt.resolve_market(frame, m, crop_name) for m in markets]
                if markets
                else None
            )
            rows = mt.compare_markets_table(frame, crop_name, resolved)
            return ok(
                crop=crop_name,
                markets=rows,
                best_market=rows[0]["market"],
                best_price=rows[0]["modal_price"],
                lowest_market=rows[-1]["market"],
                lowest_price=rows[-1]["modal_price"],
            )
        except Exception as exc:
            return err_from_exception(exc, crop=crop)

    def compare_crops(self, crops: Sequence[str], market: Optional[str] = None) -> Dict[str, Any]:
        """Side-by-side latest price + trend for several crops."""
        rows: List[Dict[str, Any]] = []
        errors: List[str] = []
        for crop in crops:
            price = self.get_price(crop, market)
            if not price["ok"]:
                errors.append(price["error"])
                continue
            trend = self.get_trend(crop, price["market"])
            rows.append(
                {
                    "crop": price["crop"],
                    "market": price["market"],
                    "modal_price": price["modal_price"],
                    "date": price["date"],
                    "trend": trend.get("direction") if trend["ok"] else None,
                    "change_pct": trend.get("change_pct") if trend["ok"] else None,
                }
            )
        if not rows:
            return {
                "ok": False,
                "error": errors[0] if errors else "No comparable crops found.",
                "code": "no_data",
            }
        rows.sort(key=lambda r: (r["modal_price"] or 0), reverse=True)
        return ok(rows=rows, errors=errors, highest=rows[0], lowest=rows[-1])

    def highest_priced_crops(self, limit: int = 5) -> Dict[str, Any]:
        try:
            return ok(rows=mt.top_crops_by_price(self.frame(), limit=limit))
        except Exception as exc:
            return err_from_exception(exc)


_default_api: Optional[MarketAPI] = None


def get_market_api() -> MarketAPI:
    """Shared MarketAPI instance (keeps the dataset and model loaded once)."""
    global _default_api
    if _default_api is None:
        _default_api = MarketAPI()
    return _default_api
