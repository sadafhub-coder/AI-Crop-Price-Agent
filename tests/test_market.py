"""
tests/test_market.py
--------------------
Market Price Agent + MarketAPI + data loader:
price retrieval, name resolution, market comparison, and every documented
error case (invalid crop, invalid market, missing data).
"""

from __future__ import annotations

import pandas as pd
import pytest

from agents.market_agent import MarketPriceAgent
from tools import market_tools as mt
from tools.data_loader import dataset_summary, generate_sample_dataset
from utils.helpers import InvalidCropError, InvalidMarketError
from utils.preprocessing import REQUIRED_COLUMNS, clean_dataframe

EXPECTED_CROPS = {
    "Wheat", "Rice", "Maize", "Cotton", "Soybean",
    "Onion", "Tomato", "Sugarcane", "Potato", "Pulses",
}


# ----------------------------------------------------------------- dataset
def test_dataset_has_required_columns(data_frame):
    for column in REQUIRED_COLUMNS:
        assert column in data_frame.columns


def test_dataset_covers_all_ten_crops(data_frame):
    assert EXPECTED_CROPS.issubset(set(data_frame["crop"].unique()))


def test_dataset_has_months_of_history(data_frame):
    span = (data_frame["date"].max() - data_frame["date"].min()).days
    assert span > 180, "need at least ~6 months of history to train meaningfully"


def test_dataset_prices_are_positive_and_ordered(data_frame):
    assert (data_frame["modal_price"] > 0).all()
    assert (data_frame["price_min"] <= data_frame["price_max"]).all()


def test_dataset_summary_shape(data_frame):
    summary = dataset_summary(data_frame)
    assert summary["rows"] == len(data_frame)
    assert summary["series"] > 20


def test_cleaning_repairs_broken_rows():
    """Missing modal price, inverted min/max, junk date and blank crop."""
    dirty = pd.DataFrame(
        {
            "date": ["2026-01-01", "not-a-date", "2026-01-03", "2026-01-04"],
            "crop": ["wheat ", "Wheat", "  ", "wheat"],
            "market": ["pune", "Pune", "Pune", "Pune"],
            "state": ["Maharashtra"] * 4,
            "arrival_quantity": [100, 100, 100, None],
            "price_min": [2600, 2000, 2000, 2400],
            "price_max": [2400, 2100, 2100, 2500],  # row 0 inverted
            "modal_price": [None, 2050, 2050, 2450],  # row 0 missing
        }
    )
    clean = clean_dataframe(dirty)
    assert len(clean) == 2  # junk date and blank crop dropped
    assert (clean["price_min"] <= clean["price_max"]).all()
    assert clean["modal_price"].notna().all()
    assert set(clean["crop"]) == {"Wheat"}


def test_generator_is_deterministic(tmp_path):
    first = generate_sample_dataset(tmp_path / "a.csv", seed=7)
    second = generate_sample_dataset(tmp_path / "b.csv", seed=7)
    pd.testing.assert_frame_equal(first, second)


# ------------------------------------------------------------ name resolution
@pytest.mark.parametrize(
    "given,expected",
    [("wheat", "Wheat"), ("WHEAT", "Wheat"), ("gehu", "Wheat"), ("wheet", "Wheat"),
     ("pyaz", "Onion"), ("tamatar", "Tomato"), ("paddy", "Rice"), ("corn", "Maize")],
)
def test_resolve_crop_handles_aliases_and_typos(data_frame, given, expected):
    assert mt.resolve_crop(data_frame, given) == expected


def test_resolve_market_handles_aliases(data_frame):
    if "Bengaluru" in mt.list_markets(data_frame):
        assert mt.resolve_market(data_frame, "bangalore") == "Bengaluru"


def test_resolve_crop_rejects_unknown(data_frame):
    with pytest.raises(InvalidCropError) as info:
        mt.resolve_crop(data_frame, "dragonfruit")
    assert "Available crops" in str(info.value)


def test_resolve_market_rejects_unknown(data_frame):
    with pytest.raises(InvalidMarketError):
        mt.resolve_market(data_frame, "Atlantis")


# --------------------------------------------------------------- MarketAPI
def test_get_price_returns_latest_record(api, sample_crop, sample_market):
    result = api.get_price(sample_crop, sample_market)
    assert result["ok"] is True
    assert result["crop"] == sample_crop and result["market"] == sample_market
    assert result["modal_price"] > 0
    assert result["price_min"] <= result["modal_price"] <= result["price_max"]


def test_get_price_defaults_the_market(api, sample_crop):
    result = api.get_price(sample_crop)
    assert result["ok"] is True and result["market_defaulted"] is True


def test_get_price_invalid_crop_is_graceful(api):
    result = api.get_price("unobtainium")
    assert result["ok"] is False and result["code"] == "invalid_crop"


def test_get_price_invalid_market_is_graceful(api, sample_crop):
    result = api.get_price(sample_crop, "Atlantis")
    assert result["ok"] is False and result["code"] == "invalid_market"


def test_get_history_returns_series(api, sample_crop, sample_market):
    result = api.get_history(sample_crop, sample_market, days=90)
    assert result["ok"] is True and result["points"] > 30
    assert all("modal_price" in record for record in result["history"])


def test_get_history_empty_range_is_graceful(api, sample_crop, sample_market):
    result = api.get_history(
        sample_crop, sample_market, days=None, date_from="1990-01-01", date_to="1990-02-01"
    )
    assert result["ok"] is False and result["code"] == "no_data"


def test_health_reports_data_source(api):
    health = api.health()
    assert health["data"]["ok"] is True
    assert "model" in health


# ----------------------------------------------------------- MarketPriceAgent
def test_agent_current_price_summary(api, sample_crop, sample_market):
    agent = MarketPriceAgent(api)
    result = agent.get_current_price(sample_crop, sample_market)
    assert result["ok"] is True
    assert "₹" in result["summary"] and sample_market in result["summary"]


def test_agent_requires_a_crop(api):
    result = MarketPriceAgent(api).get_current_price("")
    assert result["ok"] is False and result["code"] == "missing_crop"


def test_agent_market_comparison_is_sorted_best_first(api, sample_crop):
    result = MarketPriceAgent(api).get_market_comparison(sample_crop)
    assert result["ok"] is True and len(result["markets"]) >= 2
    prices = [row["modal_price"] for row in result["markets"]]
    assert prices == sorted(prices, reverse=True)
    assert result["best_market"] == result["markets"][0]["market"]
    assert result["price_gap"] >= 0


def test_agent_multi_crop_prices_skip_bad_names(api, sample_crop):
    result = MarketPriceAgent(api).get_prices([sample_crop, "notacrop"])
    assert result["ok"] is True
    assert len(result["rows"]) == 1 and len(result["problems"]) == 1


def test_agent_highest_priced_crop(api):
    result = MarketPriceAgent(api).highest_priced_crop(limit=5)
    assert result["ok"] is True and len(result["rows"]) == 5
    averages = [row["average_price"] for row in result["rows"]]
    assert averages == sorted(averages, reverse=True)


def test_explain_falls_back_without_gemini(api, sample_crop, sample_market):
    """With no API key, explain() must return the deterministic summary."""
    agent = MarketPriceAgent(api)
    price = agent.get_current_price(sample_crop, sample_market)
    assert agent.explain(price) == price["summary"]
