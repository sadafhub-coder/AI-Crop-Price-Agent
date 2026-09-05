"""
tests/test_supervisor.py
------------------------
Query Parser + Supervisor routing: every intent in the spec reaches the right
agent, replies are farmer-readable, and no user mistake crashes the chatbot.
"""

from __future__ import annotations

import pytest

from agents.query_parser import INTENTS, QueryParserAgent


@pytest.fixture(scope="module")
def parser(api):
    return QueryParserAgent(
        known_crops=api.crops(), known_markets=api.markets(), use_gemini=False
    )


# --------------------------------------------------------------- query parsing
@pytest.mark.parametrize(
    "query,intent",
    [
        ("What is the current price of wheat?", "current_price"),
        ("Predict wheat price for the next 7 days.", "prediction"),
        ("What will be the price of onion next month?", "prediction"),
        ("Which crop has the highest price currently?", "highest_price"),
        ("Show me the price trend of rice.", "trend"),
        ("Should I sell my wheat now or wait?", "recommendation"),
        ("Compare wheat and rice prices.", "compare_crops"),
        ("Which market has the best price for tomatoes?", "compare_markets"),
        ("What is the expected price of maize next week?", "prediction"),
    ],
)
def test_spec_example_questions_are_classified(parser, query, intent):
    parsed = parser.parse(query)
    assert parsed["ok"] is True
    assert parsed["intent"] == intent, f"{query} -> {parsed['intent']}"


@pytest.mark.parametrize(
    "query,crop",
    [
        ("price of wheat", "Wheat"),
        ("gehu ka bhav kya hai", "Wheat"),
        ("pyaz price today", "Onion"),
        ("tamatar rate in pune", "Tomato"),
        ("paddy price", "Rice"),
    ],
)
def test_crop_extraction_including_local_names(parser, query, crop):
    assert parser.parse(query)["crop"] == crop


def test_market_extraction(parser):
    parsed = parser.parse("wheat price in Aurangabad")
    assert parsed["crop"] == "Wheat" and parsed["market"] == "Aurangabad"


@pytest.mark.parametrize(
    "query,days",
    [
        ("predict wheat next 7 days", 7),
        ("predict wheat for next 15 days", 15),
        ("onion price next month", 30),
        ("maize price next week", 7),
        ("wheat forecast for next 3 months", 90),
    ],
)
def test_prediction_period_extraction(parser, query, days):
    assert parser.parse(query)["days"] == days


def test_oversized_period_is_capped_with_a_note(parser):
    parsed = parser.parse("predict wheat price for next 500 days")
    assert parsed["days"] == 90
    assert parsed["notes"], "user should be told the horizon was capped"


def test_date_range_extraction(parser):
    parsed = parser.parse("show rice prices for the last 3 months")
    assert parsed["date_from"] and parsed["date_to"]


def test_multi_crop_extraction_order(parser):
    parsed = parser.parse("compare wheat and rice prices")
    assert parsed["crops"][:2] == ["Wheat", "Rice"]


def test_empty_query_is_rejected(parser):
    for bad in ["", "   ", None]:
        parsed = parser.parse(bad)
        assert parsed["ok"] is False and parsed["code"] == "empty_query"


def test_unknown_crop_yields_unknown_intent(parser):
    parsed = parser.parse("what is the price of moon cheese")
    assert parsed["ok"] is True
    assert parsed["intent"] in INTENTS
    assert parsed["crop"] is None


def test_parser_works_without_gemini(parser):
    """No API key configured in tests: the rule-based path must be used."""
    parsed = parser.parse("predict wheat price for next 7 days")
    assert parsed["source"] == "rules"


# ------------------------------------------------------------------- routing
def test_current_price_routing(supervisor):
    result = supervisor.handle("What is the current price of wheat?")
    assert result["ok"] is True
    assert "market_agent" in result["agents_used"]
    assert "₹" in result["reply"] and "Wheat" in result["reply"]


def test_prediction_routing_calls_multiple_agents(supervisor):
    result = supervisor.handle("Predict wheat price for the next 7 days in Aurangabad.")
    assert result["ok"] is True
    for agent in ("prediction_agent", "trend_agent", "recommendation_agent"):
        assert agent in result["agents_used"]
    assert "Day 1" in result["reply"]
    assert "not guaranteed" in result["reply"].lower()
    assert result["data"]["prediction"]["market"] == "Aurangabad"


def test_trend_routing(supervisor):
    result = supervisor.handle("Show me the price trend of rice.")
    assert result["ok"] is True
    assert "trend_agent" in result["agents_used"]
    assert "Trend" in result["reply"]


def test_recommendation_routing(supervisor):
    result = supervisor.handle("Should I sell my wheat now or wait?")
    assert result["ok"] is True
    assert "recommendation_agent" in result["agents_used"]
    assert "Suggestion" in result["reply"]


def test_market_comparison_routing(supervisor):
    result = supervisor.handle("Which market has the best price for tomatoes?")
    assert result["ok"] is True
    assert "Best price" in result["reply"]
    assert result["data"]["comparison"]["crop"] == "Tomato"


def test_crop_comparison_routing(supervisor):
    result = supervisor.handle("Compare wheat and rice prices.")
    assert result["ok"] is True
    rows = result["data"]["crop_comparison"]["rows"]
    assert {row["crop"] for row in rows} == {"Wheat", "Rice"}


def test_highest_price_routing(supervisor):
    result = supervisor.handle("Which crop has the highest price currently?")
    assert result["ok"] is True
    assert "Highest Priced Crops" in result["reply"]


def test_market_info_routing(supervisor):
    result = supervisor.handle("Which crops do you have data for?")
    assert result["ok"] is True and "Crops:" in result["reply"]


def test_help_routing(supervisor):
    result = supervisor.handle("what can you do?")
    assert result["ok"] is True and "Try asking me" in result["reply"]


# ------------------------------------------------------------ error handling
def test_empty_question_is_handled(supervisor):
    result = supervisor.handle("")
    assert result["ok"] is False
    assert "question" in result["reply"].lower()


def test_invalid_crop_lists_valid_options(supervisor):
    result = supervisor.handle("what is the price of unicorn horn")
    assert result["ok"] is True  # answered with guidance, not an exception
    assert "I have prices for" in result["reply"] or "Available crops" in result["reply"]


def test_invalid_market_is_explained(supervisor):
    result = supervisor.handle("wheat price in Atlantis")
    assert isinstance(result["reply"], str) and result["reply"]


def test_gibberish_never_crashes(supervisor):
    for query in ["asdkjhasd", "!!!!", "12345", "मुझे भाव बताओ", "🌾🌾🌾"]:
        result = supervisor.handle(query)
        assert isinstance(result["reply"], str) and result["reply"]


def test_every_reply_is_a_string_with_intent(supervisor):
    for query in [
        "wheat price", "predict onion 30 days", "rice trend",
        "should i sell tomato", "compare maize and cotton", "best market for potato",
    ]:
        result = supervisor.handle(query)
        assert isinstance(result["reply"], str) and len(result["reply"]) > 10
        assert result["intent"] in INTENTS + ("invalid",)


# ---------------------------------------------------------------- dashboard
def test_dashboard_payload_is_complete(supervisor, sample_crop, sample_market):
    payload = supervisor.dashboard_payload(sample_crop, sample_market, days=7)
    assert payload["ok"] is True
    for key in ("price", "trend", "prediction", "history", "comparison", "recommendation"):
        assert payload[key] is not None, f"{key} missing from dashboard payload"
    assert payload["errors"] == []


def test_dashboard_payload_invalid_crop(supervisor):
    payload = supervisor.dashboard_payload("moonrock", None)
    assert payload["ok"] is False and payload["code"] == "invalid_crop"


def test_health_reports_all_three_subsystems(supervisor):
    health = supervisor.health()
    assert set(health) == {"data", "model", "gemini"}
    assert health["data"]["ok"] is True
    assert health["gemini"]["available"] is False  # no key in tests
