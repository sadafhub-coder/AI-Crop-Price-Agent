"""
Agent layer for the AI Crop Price Prediction & Market Insight Chatbot.

    QueryParserAgent      -> natural language -> structured intent
    MarketPriceAgent      -> current / latest prices
    TrendAgent            -> historical trend + insights
    PredictionAgent       -> ML forecasts (7 / 15 / 30 / 90 days)
    RecommendationAgent   -> sell / wait / watch guidance (never guaranteed)
    SupervisorAgent       -> routes and merges everything into one answer

Usage:
    from agents.supervisor import SupervisorAgent
    print(SupervisorAgent().handle("predict wheat price for next 7 days")["reply"])
"""

from agents.market_agent import MarketPriceAgent  # noqa: F401
from agents.prediction_agent import PredictionAgent  # noqa: F401
from agents.query_parser import QueryParserAgent  # noqa: F401
from agents.recommendation_agent import RecommendationAgent  # noqa: F401
from agents.supervisor import SupervisorAgent  # noqa: F401
from agents.trend_agent import TrendAgent  # noqa: F401

__all__ = [
    "QueryParserAgent",
    "MarketPriceAgent",
    "TrendAgent",
    "PredictionAgent",
    "RecommendationAgent",
    "SupervisorAgent",
]
