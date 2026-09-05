"""
tests/conftest.py
-----------------
Shared pytest fixtures.

Key choices:
  * `sys.path` gets the project root so `import agents...` works when pytest is
    run from anywhere.
  * Gemini is disabled for the whole test session (`GEMINI_API_KEY=""`), so tests
    exercise the deterministic fallback paths and never make network calls.
  * The MarketAPI / SupervisorAgent are session-scoped: the dataset and model are
    loaded once for the entire suite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# No network, no key: every agent must work on its fallback path.
os.environ["GEMINI_API_KEY"] = ""
os.environ.pop("MARKET_API_URL", None)


@pytest.fixture(scope="session")
def data_frame():
    from tools.data_loader import load_data

    return load_data(source="csv")


@pytest.fixture(scope="session")
def api():
    from tools.market_api import LocalDatasetSource, MarketAPI

    return MarketAPI(source=LocalDatasetSource(source="csv"))


@pytest.fixture(scope="session")
def supervisor(api):
    from agents.supervisor import SupervisorAgent

    return SupervisorAgent(api=api, use_gemini=False)


@pytest.fixture(scope="session")
def sample_crop(api):
    crops = api.crops()
    return "Wheat" if "Wheat" in crops else crops[0]


@pytest.fixture(scope="session")
def sample_market(api, sample_crop):
    return api.markets(sample_crop)[0]
