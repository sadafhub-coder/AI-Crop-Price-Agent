# 🌾 AI Crop Price Prediction & Market Insight Chatbot

A production-ready, multi-agent chatbot that answers real questions about Indian mandi
(agricultural market) prices — current rates, historical trends, ML price forecasts, market
comparison and sell/wait guidance — through a Streamlit dashboard.

Ask it things like *"Predict wheat price for the next 7 days in Aurangabad"* or *"Should I sell my
onion now or wait?"* and it routes the question to the right specialist agent, runs the numbers and
answers in language a farmer can act on.

**Works with no API key.** Gemini adds AI-written explanations and stronger language
understanding, but every agent has a deterministic fallback, so the app never breaks without it.

---

## Table of contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Folder structure](#folder-structure)
4. [Installation](#installation)
5. [Environment variables](#environment-variables)
6. [Dataset format](#dataset-format)
7. [Model training](#model-training)
8. [Running the app](#running-the-app)
9. [Testing](#testing)
10. [Replacing the sample data with a real API](#replacing-the-sample-data-with-a-real-api)
11. [Example chatbot questions](#example-chatbot-questions)
12. [Error handling](#error-handling)
13. [Extending the project](#extending-the-project)

---

## Features

| Area | What you get |
|---|---|
| **Chatbot** | Natural-language Q&A over prices, trends, forecasts, comparisons and selling advice |
| **6 agents** | Query Parser, Market Price, Trend, Prediction, Recommendation, Supervisor |
| **ML pipeline** | Cleaning → time/lag/rolling features → chronological split → Random Forest → MAE/RMSE/R² → saved model |
| **Forecasts** | 7 / 15 / 30 / 90-day horizons (any length up to 90) with uncertainty bands |
| **Dashboard** | 6 KPI cards, Plotly history chart, forecast chart, market comparison table, key insights, model card |
| **Data layer** | `MarketAPI` façade over a local CSV/SQLite dataset **or** a live REST API — agents never change |
| **Safety** | Every forecast and recommendation is framed as an estimate; nothing is ever presented as guaranteed |
| **Tests** | 136 pytest tests covering price retrieval, trend maths, leakage-free features, forecasting, recommendations and supervisor routing |

---

## Architecture

```
                       ┌──────────────────────────────┐
   user question ──►   │      SupervisorAgent         │
                       │  routes + merges the answer  │
                       └───────────┬──────────────────┘
                                   │
             ┌─────────────────────┼─────────────────────┬────────────────────┐
             ▼                     ▼                     ▼                    ▼
    ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │ QueryParser     │  │ MarketPriceAgent │  │   TrendAgent     │  │ PredictionAgent  │
    │ intent + crop + │  │ latest prices,   │  │ direction, %,    │  │ RandomForest     │
    │ market + days   │  │ market compare   │  │ volatility       │  │ recursive walk   │
    └─────────────────┘  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
                                  │                     │                     │
                                  └──────────┬──────────┴──────────┬──────────┘
                                             ▼                     ▼
                                  ┌────────────────────┐  ┌───────────────────┐
                                  │ RecommendationAgent│  │    MarketAPI      │
                                  │ sell / wait / watch│  │  data façade      │
                                  └────────────────────┘  └─────────┬─────────┘
                                                                     │
                                              ┌──────────────────────┴──────────────────┐
                                              ▼                                         ▼
                                   LocalDatasetSource (CSV/SQLite)          RestMarketDataSource
                                                                        (any real market-price API,
                                                                         auto-falls back to local)
```

**Agent responsibilities**

- **QueryParserAgent** — Gemini-first JSON extraction with a deterministic regex/keyword parser as
  both fallback and repair layer. Extracts intent, crop(s), market, date range and forecast horizon.
  Understands local names (*gehu, pyaz, tamatar, aloo, dhan, makka, kapas, ganna*).
- **MarketPriceAgent** — latest modal price, day's range, arrivals, multi-crop pricing, market
  comparison, highest-priced crops. Defaults to the best-covered mandi when none is named.
- **TrendAgent** — direction (increasing/decreasing/stable) from a blend of endpoint change **and**
  least-squares slope, so one noisy day cannot flip the verdict. Adds volatility, week/month change
  and range position insights.
- **PredictionAgent** — validates the horizon, forecasts, and reduces long horizons to readable
  checkpoint days (1, 3, 7, 15, 30, 60, 90).
- **RecommendationAgent** — deterministic, auditable scoring over forecast + trend + range position
  + volatility → *consider waiting / consider selling soon / monitor the market*, plus a
  better-paying-mandi hint when the premium exceeds 2%. Gemini only rewrites the wording.
- **SupervisorAgent** — routes by intent, calls several agents when needed, merges everything into
  one formatted reply plus structured `data` the dashboard charts.

**Design rules that matter**

- **No target leakage.** All rolling statistics are computed on `shift(1)` prices, and the
  train/test split is chronological on the date axis — never `train_test_split(shuffle=True)`.
- **Recursive forecasting.** Multi-day forecasts feed each prediction back in as the next day's lag,
  which is the only correct way to forecast several steps with lag features.
- **One feature order.** `utils/preprocessing.FEATURE_COLUMNS` is the single source of truth used by
  both training and inference, so the model can never be fed columns in a different order.
- **Uniform result envelopes.** Every tool/agent returns `{"ok": True, ...}` or
  `{"ok": False, "error": ..., "code": ...}`. The UI branches on that instead of catching exceptions.

---

## Folder structure

```
AI-Crop-Price-Agent/
├── app.py                     # Streamlit entrypoint: chatbot + dashboard + model card
├── requirements.txt
├── README.md
├── .env                       # your local config (git-ignored)
├── .env.example               # template to copy
├── .gitignore
│
├── agents/
│   ├── __init__.py
│   ├── query_parser.py        # natural language -> structured intent
│   ├── market_agent.py        # current / latest prices, market comparison
│   ├── trend_agent.py         # historical trend + insights
│   ├── prediction_agent.py    # ML forecasts, horizon validation
│   ├── recommendation_agent.py# sell / wait / watch guidance
│   └── supervisor.py          # routing + merged answers + dashboard payload
│
├── models/
│   ├── __init__.py
│   ├── train_model.py         # full training pipeline (CLI)
│   ├── predictor.py           # inference + statistical fallback
│   └── saved_models/          # crop_price_model.joblib, metrics.json
│
├── tools/
│   ├── __init__.py
│   ├── market_api.py          # MarketAPI façade + local/REST data sources
│   ├── data_loader.py         # sample-data generator, CSV/SQLite loading, validation
│   └── market_tools.py        # name resolution, series slicing, trend maths
│
├── utils/
│   ├── __init__.py
│   ├── gemini.py              # Gemini client, prompt loader, JSON extraction
│   ├── preprocessing.py       # cleaning, encoders, feature engineering, split
│   └── helpers.py             # errors, result envelopes, ₹ formatting
│
├── ui/
│   ├── __init__.py
│   └── components.py          # KPI cards, Plotly charts, tables, CSS
│
├── data/
│   └── crop_prices.csv        # sample dataset (~22,700 rows, 20 months)
│
├── prompts/
│   ├── query_parser_prompt.txt
│   ├── market_prompt.txt
│   ├── trend_prompt.txt
│   ├── prediction_prompt.txt
│   └── recommendation_prompt.txt
│
└── tests/
    ├── conftest.py            # fixtures; disables Gemini so fallbacks are tested
    ├── test_market.py
    ├── test_trend.py
    ├── test_prediction.py
    ├── test_recommendation.py
    └── test_supervisor.py
```

---

## Installation

Requires **Python 3.9+** (developed and tested on 3.12).

```bash
# 1. Get the project
cd AI-Crop-Price-Agent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows PowerShell / CMD

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env              # then edit .env and add your Gemini key (optional)

# 5. Generate the sample dataset (skip if data/crop_prices.csv already exists)
python tools/data_loader.py

# 6. Train the price model
python -m models.train_model

# 7. Run the app
streamlit run app.py
```

The app opens at <http://localhost:8501>.

---

## Environment variables

Copy `.env.example` to `.env`. Everything is optional — the app runs with an empty file.

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Google Generative AI key ([get one free](https://aistudio.google.com/apikey)). Enables AI-written explanations and LLM query parsing. | unset → template answers |
| `GEMINI_MODEL` | Gemini model name | `gemini-1.5-flash` |
| `GEMINI_TIMEOUT` | Per-request timeout in seconds | `20` |
| `CROP_DATA_CSV` | Path to your own price CSV | `data/crop_prices.csv` |
| `CROP_DATA_SQLITE` | Path to a SQLite DB (used automatically when it exists) | `data/crop_prices.db` |
| `MARKET_API_URL` | Set this to switch to a live REST price source | unset → local dataset |
| `MARKET_API_KEY` | API key for that endpoint | unset |
| `MARKET_API_RECORD_LIMIT` | Records to request per call | `5000` |
| `MARKET_API_FIELD_MAP` | JSON mapping of the API's field names onto this project's schema | Agmarknet defaults |

**The API key is never hardcoded** — `utils/gemini.py` reads it through `python-dotenv`, and `.env`
is git-ignored.

---

## Dataset format

`data/crop_prices.csv` — one row per crop, per mandi, per day:

| Column | Type | Description |
|---|---|---|
| `date` | `YYYY-MM-DD` | Trading date |
| `crop` | text | Wheat, Rice, Maize, Cotton, Soybean, Onion, Tomato, Sugarcane, Potato, Pulses |
| `market` | text | Mandi name (Aurangabad, Pune, Nashik, Nagpur, Indore, Bhopal, Ludhiana, Karnal, Jaipur, Bengaluru) |
| `state` | text | State the mandi is in |
| `arrival_quantity` | number | Arrivals in quintals |
| `price_min` | number | Day's low, ₹/quintal |
| `price_max` | number | Day's high, ₹/quintal |
| `modal_price` | number | Modal (most-traded) price, ₹/quintal — **the prediction target** |

Sample:

```csv
date,crop,market,state,arrival_quantity,price_min,price_max,modal_price
2026-09-04,Wheat,Aurangabad,Maharashtra,168.3,2373.0,2623.0,2498.0
2026-09-04,Onion,Nashik,Maharashtra,214.7,2122.0,2346.0,2234.0
```

The bundled sample set holds ~22,700 rows: 10 crops × 41 crop–mandi series × ~20 months of daily
records, with realistic seasonal harvest dips, yearly drift, weekly arrival cycles, supply shocks
for perishables and occasional closed-mandi gaps. It is generated deterministically
(`seed=42`) by `tools/data_loader.py`, so it is reproducible.

To use your own data: overwrite `data/crop_prices.csv` with the same columns (or point
`CROP_DATA_CSV` at your file) and retrain. Export to SQLite any time with:

```python
from tools.data_loader import export_to_sqlite
export_to_sqlite()   # -> data/crop_prices.db, picked up automatically
```

---

## Model training

```bash
python -m models.train_model                       # defaults
python -m models.train_model --n-estimators 500 --test-size 0.15
python -m models.train_model --source sqlite       # train from SQLite instead of CSV
```

The pipeline runs eight explicit stages: load → clean → engineer features → chronological split →
train → evaluate → rank importance → save.

**19 features**

- *Time*: month, day, day-of-week, day-of-year, week-of-year, quarter
- *Categorical* (unknown-tolerant integer codes): crop, market, state, season (Kharif / Rabi / Zaid)
- *Lags*: 1, 3, 7, 14, 30 days
- *Rolling* (all on shifted prices): 7 / 14 / 30-day means, 7-day standard deviation

**Reference run on the bundled dataset** (Random Forest, 300 trees, 17,126 train / 4,362 held-out
future rows):

| Split | MAE (₹/qtl) | RMSE (₹/qtl) | R² | MAPE |
|---|---|---|---|---|
| Train | 22.26 | 36.30 | 0.9997 | 0.84% |
| **Test (future window)** | **48.95** | **72.26** | **0.9989** | **1.64%** |

Top features: `lag_1` (0.204), `roll_mean_7` (0.184), `lag_3` (0.159), `roll_mean_14` (0.121),
`lag_7` (0.103).

Artefacts land in `models/saved_models/`: `crop_price_model.joblib` (model + encoders + metrics +
feature order) and `metrics.json`. The Streamlit **Model** tab renders all of it.

> R² is high because next-day mandi prices are strongly autocorrelated — a genuine property of this
> data, not leakage: the held-out window is strictly *after* the training cut-off and rolling stats
> never touch the current day. Accuracy still degrades with horizon, which is why forecasts carry
> √t-widening uncertainty bands.

---

## Running the app

```bash
streamlit run app.py
```

**Sidebar** — crop, market and prediction-period (7/15/30/90) selection, plus live status for the
data source, the trained model (type, R², MAE) and Gemini.

**Chatbot tab** — free-text Q&A with example-question buttons; charts render inline with the answer,
and the agents used for each reply are shown.

**Dashboard tab** — six KPI cards (current price, predicted price, expected change %, 30-day trend,
highest market price, lowest market price), a history chart with the daily min–max band and 7-day
average, a forecast chart with uncertainty band, key insights, the recommendation with its reasons,
a market comparison table + chart identifying the best-paying mandi, and estimates by horizon.

**Model tab** — metrics, train-vs-test comparison, hyper-parameters, feature importance, and the
retraining command.

---

## Testing

```bash
python -m pytest tests -q              # all 136 tests
python -m pytest tests/test_prediction.py -v
python -m pytest tests -k "leakage or fallback"
```

`tests/conftest.py` clears `GEMINI_API_KEY` for the whole session, so the suite makes **no network
calls** and specifically exercises the offline fallback paths.

Coverage highlights:

- **test_market.py** — schema, 10-crop coverage, history span, cleaning of broken rows (inverted
  min/max, missing modal price, junk dates, blank crops), deterministic generation, alias/typo
  resolution (`gehu`, `wheet`, `pyaz`, `bangalore`), invalid crop/market, empty ranges, comparison
  ordering.
- **test_trend.py** — rising / falling / flat detection on synthetic series, % change, volatility,
  short-series tolerance, empty-series error, window handling.
- **test_prediction.py** — horizon validation (0, −5, 400, `"soon"`, `None`), leakage assertion
  (`lag_1` equals the previous day, never today), chronological split ordering, all four horizons,
  future-dated sequential forecasts, plausibility bounds, checkpoint compression, missing-model
  fallback, model quality gate (R² > 0.5).
- **test_recommendation.py** — rising forecast → wait, falling → sell, flat → watch, fallback
  down-weighting, volatility penalty, forbidden-phrase check (`guaranteed`, `you will get`,
  `risk-free`, …), better-mandi hint.
- **test_supervisor.py** — all nine spec example questions classified correctly, crop/market/period
  extraction, capped oversized horizons, empty query, every intent's routing, multi-agent
  fan-out, gibberish and non-English input, dashboard payload completeness, health reporting.

---

## Replacing the sample data with a real API

The agents talk only to `MarketAPI`, so swapping the source touches **no agent code**.

**Option A — environment variables only (no code change).**

```bash
# .env
MARKET_API_URL=https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070
MARKET_API_KEY=your_data_gov_in_key
MARKET_API_FIELD_MAP={"arrival_date":"date","commodity":"crop","market":"market","state":"state","min_price":"price_min","max_price":"price_max","modal_price":"modal_price"}
```

`build_default_source()` sees `MARKET_API_URL` and switches to `RestMarketDataSource`, which fetches,
renames fields onto this project's schema, cleans and validates. **Any failure falls back to the
local dataset automatically**, and the sidebar shows that a fallback is in use.

**Option B — your own source class** (for GraphQL, a database, a paid feed, pagination):

```python
from tools.market_api import BaseMarketDataSource, MarketAPI

class MyFeedSource(BaseMarketDataSource):
    name = "my_feed"

    def frame(self, refresh: bool = False):
        raw = fetch_from_my_feed()                      # -> DataFrame
        from utils.preprocessing import clean_dataframe
        return clean_dataframe(raw.rename(columns={...}))

    def health(self):
        return {"source": self.name, "ok": True}

api = MarketAPI(source=MyFeedSource())     # pass to SupervisorAgent(api=api)
```

Required columns after renaming: `date, crop, market, state, arrival_quantity, price_min,
price_max, modal_price`. Retrain (`python -m models.train_model`) once real history is flowing.

---

## Example chatbot questions

**Current price**
- What is the current price of wheat?
- Tomato rate in Bengaluru today
- gehu ka bhav kya hai

**Prediction**
- Predict wheat price for the next 7 days.
- What will be the price of onion next month?
- What is the expected price of maize next week?
- Predict cotton price for the next 90 days in Nagpur.

**Trend**
- Show me the price trend of rice.
- How have soybean prices moved in the last 3 months?
- Is potato going up or down?

**Recommendation**
- Should I sell my wheat now or wait?
- Should I sell my onion now or wait?

**Comparison**
- Compare wheat and rice prices.
- Which market has the best price for tomatoes?
- Where should I sell my soybean?

**Discovery**
- Which crop has the highest price currently?
- Which crops do you have data for?
- What can you do?

Sample reply:

```
🌾 Wheat Price Prediction — Aurangabad

Current Price: ₹2,498/quintal
Current Trend: 📉 Decreasing

Predicted Price (next 1 week):
  Day 1 (2026-09-06): ₹2,489
  Day 3 (2026-09-08): ₹2,472
  Day 7 (2026-09-12): ₹2,454

Insight:
  The model predicts a slight downward move of -1.77% over the next 1 week.

Recommendation:
  Current price: ₹2,498/quintal. Prices for Wheat in Aurangabad are decreasing and the forecast
  points lower, so selling sooner rather than holding may be worth considering — especially if
  storage is costly for you. Ludhiana is currently paying ₹2,782/quintal, about 11.4% more than
  Aurangabad — worth checking transport cost against that gap.

⚠️ Prediction is based on historical data and is not guaranteed.
```

---

## Error handling

Every failure listed in the spec is handled explicitly and never crashes the app:

| Failure | Behaviour |
|---|---|
| Invalid crop | Friendly message listing every available crop; aliases and typos resolved first |
| Invalid market | Message listing the markets that actually trade that crop |
| Missing historical data | Clear "no records in that range" message; short series still analysed where possible |
| Market API failure | Automatic fallback to the local dataset; sidebar shows the fallback is active |
| Gemini failure / missing key | Deterministic template answers; rule-based query parsing; status shown in the sidebar |
| Model failure / not trained | Damped linear-trend fallback, labelled `trend_fallback` with lower confidence |
| Empty user query | Prompt for a question with a concrete example |
| Invalid prediction period | Rejected below 1 day and above 90; non-canonical values accepted; oversized values capped with a note |
| Missing API key | App runs fully; sidebar explains what enabling the key adds |
| Unexpected exception | Supervisor's guard returns a helpful rephrase suggestion instead of a stack trace |

---

## Extending the project

- **More crops/markets** — just add rows with the same schema and retrain; encoders map unseen
  categories to `-1` rather than crashing, so the app keeps working even before a retrain.
- **A different model** — swap `RandomForestRegressor` in `models/train_model.py` for
  `GradientBoostingRegressor`, `HistGradientBoostingRegressor` or XGBoost. `predictor.py` needs no
  change as long as the bundle keeps its keys (`model`, `encoders`, `feature_columns`).
- **Weather / MSP / mandi arrivals as features** — add the column in
  `utils/preprocessing.build_feature_frame` and append its name to `FEATURE_COLUMNS`; for recursive
  forecasting also extend `make_feature_row`.
- **Serve as a REST API** — `MarketAPI` and `SupervisorAgent` are plain Python with JSON-ready
  dicts, so a FastAPI wrapper is a thin layer: `POST /ask` → `SupervisorAgent().handle(question)`.
- **Scheduled alerts** — call `SupervisorAgent().dashboard_payload(crop, market)` from a cron job and
  notify when `expected_change_pct` crosses your threshold.

---

## Disclaimer

Price forecasts are statistical estimates derived from historical mandi data. They are **not
guaranteed prices** and must not be treated as financial advice. Always weigh your own storage
cost, crop quality, transport cost and cash requirements before making a selling decision.
