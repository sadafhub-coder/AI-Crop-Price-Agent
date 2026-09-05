"""
app.py
------
Streamlit entrypoint: the chatbot + the market dashboard.

Run with:
    streamlit run app.py

Layout
  Sidebar : crop / market / horizon selection, data & model status, model card
  Main    : Chatbot tab   -> ask anything, answered by the Supervisor Agent
            Dashboard tab -> KPIs, history chart, forecast chart, market
                             comparison table and key insights
            Model tab     -> metrics, feature importance, how to retrain
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make imports work no matter where streamlit is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import os  # noqa: E402
import streamlit as st  # noqa: E402
# Transfer Streamlit Cloud secrets into os.environ for Gemini / tools
for key in ["GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_TIMEOUT"]:
    if key in st.secrets:
        os.environ[key] = str(st.secrets[key])

st.set_page_config(
    page_title="AI Crop Price & Market Insight Chatbot",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

from agents.supervisor import SupervisorAgent  # noqa: E402
from tools.market_api import MarketAPI  # noqa: E402
from ui import components as ui  # noqa: E402
from utils.helpers import format_currency, format_pct, trend_emoji  # noqa: E402

EXAMPLE_QUESTIONS = [
    "What is the current price of wheat?",
    "Predict wheat price for the next 7 days in Aurangabad.",
    "Show me the price trend of rice.",
    "Should I sell my onion now or wait?",
    "Which market has the best price for tomatoes?",
    "Compare wheat and rice prices.",
    "Which crop has the highest price currently?",
    "What is the expected price of maize next week?",
]


# --------------------------------------------------------------------------
# Cached resources — built once per session, not on every rerun
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading market data and model …")
def get_supervisor() -> SupervisorAgent:
    return SupervisorAgent(api=MarketAPI())


@st.cache_data(ttl=600, show_spinner=False)
def get_options() -> Dict[str, Any]:
    supervisor = get_supervisor()
    crops = supervisor.api.crops()
    return {
        "crops": crops,
        "markets": {crop: supervisor.api.markets(crop) for crop in crops},
    }


@st.cache_data(ttl=300, show_spinner="Crunching prices …")
def get_dashboard(crop: str, market: str, days: int) -> Dict[str, Any]:
    return get_supervisor().dashboard_payload(crop, market, days=days)


@st.cache_data(ttl=300, show_spinner=False)
def get_multi_horizon(crop: str, market: str) -> Dict[str, Any]:
    return get_supervisor().prediction_agent.predict_multi_horizon(crop, market)


def ask(question: str) -> Dict[str, Any]:
    """One chatbot turn (not cached — answers reflect the latest data)."""
    return get_supervisor().handle(question)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------


def render_sidebar(options: Dict[str, Any], health: Dict[str, Any]) -> Dict[str, Any]:
    st.sidebar.title("🌾 Crop Price AI")
    st.sidebar.caption("Prices, trends and forecasts for Indian mandis")

    crops: List[str] = options["crops"]
    if not crops:
        st.sidebar.error("No crop data found. Run: python tools/data_loader.py")
        return {"crop": None, "market": None, "days": 7}

    default_crop = crops.index("Wheat") if "Wheat" in crops else 0
    crop = st.sidebar.selectbox("Crop", crops, index=default_crop)

    markets: List[str] = options["markets"].get(crop, [])
    market = st.sidebar.selectbox("Market (mandi)", markets, index=0 if markets else None)

    days = st.sidebar.select_slider(
        "Prediction period (days)", options=[7, 15, 30, 90], value=7
    )

    st.sidebar.divider()
    st.sidebar.subheader("System status")

    data_health = health.get("data", {})
    if data_health.get("ok"):
        st.sidebar.success(
            f"Data: {data_health.get('source', 'local')} · "
            f"{data_health.get('rows', 0):,} rows · latest {data_health.get('latest_date', '—')}"
        )
    else:
        st.sidebar.warning(f"Data: fallback in use — {data_health.get('error', 'unknown')}")

    model = health.get("model", {})
    if model.get("available"):
        test = (model.get("metrics") or {}).get("test", {})
        st.sidebar.success(
            f"Model: {model.get('model_type')} · R² {test.get('r2', '—')} · "
            f"MAE ₹{test.get('mae', '—')}"
        )
    else:
        st.sidebar.warning(
            "Model: not trained yet — statistical fallback in use.\n\n"
            "Train it with `python -m models.train_model`"
        )

    gemini = health.get("gemini", {})
    if gemini.get("available"):
        st.sidebar.success(f"Gemini: connected ({gemini.get('model')})")
    else:
        st.sidebar.info(
            "Gemini: not connected — answers use built-in templates.\n\n"
            "Add `GEMINI_API_KEY` to your `.env` for AI-written explanations."
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "⚠️ Forecasts are statistical estimates from historical mandi prices, "
        "not guaranteed prices."
    )
    return {"crop": crop, "market": market, "days": int(days)}


# --------------------------------------------------------------------------
# Chatbot tab
# --------------------------------------------------------------------------


def render_chat(selection: Dict[str, Any]) -> None:
    st.subheader("💬 Ask about crop prices")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Namaste 🙏 Ask me about mandi prices, trends, forecasts or whether to "
                    "sell now.\n\nFor example: *Predict wheat price for the next 7 days in "
                    "Aurangabad.*"
                ),
            }
        ]

    with st.expander("Example questions", expanded=False):
        columns = st.columns(2)
        for index, question in enumerate(EXAMPLE_QUESTIONS):
            if columns[index % 2].button(question, key=f"example_{index}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(
                    f'<div class="chat-answer">{message["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(message["content"])

    question = st.chat_input("e.g. Should I sell my onion now or wait?")
    if not question and st.session_state.get("pending_question"):
        question = st.session_state.pop("pending_question")

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking mandi data …"):
            result = ask(question)
        st.markdown(f'<div class="chat-answer">{result["reply"]}</div>', unsafe_allow_html=True)

        data = result.get("data") or {}
        if data.get("prediction") and data.get("history") is None:
            history = get_supervisor().api.get_history(
                data["prediction"]["crop"], data["prediction"]["market"], days=180
            )
            if history["ok"]:
                st.plotly_chart(
                    ui.prediction_chart(history, data["prediction"]),
                    use_container_width=True,
                )
        elif data.get("trend"):
            history = get_supervisor().api.get_history(
                data["trend"]["crop"], data["trend"]["market"], days=365
            )
            if history["ok"]:
                st.plotly_chart(ui.history_chart(history), use_container_width=True)
        elif data.get("comparison"):
            st.plotly_chart(
                ui.market_comparison_chart(data["comparison"]), use_container_width=True
            )

        st.caption(
            "Agents used: " + ", ".join(result.get("agents_used", [])) +
            f" · intent: {result.get('intent')}"
        )

    st.session_state.messages.append({"role": "assistant", "content": result["reply"]})


# --------------------------------------------------------------------------
# Dashboard tab
# --------------------------------------------------------------------------


def render_dashboard(selection: Dict[str, Any]) -> None:
    crop, market, days = selection["crop"], selection["market"], selection["days"]
    if not crop or not market:
        st.info("Select a crop and market in the sidebar to see the dashboard.")
        return

    payload = get_dashboard(crop, market, days)
    if not payload.get("ok"):
        st.error(payload.get("error", "Could not build the dashboard."))
        return

    st.subheader(f"📊 {crop} — {market}")
    ui.render_kpis(payload)
    for problem in payload.get("errors", []):
        st.warning(problem)

    st.write("")
    left, right = st.columns([3, 2])

    with left:
        if payload.get("history"):
            st.plotly_chart(ui.history_chart(payload["history"]), use_container_width=True)
        if payload.get("prediction") and payload.get("history"):
            st.plotly_chart(
                ui.prediction_chart(payload["history"], payload["prediction"]),
                use_container_width=True,
            )
            with st.expander("Day-by-day forecast table"):
                st.dataframe(
                    ui.prediction_table(payload["prediction"]),
                    use_container_width=True,
                    hide_index=True,
                )

    with right:
        st.markdown("##### 🔎 Key insights")
        trend = payload.get("trend")
        if trend:
            ui.insight_box(
                f"<b>Trend:</b> {trend_emoji(trend['direction'])} "
                f"{trend['direction'].title()} — {format_pct(trend['change_pct'])} over "
                f"{trend['window_days']} days."
            )
            for insight in trend.get("insights", [])[:4]:
                ui.insight_box(insight)

        prediction = payload.get("prediction")
        if prediction:
            ui.insight_box(
                f"<b>Forecast ({prediction['horizon_days']} days):</b> "
                f"{format_currency(prediction['predicted_final_price'])} "
                f"({format_pct(prediction['expected_change_pct'])}) · "
                f"model: {prediction['method'].replace('_', ' ')}"
            )
            ui.insight_box(prediction["disclaimer"], kind="warn")

        recommendation = payload.get("recommendation")
        if recommendation:
            st.markdown("##### 🧑‍🌾 Recommendation")
            ui.insight_box(
                f"<b>{recommendation['action_label']}</b><br>{recommendation['recommendation']}"
            )
            with st.expander("Why this suggestion?"):
                for reason in recommendation["reasons"]:
                    st.markdown(f"- {reason}")
            st.caption(recommendation["disclaimer"])

    st.divider()
    st.markdown("##### 🏪 Market comparison")
    comparison = payload.get("comparison")
    if comparison:
        table_column, chart_column = st.columns([2, 3])
        with table_column:
            st.dataframe(
                ui.comparison_table(comparison), use_container_width=True, hide_index=True
            )
            st.success(
                f"Best price: **{comparison['best_market']}** at "
                f"{format_currency(comparison['best_price'])} "
                f"(₹{comparison['price_gap']:,.0f}/qtl above the lowest mandi)"
            )
        with chart_column:
            st.plotly_chart(ui.market_comparison_chart(comparison), use_container_width=True)
    else:
        st.info("Market comparison is unavailable for this crop.")

    st.divider()
    st.markdown("##### ⏱️ Estimates by horizon")
    multi = get_multi_horizon(crop, market)
    if multi.get("ok"):
        st.plotly_chart(ui.multi_horizon_chart(multi["horizons"]), use_container_width=True)
        st.dataframe(
            pd.DataFrame(multi["horizons"]).rename(
                columns={
                    "horizon_days": "Horizon (days)",
                    "predicted_price": "Predicted ₹/qtl",
                    "expected_change_pct": "Change %",
                    "direction": "Direction",
                    "method": "Method",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(multi.get("error", "Multi-horizon estimates unavailable."))


# --------------------------------------------------------------------------
# Model tab
# --------------------------------------------------------------------------


def render_model_tab() -> None:
    st.subheader("🧠 Model information")
    info = get_supervisor().api.predictor.info()

    if not info.get("available"):
        st.warning(info.get("reason") or "No trained model found.")
        st.code("python -m models.train_model", language="bash")
        st.caption(
            "Until a model is trained, forecasts use a damped linear-trend fallback so the "
            "app keeps working."
        )
        return

    metrics = info.get("metrics", {})
    test, train = metrics.get("test", {}), metrics.get("train", {})
    columns = st.columns(4)
    columns[0].metric("Model", info.get("model_type", "—"))
    columns[1].metric("Test MAE (₹/qtl)", test.get("mae", "—"))
    columns[2].metric("Test RMSE (₹/qtl)", test.get("rmse", "—"))
    columns[3].metric("Test R²", test.get("r2", "—"))

    st.caption(
        f"Trained {info.get('trained_at', '—')} · training data up to "
        f"{info.get('train_cutoff', '—')} · {info.get('n_features', '—')} features · "
        f"held-out rows: {test.get('n', '—')}"
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Training vs test**")
        st.dataframe(
            pd.DataFrame([{"split": "train", **train}, {"split": "test", **test}]),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Parameters**")
        st.json(info.get("params", {}))
    with right:
        st.markdown("**Top feature importance**")
        importance = info.get("feature_importance", {})
        if importance:
            frame = (
                pd.DataFrame({"feature": list(importance.keys()), "importance": list(importance.values())})
                .sort_values("importance", ascending=False)
                .head(12)
            )
            st.bar_chart(frame.set_index("feature"))

    st.markdown("**Data used**")
    st.json(info.get("data", {}))
    st.info(
        "Retrain any time after replacing the dataset: `python -m models.train_model` — "
        "the app picks the new model up on its next restart."
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    ui.inject_css()
    try:
        supervisor = get_supervisor()
        options = get_options()
        health = supervisor.health()
    except Exception as exc:  # dataset unreadable, disk issue, ...
        st.title("🌾 AI Crop Price & Market Insight Chatbot")
        st.error(f"Startup failed: {exc}")
        st.code("python tools/data_loader.py   # regenerate the sample dataset", language="bash")
        return

    st.title("🌾 AI Crop Price Prediction & Market Insight Chatbot")
    st.caption(
        "Ask in plain language. Answers combine live mandi records, trend analysis and a "
        "Random Forest price model."
    )

    selection = render_sidebar(options, health)
    chat_tab, dashboard_tab, model_tab = st.tabs(["💬 Chatbot", "📊 Dashboard", "🧠 Model"])
    with chat_tab:
        render_chat(selection)
    with dashboard_tab:
        render_dashboard(selection)
    with model_tab:
        render_model_tab()


if __name__ == "__main__":
    main()
