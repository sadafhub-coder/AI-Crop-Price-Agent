"""
ui/components.py
----------------
Presentation helpers for the Streamlit app: KPI cards, Plotly charts and
tables. Keeping them here makes app.py readable and lets the charts be reused
(or unit-tested) without importing Streamlit page logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.helpers import (
    TREND_DOWN,
    TREND_UP,
    format_currency,
    format_pct,
    trend_emoji,
)

PRIMARY = "#2E7D32"
ACCENT = "#F9A825"
DANGER = "#C62828"
MUTED = "#90A4AE"

CUSTOM_CSS = """
<style>
  .main .block-container { padding-top: 1.6rem; max-width: 1400px; }
  .kpi-card {
      background: linear-gradient(160deg, #1e293b 0%, #0f172a 100%);
      border: 1px solid #334155; border-left: 5px solid #2E7D32;
      border-radius: 12px; padding: 14px 16px; height: 100%;
      box-shadow: 0 1px 3px rgba(0,0,0,.25);
  }
  .kpi-card .label { font-size: .78rem; text-transform: uppercase;
      letter-spacing: .04em; color: #94a3b8; margin-bottom: 4px; }
  .kpi-card .value { font-size: 1.45rem; font-weight: 700; color: #f8fafc; line-height: 1.2; }
  .kpi-card .delta { font-size: .82rem; margin-top: 2px; }
  .up { color: #4ade80; } .down { color: #f87171; } .flat { color: #94a3b8; }
  
  /* Insight Boxes with clear dark mode contrast */
  .insight-box {
      background-color: #1f2937 !important;
      color: #f3f4f6 !important;
      border-left: 4px solid #10b981 !important;
      border-radius: 8px;
      padding: 12px 16px;
      margin-bottom: 10px;
      font-size: 0.92rem;
      line-height: 1.45;
  }
  .insight-box b, .insight-box strong {
      color: #ffffff !important;
  }
  
  .warn-box {
      background-color: #372b1f !important;
      color: #fef08a !important;
      border-left: 4px solid #f59e0b !important;
      border-radius: 8px;
      padding: 12px 16px;
      margin-bottom: 10px;
      font-size: 0.92rem;
      line-height: 1.45;
  }
  .warn-box b, .warn-box strong {
      color: #fffbeb !important;
  }

  .chat-answer { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: pre-wrap; font-size: .92rem; line-height: 1.45; }
  div[data-testid="stMetricValue"] { font-size: 1.3rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def insight_box(text: str, kind: str = "info") -> None:
    """Renders a styled notification card with dark background and contrasting text."""
    box_class = "warn-box" if kind == "warn" else "insight-box"
    st.markdown(f'<div class="{box_class}">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# KPI cards
# --------------------------------------------------------------------------


def kpi_card(label: str, value: str, delta: Optional[str] = None, tone: str = "flat") -> str:
    delta_html = f'<div class="delta {tone}">{delta}</div>' if delta else ""
    return (
        f'<div class="kpi-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{delta_html}</div>'
    )


def render_kpis(payload: Dict[str, Any]) -> None:
    """
    Six KPI cards: current price, predicted price, change %, trend,
    highest market price, lowest market price.
    """
    price = payload.get("price") or {}
    trend = payload.get("trend") or {}
    prediction = payload.get("prediction") or {}
    comparison = payload.get("comparison") or {}

    current = price.get("modal_price")
    predicted = prediction.get("predicted_final_price")
    change_pct = prediction.get("expected_change_pct")
    direction = trend.get("direction")

    tone = "up" if (change_pct or 0) > 0 else "down" if (change_pct or 0) < 0 else "flat"
    trend_tone = (
        "up" if direction == TREND_UP else "down" if direction == TREND_DOWN else "flat"
    )

    cards = [
        kpi_card(
            "Current Price",
            format_currency(current) if current else "N/A",
            f"as of {price.get('date', 'N/A')}",
        ),
        kpi_card(
            f"Predicted ({prediction.get('horizon_days', '–')}d)",
            format_currency(predicted) if predicted else "N/A",
            prediction.get("method", "").replace("_", " ") or None,
            tone,
        ),
        kpi_card(
            "Expected Change",
            format_pct(change_pct) if change_pct is not None else "N/A",
            f"{format_currency(prediction.get('expected_change'), None)} /qtl"
            if prediction.get("expected_change") is not None
            else None,
            tone,
        ),
        kpi_card(
            "30-Day Trend",
            f"{trend_emoji(direction)} {str(direction or 'N/A').title()}",
            format_pct(trend.get("change_pct")) if trend.get("change_pct") is not None else None,
            trend_tone,
        ),
        kpi_card(
            "Highest Market Price",
            format_currency(comparison.get("best_price")) if comparison.get("best_price") else "N/A",
            comparison.get("best_market"),
            "up",
        ),
        kpi_card(
            "Lowest Market Price",
            format_currency(comparison.get("lowest_price"))
            if comparison.get("lowest_price")
            else "N/A",
            comparison.get("lowest_market"),
            "down",
        ),
    ]
    for column, card in zip(st.columns(6), cards):
        column.markdown(card, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def history_chart(history: Dict[str, Any], title: Optional[str] = None) -> go.Figure:
    """Historical modal price with the daily min–max band and a 7-day average."""
    records = history.get("history", [])
    frame = pd.DataFrame(records)
    figure = go.Figure()
    if frame.empty:
        return _empty_figure("No historical data available")

    frame["date"] = pd.to_datetime(frame["date"])
    figure.add_trace(
        go.Scatter(
            x=frame["date"], y=frame["price_max"], name="Day high",
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["date"], y=frame["price_min"], name="Daily range",
            fill="tonexty", fillcolor="rgba(46,125,50,.12)",
            line=dict(width=0), hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["date"], y=frame["modal_price"], name="Modal price",
            line=dict(color=PRIMARY, width=2),
            hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.0f}/qtl<extra></extra>",
        )
    )
    if len(frame) >= 7:
        figure.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["modal_price"].rolling(7, min_periods=3).mean(),
                name="7-day average",
                line=dict(color=ACCENT, width=1.6, dash="dash"),
            )
        )
    return _style(
        figure,
        title or f"{history.get('crop','')} price history — {history.get('market','')}",
    )


def prediction_chart(
    history: Dict[str, Any], prediction: Dict[str, Any], lookback_days: int = 60
) -> go.Figure:
    """Recent actual prices + the forecast path with its uncertainty band."""
    figure = go.Figure()
    records = pd.DataFrame(history.get("history", []))
    forecast = pd.DataFrame(prediction.get("predictions", []))
    if forecast.empty:
        return _empty_figure("No forecast available")

    forecast["date"] = pd.to_datetime(forecast["date"])

    if not records.empty:
        records["date"] = pd.to_datetime(records["date"])
        recent = records.tail(lookback_days)
        figure.add_trace(
            go.Scatter(
                x=recent["date"], y=recent["modal_price"], name="Actual",
                line=dict(color=PRIMARY, width=2),
                hovertemplate="%{x|%d %b}<br>₹%{y:,.0f}<extra></extra>",
            )
        )
        bridge_x = [recent["date"].iloc[-1], forecast["date"].iloc[0]]
        bridge_y = [recent["modal_price"].iloc[-1], forecast["predicted_price"].iloc[0]]
        figure.add_trace(
            go.Scatter(x=bridge_x, y=bridge_y, line=dict(color=ACCENT, width=1.5, dash="dot"),
                       showlegend=False, hoverinfo="skip")
        )

    figure.add_trace(
        go.Scatter(x=forecast["date"], y=forecast["upper_bound"], line=dict(width=0),
                   name="Upper bound", hoverinfo="skip", showlegend=False)
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["date"], y=forecast["lower_bound"], name="Likely range",
            fill="tonexty", fillcolor="rgba(249,168,37,.18)", line=dict(width=0),
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["date"], y=forecast["predicted_price"], name="Predicted",
            line=dict(color=ACCENT, width=2.4),
            marker=dict(size=5), mode="lines+markers",
            hovertemplate="%{x|%d %b}<br>₹%{y:,.0f} (estimate)<extra></extra>",
        )
    )
    return _style(
        figure,
        f"{prediction.get('crop','')} — next {prediction.get('horizon_days','')} days "
        f"(estimate, not guaranteed)",
    )


def market_comparison_chart(comparison: Dict[str, Any]) -> go.Figure:
    """Horizontal bars of the latest price per mandi, best one highlighted."""
    rows = comparison.get("markets", [])
    if not rows:
        return _empty_figure("No market comparison available")
    frame = pd.DataFrame(rows).sort_values("modal_price")
    best = comparison.get("best_market")
    colors = [PRIMARY if market == best else MUTED for market in frame["market"]]
    figure = go.Figure(
        go.Bar(
            x=frame["modal_price"], y=frame["market"], orientation="h",
            marker_color=colors,
            text=[format_currency(v, None) for v in frame["modal_price"]],
            textposition="outside",
            hovertemplate="%{y}<br>₹%{x:,.0f}/qtl<extra></extra>",
        )
    )
    figure.update_xaxes(title="₹ per quintal")
    return _style(figure, f"{comparison.get('crop','')} — price by mandi", height=340)


def multi_horizon_chart(horizons: List[Dict[str, Any]]) -> go.Figure:
    """Predicted price at each supported horizon (7 / 15 / 30 / 90 days)."""
    if not horizons:
        return _empty_figure("No multi-horizon forecast available")
    frame = pd.DataFrame(horizons)
    colors = [PRIMARY if v >= 0 else DANGER for v in frame["expected_change_pct"]]
    figure = go.Figure(
        go.Bar(
            x=[f"{d} days" for d in frame["horizon_days"]],
            y=frame["predicted_price"],
            marker_color=colors,
            text=[format_currency(v, None) for v in frame["predicted_price"]],
            textposition="outside",
            hovertemplate="%{x}<br>₹%{y:,.0f}<extra></extra>",
        )
    )
    figure.update_yaxes(title="₹ per quintal")
    return _style(figure, "Estimated price by horizon", height=320)


def _style(figure: go.Figure, title: str, height: int = 420) -> go.Figure:
    figure.update_layout(
        title=dict(text=title, font=dict(size=15)),
        height=height,
        margin=dict(l=10, r=20, t=50, b=10),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        font=dict(family="system-ui, sans-serif", size=12),
    )
    figure.update_xaxes(showgrid=True, gridcolor="#ECEFF1")
    figure.update_yaxes(showgrid=True, gridcolor="#ECEFF1")
    return figure


def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message, showarrow=False, font=dict(size=14, color=MUTED),
        xref="paper", yref="paper", x=0.5, y=0.5,
    )
    figure.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white")
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


# --------------------------------------------------------------------------
# Tables and text blocks
# --------------------------------------------------------------------------


def comparison_table(comparison: Dict[str, Any]) -> pd.DataFrame:
    """Market comparison as a display-ready DataFrame."""
    rows = comparison.get("markets", [])
    if not rows:
        return pd.DataFrame(columns=["Market", "Current Price"])
    frame = pd.DataFrame(rows)
    display = pd.DataFrame(
        {
            "Market": frame["market"],
            "State": frame["state"],
            "Current Price": [format_currency(v, None) for v in frame["modal_price"]],
            "30-Day Change": [format_pct(v) for v in frame["change_30d_pct"]],
            "Trend": [f"{trend_emoji(t)} {str(t or '').title()}" for t in frame["trend"]],
            "Last Updated": frame["date"],
        }
    )
    return display


def prediction_table(prediction: Dict[str, Any]) -> pd.DataFrame:
    rows = prediction.get("predictions", [])
    if not rows:
        return pd.DataFrame(columns=["Day", "Date", "Predicted Price"])
    frame = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "Day": frame["day"],
            "Date": frame["date"],
            "Predicted Price": [format_currency(v, None) for v in frame["predicted_price"]],
            "Likely Range": [
                f"{format_currency(low, None)} – {format_currency(high, None)}"
                for low, high in zip(frame["lower_bound"], frame["upper_bound"])
            ],
        }
    )