"""
Streamlit dashboard for NQ Futures Day Trading Assistant.
Reads state from ui_state.json written periodically by main.py.
"""

import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_STATE_FILE = Path(__file__).parent.parent / "ui_state.json"
_REFRESH_S = 10

st.set_page_config(
    page_title="NQ Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _load_state() -> dict:
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _direction_color(direction: str) -> str:
    if direction == "LONG":
        return "#00ff88"
    if direction == "SHORT":
        return "#ff4444"
    return "#888888"


def main():
    st.title("NQ Futures Day Trading Assistant")

    state = _load_state()

    if not state:
        st.info("Warte auf Daten... (ui_state.json nicht gefunden)")
        time.sleep(2)
        st.rerun()
        return

    # ----------------------------------------------------------------
    # BEREICH 1 — NQ Preischart mit Timeframe-Auswahl
    # ----------------------------------------------------------------
    tf = st.radio("Timeframe", ["1m", "5m", "15m"], horizontal=True, index=1)

    bars = state.get("bars", {}).get(tf, [])

    if bars:
        df = pd.DataFrame(bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="NQ",
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff4444",
        ))

        if "vwap" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["vwap"],
                name="VWAP", line=dict(color="#ffaa00", width=1.5, dash="dot"),
            ))

        if len(df) >= 21:
            df["ema9"]  = df["close"].ewm(span=9).mean()
            df["ema21"] = df["close"].ewm(span=21).mean()
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["ema9"],
                name="EMA9", line=dict(color="#00aaff", width=1),
            ))
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["ema21"],
                name="EMA21", line=dict(color="#ff6600", width=1),
            ))

        signals = state.get("signals", {})
        trade_setup = signals.get("trade_setup")
        if trade_setup and signals.get("direction") != "NEUTRAL":
            direction = signals.get("direction")
            color = "#00ff88" if direction == "LONG" else "#ff4444"

            fig.add_hline(y=trade_setup["entry_price"],
                         line_color=color, line_dash="solid", line_width=2,
                         annotation_text=f"Entry {trade_setup['entry_price']:.2f}")
            fig.add_hline(y=trade_setup["stop_loss_price"],
                         line_color="#ff0000", line_dash="dash", line_width=1.5,
                         annotation_text=f"SL {trade_setup['stop_loss_price']:.2f}")
            fig.add_hline(y=trade_setup["take_profit_1_price"],
                         line_color="#00ff88", line_dash="dash", line_width=1.5,
                         annotation_text=f"TP1 {trade_setup['take_profit_1_price']:.2f}")
            fig.add_hline(y=trade_setup["take_profit_2_price"],
                         line_color="#00ff88", line_dash="dot", line_width=1,
                         annotation_text=f"TP2 {trade_setup['take_profit_2_price']:.2f}")

        market = state.get("market", {})
        if market.get("session_high"):
            fig.add_hline(y=market["session_high"],
                         line_color="#888888", line_dash="dot", line_width=1,
                         annotation_text="Session High")
            fig.add_hline(y=market["session_low"],
                         line_color="#888888", line_dash="dot", line_width=1,
                         annotation_text="Session Low")

        fig.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=1.02),
            xaxis_title=None,
            yaxis_title="Preis",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Warte auf {tf} Bars...")

    # ----------------------------------------------------------------
    # BEREICH 2 — Konfidenz-Analyse Panel
    # ----------------------------------------------------------------
    with st.expander("🔍 Konfidenz-Analyse — Wie wurde der Score berechnet?",
                      expanded=False):

        signals_list = state.get("signals", {}).get("signals", [])
        direction    = state.get("signals", {}).get("direction", "NEUTRAL")
        confidence   = state.get("signals", {}).get("confidence", 0)

        if not signals_list:
            st.info("Noch keine aktiven Signale.")
        else:
            st.markdown(f"### Gesamtkonfidenz: {confidence:.0%}")
            st.progress(confidence)

            if confidence >= 0.80:
                st.success("✅ SEHR HOCH — Signal wird ausgegeben")
            elif confidence >= 0.65:
                st.warning("⚡ HOCH — Signal wird ausgegeben")
            else:
                st.error("❌ ZU NIEDRIG — Kein Trade-Signal (Schwelle: 65%)")

            st.divider()

            st.markdown("**Aktive Signale im Detail:**")
            for s in signals_list:
                sig_type = s.get("type", "?")
                sig_dir  = s.get("direction", "?")
                sig_conf = s.get("confidence", 0)
                sig_desc = s.get("description", "")

                if sig_dir in ["BULLISH", "LONG"]:
                    icon = "🟢"
                elif sig_dir in ["BEARISH", "SHORT"]:
                    icon = "🔴"
                else:
                    icon = "⚪"

                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"{icon} **{sig_type}** — {sig_desc}")
                with col_b:
                    st.metric("Konfidenz", f"{sig_conf:.0%}")
                st.progress(sig_conf)

            st.divider()

            if confidence < 0.65:
                st.markdown("**Warum kein Trade-Signal?**")
                missing = []
                if len(signals_list) < 2:
                    missing.append("• Weniger als 2 übereinstimmende Signale")
                if confidence < 0.65:
                    missing.append(f"• Konfidenz {confidence:.0%} unter Schwelle (65%)")
                bearish = sum(1 for s in signals_list if s.get("direction") in ["BEARISH", "SHORT"])
                bullish = sum(1 for s in signals_list if s.get("direction") in ["BULLISH", "LONG"])
                if bearish > 0 and bullish > 0:
                    missing.append("• Signale widersprechen sich (gemischt bullish/bearish)")
                for m in missing:
                    st.markdown(m)

        st.caption(f"Berechnet: {state.get('last_update', '—')} | "
                  f"Nächstes Update: ~60s")

    # ----------------------------------------------------------------
    # HAUPTSPALTEN — Markt / Signale / Claude
    # ----------------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    market  = state.get("market", {})
    signals = state.get("signals", {})
    claude  = state.get("claude", {})

    with col1:
        st.subheader("Markt")
        last_price = market.get("last_price", 0)
        st.metric("Preis", f"{last_price:.2f}" if last_price else "–")
        if market.get("session_high"):
            st.metric("Session High", f"{market['session_high']:.2f}")
            st.metric("Session Low",  f"{market['session_low']:.2f}")
        if market.get("vwap"):
            st.metric("VWAP", f"{market['vwap']:.2f}")
        if market.get("vix"):
            st.metric("VIX", f"{market['vix']:.1f}")
        connected = state.get("connected", False)
        contract  = state.get("contract", "–")
        st.caption(f"Verbunden: {'✅' if connected else '❌'}  {contract}")

    with col2:
        st.subheader("Signale")
        direction  = signals.get("direction", "NEUTRAL")
        confidence = signals.get("confidence", 0)
        color = _direction_color(direction)
        st.markdown(
            f"<span style='color:{color}; font-size:1.8rem; font-weight:bold'>"
            f"{direction}</span>",
            unsafe_allow_html=True,
        )
        st.metric("Konfidenz", f"{confidence:.0%}")

        trade_setup = signals.get("trade_setup")
        if trade_setup and direction != "NEUTRAL":
            st.markdown("**Trade Setup:**")
            st.markdown(f"- Entry: `{trade_setup['entry_price']:.2f}`")
            st.markdown(f"- SL: `{trade_setup['stop_loss_price']:.2f}`")
            st.markdown(f"- TP1: `{trade_setup['take_profit_1_price']:.2f}`")
            st.markdown(f"- TP2: `{trade_setup['take_profit_2_price']:.2f}`")

    with col3:
        st.subheader("Claude KI")
        verdict = claude.get("verdict", "–")
        verdict_color = (
            "#00ff88" if verdict == "BESTÄTIGT" else
            "#ff4444" if verdict == "ABGELEHNT" else
            "#ffaa00"
        )
        st.markdown(
            f"<span style='color:{verdict_color}; font-size:1.4rem; font-weight:bold'>"
            f"{verdict}</span>",
            unsafe_allow_html=True,
        )
        if claude.get("begruendung"):
            st.markdown(claude["begruendung"])
        if claude.get("beachtung"):
            st.info(f"⚠️ {claude['beachtung']}")

    # Auto-refresh
    time.sleep(_REFRESH_S)
    st.rerun()


if __name__ == "__main__":
    main()
