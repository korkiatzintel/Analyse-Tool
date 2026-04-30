"""
Streamlit dashboard for NQ Futures Trading Assistant.

Reads shared state from logs/ui_state.json (written atomically by main.py on
every evaluation cycle). Auto-refreshes every 5 seconds via st.rerun().

Columns: Market data (30 %) | Signals (40 %) | AI analysis (30 %)
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="NQ Trading Assistant",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

_STATE_FILE   = Path(__file__).parent.parent / "logs" / "ui_state.json"
_COMMAND_FILE = Path(__file__).parent.parent / "logs" / "ui_command.json"
_REFRESH_S    = 5   # seconds between auto-reruns

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Price display */
.price-bull    { color:#2ea043; font-size:2.4rem; font-weight:800; line-height:1.1; }
.price-bear    { color:#f85149; font-size:2.4rem; font-weight:800; line-height:1.1; }
.price-neutral { color:#e6edf3; font-size:2.4rem; font-weight:800; line-height:1.1; }

/* Signal direction box */
.sig-long    { background:#0d3a22; border:1px solid #2ea043; border-radius:8px;
               padding:14px; text-align:center; }
.sig-short   { background:#3a0d0d; border:1px solid #f85149; border-radius:8px;
               padding:14px; text-align:center; }
.sig-neutral { background:#1e1e1e; border:1px solid #444;    border-radius:8px;
               padding:14px; text-align:center; }

/* Claude verdict cards */
.verdict-ok   { background:#0d3a22; border-left:4px solid #2ea043;
                border-radius:4px; padding:10px 14px; }
.verdict-no   { background:#3a0d0d; border-left:4px solid #f85149;
                border-radius:4px; padding:10px 14px; }
.verdict-wait { background:#1e1e1e; border-left:4px solid #666;
                border-radius:4px; padding:10px 14px; }

/* Calendar warning banner */
.event-warn { background:#2d2200; border:1px solid #e3b341; border-radius:6px;
              padding:10px 14px; color:#e3b341; font-weight:600; }

/* Signal pills */
.pill-bull { background:#2ea043; color:#fff; padding:2px 9px; border-radius:12px;
             font-size:.72rem; display:inline-block; margin:2px; }
.pill-bear { background:#f85149; color:#fff; padding:2px 9px; border-radius:12px;
             font-size:.72rem; display:inline-block; margin:2px; }
.pill-grey { background:#444;    color:#ccc; padding:2px 9px; border-radius:12px;
             font-size:.72rem; display:inline-block; margin:2px; }
</style>
""", unsafe_allow_html=True)


# ── State I/O ──────────────────────────────────────────────────────────────────

def _load_state() -> Optional[dict]:
    try:
        if not _STATE_FILE.exists():
            return None
        with _STATE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_command(cmd: dict) -> None:
    try:
        _COMMAND_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _COMMAND_FILE.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(cmd, f)
        tmp.replace(_COMMAND_FILE)
    except Exception:
        pass


# ── Session-state defaults ─────────────────────────────────────────────────────

def _init_ss() -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    defaults: dict = {
        "trade_log":            [],
        "api_calls_today":      0,
        "api_calls_date":       today,
        "_last_claude_ts":      "",
        "confidence_threshold": 65,
        "w_order_flow":         50,
        "w_technical":          30,
        "contract_type":        "MNQ (Micro)",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Reset daily counter at midnight
    if st.session_state["api_calls_date"] != today:
        st.session_state["api_calls_today"] = 0
        st.session_state["api_calls_date"]  = today


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def _sidebar(state: Optional[dict]) -> None:
    with st.sidebar:
        st.title("📈 NQ Assistant")
        st.divider()

        # ── Connection status ──────────────────────────────────────────────
        st.subheader("Verbindung")
        if state is None:
            st.markdown("🔴 **Offline** — warte auf Backend")
            st.caption(f"Suche: `{_STATE_FILE.name}`")
        else:
            mode      = state.get("mode", "demo")
            connected = state.get("connected", False)
            contract  = state.get("contract", "–")
            updated   = state.get("last_update", "–")

            icons = {"live": ("🟢", "Live (Rithmic)"),
                     "free": ("🟡", "Free (yfinance)"),
                     "demo": ("🔵", "Demo (synthetisch)")}
            icon, label = icons.get(mode, ("🔴", "Unbekannt"))
            if not connected:
                icon = "🔴"
            st.markdown(f"{icon} **{label}**")
            st.caption(f"Kontrakt: **{contract}**")
            st.caption(f"Update: {updated[-8:] if updated != '–' else '–'} UTC")

            # Tradovate feed status
            if state.get("tradovate_connected"):
                st.success("📡 Tradovate: Echtzeit")
            else:
                st.warning("📡 yfinance: ~15min Delay")

        st.divider()

        # ── Settings ───────────────────────────────────────────────────────
        st.subheader("Einstellungen")
        st.session_state["contract_type"] = st.radio(
            "Kontrakt",
            ["NQ (Standard)", "MNQ (Micro)"],
            index=["NQ (Standard)", "MNQ (Micro)"].index(
                st.session_state.get("contract_type", "MNQ (Micro)")
            ),
            horizontal=True,
        )
        st.session_state["confidence_threshold"] = st.slider(
            "Konfidenz-Schwelle", 50, 90,
            st.session_state["confidence_threshold"], 5, format="%d%%",
        )

        st.markdown("**Signal-Gewichtungen**")
        of_w = st.slider("Order Flow", 0, 100, st.session_state["w_order_flow"], 5)
        ta_w = st.slider("Technical",  0, 100, st.session_state["w_technical"],  5)
        ma_w = max(0, 100 - of_w - ta_w)
        st.session_state["w_order_flow"] = of_w
        st.session_state["w_technical"]  = ta_w
        st.caption(f"Macro (auto): **{ma_w}%**")

        st.divider()

        # ── Claude cost & cache stats ──────────────────────────────────────
        if state is not None:
            cs = state.get("claude_cost_stats", {})
            if cs.get("total_calls", 0) > 0:
                st.subheader("Claude API")
                hit_pct = cs.get("cache_hit_rate", 0.0) * 100
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Calls",  cs.get("total_calls", 0))
                    st.metric("Cache ✓", cs.get("cached_calls", 0))
                with c2:
                    st.metric("Hit-Rate", f"{hit_pct:.0f}%")
                    st.metric("Kosten",   f"${cs.get('estimated_cost_usd', 0):.4f}")
                if hit_pct >= 50:
                    st.caption("🟢 Cache aktiv — ~90 % Ersparnis auf gecachte Tokens")
                elif cs.get("total_calls", 0) == 1:
                    st.caption("⏳ Erster Call — Cache wird beim nächsten Aufruf greifen")
                else:
                    st.caption("🟡 Cache-Rate niedrig")
                st.divider()

        # ── Manual trade log ───────────────────────────────────────────────
        st.subheader("Trade-Log")
        with st.form("trade_entry", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                entry = st.number_input("Entry",      value=0.00, step=0.25, format="%.2f")
                size  = st.number_input("Kontrakte",  value=1,    min_value=1)
            with c2:
                exit_ = st.number_input("Exit",       value=0.00, step=0.25, format="%.2f")
                side  = st.selectbox("Seite", ["LONG", "SHORT"])
            saved = st.form_submit_button("Speichern", use_container_width=True)
            if saved and entry > 0 and exit_ > 0:
                # NQ: $20 per full point
                direction_factor = 1 if side == "LONG" else -1
                pnl = (exit_ - entry) * size * 20 * direction_factor
                st.session_state["trade_log"].append({
                    "ts": datetime.now(timezone.utc).strftime("%H:%M"),
                    "side": side, "size": size,
                    "entry": entry, "exit": exit_, "pnl": pnl,
                })

        log = st.session_state["trade_log"]
        if log:
            total_pnl = sum(t["pnl"] for t in log)
            wins      = sum(1 for t in log if t["pnl"] > 0)
            pnl_col   = "green" if total_pnl >= 0 else "red"
            st.markdown(
                f"**P&L:** :{pnl_col}[${total_pnl:+,.0f}]"
                f"  |  Win: {wins}/{len(log)}"
            )
            with st.expander("Verlauf"):
                for t in reversed(log[-15:]):
                    icon = "🟢" if t["pnl"] > 0 else "🔴"
                    st.caption(
                        f"{icon} {t['ts']} {t['side']} "
                        f"{t['size']}×{t['entry']:.2f}→{t['exit']:.2f} "
                        f"${t['pnl']:+,.0f}"
                    )
            if st.button("Log löschen", use_container_width=True):
                st.session_state["trade_log"] = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Marktdaten
# ══════════════════════════════════════════════════════════════════════════════

def _col_market(state: dict) -> None:
    st.subheader("Marktdaten")

    market  = state.get("market", {})
    price   = market.get("last_price") or 0.0
    vwap    = market.get("vwap")       or 0.0
    s_high  = market.get("session_high")
    s_low   = market.get("session_low")
    spread  = market.get("spread")     or 0.0
    vix     = state.get("vix", 0.0)
    vix_reg = state.get("vix_regime", "unknown").upper()
    mode    = state.get("mode", "demo")

    # ── Price (large, coloured by VWAP position) ───────────────────────────
    vs_vwap = price - vwap if vwap else 0.0
    if vs_vwap > 0:
        p_cls = "price-bull"
    elif vs_vwap < 0:
        p_cls = "price-bear"
    else:
        p_cls = "price-neutral"
    st.markdown(
        f'<div style="text-align:center;margin-bottom:6px;">'
        f'<span class="{p_cls}">{price:,.2f}</span><br>'
        f'<span style="color:#888;font-size:.82rem;">NQ Futures</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Key metrics grid ───────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "VWAP", f"{vwap:.2f}" if vwap else "–",
            delta=f"{vs_vwap:+.1f} Pkt" if vwap else None,
            delta_color="normal",
        )
        st.metric("Session High", f"{s_high:.2f}" if s_high else "–")
    with c2:
        st.metric("Spread", f"{spread:.2f} Pkt" if spread else "–")
        st.metric("Session Low",  f"{s_low:.2f}"  if s_low  else "–")

    st.divider()

    # ── VIX & Yield ────────────────────────────────────────────────────────
    vix_icons = {"LOW": "🟢", "NORMAL": "🟡", "HIGH": "🔴", "EXTREME": "🚨"}
    v_icon    = vix_icons.get(vix_reg, "⚪")
    st.markdown(
        f"**VIX:** {vix:.1f}&nbsp;&nbsp;{v_icon}&nbsp;**{vix_reg}**",
        unsafe_allow_html=True,
    )
    y10 = state.get("yield_10y", 0.0)
    if y10:
        st.caption(f"10j Yield: {y10:.2f}%")
    if vix_reg == "EXTREME":
        st.error("VIX > 30 — alle Signale blockiert")
    elif vix_reg == "HIGH":
        st.warning("VIX > 25 — Konfidenz ×0.8")

    st.divider()

    # ── Cumulative Delta chart ─────────────────────────────────────────────
    st.markdown("**Cumulative Delta** (letzte 60 Datenpunkte)")
    delta_hist = state.get("delta_history", [])
    if len(delta_hist) >= 2:
        import pandas as pd
        df_d = pd.DataFrame({"Delta": delta_hist[-60:]})
        st.line_chart(df_d, use_container_width=True, height=115)
    else:
        st.caption("Warte auf Delta-Daten…")

    st.divider()

    # ── Mode-specific extras ───────────────────────────────────────────────
    if mode == "free":
        updated  = state.get("last_update", "–")
        next_s   = state.get("next_update_in")
        ts_short = updated[-19:-4] if len(updated) > 19 else updated
        st.markdown(f"📡 **yfinance** | `{ts_short}`")
        if next_s is not None:
            st.caption(f"Nächstes Update in **{next_s:.0f}s**")

    elif mode == "live":
        st.markdown("**Order Book L2** (Top 5)")
        bids = market.get("bid_ladder", [])[:5]
        asks = market.get("ask_ladder", [])[:5]
        if bids or asks:
            _order_book_html(asks, bids)
        else:
            st.caption("Kein L2 Book verfügbar")

    elif mode == "demo":
        st.info("Demo-Modus (synthetische Daten)")


def _order_book_html(asks: list, bids: list) -> None:
    rows = ""
    for a in reversed(asks):
        rows += (
            f'<tr>'
            f'<td style="color:#f85149;font-family:monospace;padding:1px 4px">'
            f'{a["price"]:.2f}</td>'
            f'<td style="color:#888;text-align:right;padding:1px 4px">{a["size"]}</td>'
            f'</tr>'
        )
    rows += '<tr><td colspan="2" style="border-top:1px solid #333;height:3px"></td></tr>'
    for b in bids:
        rows += (
            f'<tr>'
            f'<td style="color:#2ea043;font-family:monospace;padding:1px 4px">'
            f'{b["price"]:.2f}</td>'
            f'<td style="color:#888;text-align:right;padding:1px 4px">{b["size"]}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<table style="width:100%;font-size:.82rem;border-collapse:collapse">'
        f'<thead><tr>'
        f'<th style="color:#666;text-align:left;font-weight:500;padding:2px 4px">Preis</th>'
        f'<th style="color:#666;text-align:right;font-weight:500;padding:2px 4px">Größe</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MIDDLE COLUMN — Signale
# ══════════════════════════════════════════════════════════════════════════════

def _col_signals(state: dict) -> None:
    threshold  = st.session_state.get("confidence_threshold", 65) / 100.0
    scan       = state.get("scan", {})
    candidates = scan.get("candidates", [])

    st.subheader("📊 Trade Scanner — Top 3 Setups")

    if not candidates:
        st.info("Scanner wertet aus...")
        return

    for c in candidates:
        conf      = c.get("confidence", 0)
        direction = c.get("direction", "?")
        is_signal = conf >= threshold
        rank      = c.get("rank", "?")
        label     = c.get("label", direction)

        with st.container():
            col_rank, col_dir, col_conf = st.columns([1, 3, 2])
            with col_rank:
                st.markdown(f"### #{rank}")
            with col_dir:
                arrow = "▲" if direction == "LONG" else "▼"
                if is_signal:
                    if direction == "LONG":
                        st.success(f"{arrow} {direction} 🔔 SIGNAL")
                    else:
                        st.error(f"{arrow} {direction} 🔔 SIGNAL")
                else:
                    st.info(f"{arrow} {direction} — Kandidat")
            with col_conf:
                st.metric("Konfidenz", f"{conf:.0%}")

            st.progress(conf)

            sigs = c.get("signals", [])
            for s in sigs:
                icon = ("🟢" if s.get("direction") in ["BULLISH", "LONG"]
                        else "🔴")
                st.caption(f"{icon} {s.get('type', '?')}: {s.get('description', '')}")

            ts = c.get("trade_setup")
            if ts and is_signal:
                c1, c2, c3 = st.columns(3)
                c1.metric("Entry", f"{ts.get('entry_price', 0):.2f}")
                c2.metric(
                    "SL",
                    f"{ts.get('stop_loss_price', 0):.2f} "
                    f"({ts.get('stop_loss_ticks', 0)} Ticks)",
                )
                c3.metric(
                    "TP1",
                    f"{ts.get('take_profit_1_price', 0):.2f} "
                    f"({ts.get('take_profit_1_ticks', 0)} Ticks)",
                )

            st.divider()

    # ── Calendar warning + events (kept from previous column) ─────────────
    if state.get("event_window_active"):
        hi    = [e for e in state.get("upcoming_events", []) if e.get("impact") == "high"]
        names = ", ".join(e.get("name", "?") for e in hi[:2])
        mins  = state.get("minutes_to_next_event")
        msg   = (f"⚠️ High-Impact Event in {float(mins):.0f} Min — Signale pausiert"
                 if (mins and float(mins) > 0) else
                 "⚠️ Post-Event Wartezeit aktiv — Signale pausiert")
        if names:
            msg += f": {names}"
        st.markdown(f'<div class="event-warn">{msg}</div>', unsafe_allow_html=True)
        st.markdown("")

    events = [
        e for e in state.get("upcoming_events", [])
        if e.get("impact") in ("high", "medium")
    ]
    if events:
        rows = []
        for ev in sorted(events, key=lambda x: x.get("minutes_away") or 9999):
            impact = ev.get("impact", "low")
            icon   = "🔴" if impact == "high" else "🟡"
            mins   = ev.get("minutes_away")
            t_str  = ev.get("time", "–")
            if mins is not None:
                try:
                    t_str += f" ({float(mins):+.0f}min)"
                except (TypeError, ValueError):
                    pass
            rows.append({"Zeit": t_str, "Event": ev.get("name", "–")[:48], "⚡": icon})
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, use_container_width=True,
            height=min(210, 42 + 35 * len(rows)),
        )


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — KI Analyse
# ══════════════════════════════════════════════════════════════════════════════

def _col_ai(state: dict) -> None:
    st.subheader("KI Analyse")

    claude  = state.get("claude", {})
    memory  = state.get("claude_memory", [])
    verdict = claude.get("verdict", "–")
    beg     = claude.get("begruendung", "")
    beacht  = claude.get("beachtung", "")
    ts_str  = claude.get("timestamp", "")

    # ── Verdict card ───────────────────────────────────────────────────────
    if verdict == "BESTÄTIGT":
        css, emoji, col = "verdict-ok",   "✅", "#2ea043"
    elif verdict == "ABGELEHNT":
        css, emoji, col = "verdict-no",   "❌", "#f85149"
    elif verdict == "FEHLER":
        css, emoji, col = "verdict-no",   "⚠️", "#e3b341"
    elif verdict in ("WARTE", "ÜBERSPRUNGEN"):
        css, emoji, col = "verdict-wait", "⏳", "#888"
    else:
        css, emoji, col = "verdict-wait", "–",  "#555"

    st.markdown(
        f'<div class="{css}">'
        f'<span style="font-size:1.35rem;font-weight:700;color:{col}">'
        f'{emoji} {verdict}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    if beg:
        st.markdown("**Begründung**")
        st.markdown(beg)
    if beacht:
        st.markdown("**Beachtung**")
        st.info(beacht)
    if ts_str:
        ts_short = ts_str[-19:] if len(ts_str) >= 19 else ts_str
        st.caption(f"Analysiert: {ts_short}")

    # Track new API calls across reruns
    if ts_str and ts_str != st.session_state.get("_last_claude_ts", ""):
        st.session_state["api_calls_today"] = (
            st.session_state.get("api_calls_today", 0) + 1
        )
        st.session_state["_last_claude_ts"] = ts_str

    st.divider()

    # ── Force-analyse button ───────────────────────────────────────────────
    if st.button(
        "🔍 Jetzt analysieren", use_container_width=True,
        help="Sendet Force-Analyse-Request an main.py",
    ):
        _write_command({
            "force_analyze": True,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        st.toast("Analyse angefordert!", icon="🔍")

    # ── Cost & cache stats (from backend) ─────────────────────────────────
    cs = state.get("claude_cost_stats", {})
    total   = cs.get("total_calls", 0)
    cached  = cs.get("cached_calls", 0)
    hit_pct = cs.get("cache_hit_rate", 0.0) * 100
    cost    = cs.get("estimated_cost_usd", 0.0)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("API Calls", total,
                  help="Calls seit Prozessstart")
        st.metric("Cache Hits", f"{cached}",
                  delta=f"{hit_pct:.0f}% Hit-Rate" if total else None,
                  delta_color="normal")
    with c2:
        st.metric("Kosten (gesamt)", f"${cost:.4f}",
                  help="Input $3/MTok · Cache $0.30/MTok · Output $15/MTok")
        if total == 1:
            st.caption("⏳ 1. Call — Cache greift ab dem 2.")
        elif hit_pct >= 50:
            st.caption("🟢 Cache aktiv")
        elif total > 1:
            st.caption("🟡 Cache-Rate niedrig")

    st.divider()

    # ── Verdict timeline (last 5) ──────────────────────────────────────────
    st.markdown("**Analyse-Verlauf**")
    shown = memory[:5]          # memory is newest-first from get_memory()
    if shown:
        for entry in shown:
            v       = entry.get("verdict", "–")
            skipped = entry.get("skipped", False)
            v_dir   = entry.get("direction", "")
            v_conf  = entry.get("confidence", 0.0)
            v_ts    = entry.get("timestamp", "")
            ts_disp = v_ts[-8:] if len(v_ts) >= 8 else v_ts   # HH:MM:SS

            if skipped:
                icon, bc = "⏭️", "#444"
            elif v == "BESTÄTIGT":
                icon, bc = "✅", "#2ea043"
            elif v == "ABGELEHNT":
                icon, bc = "❌", "#f85149"
            elif v == "FEHLER":
                icon, bc = "⚠️", "#e3b341"
            else:
                icon, bc = "⏳", "#666"

            st.markdown(
                f'<div style="border-left:3px solid {bc};padding:3px 9px;margin:3px 0;">'
                f'<span style="color:{bc}">{icon} <b>{v}</b></span>'
                f'<span style="color:#555;font-size:.77rem;margin-left:6px">'
                f'{v_dir} {v_conf:.0%} | {ts_disp}'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Noch keine Analysen in dieser Session")

    # ── Signal reasoning (collapsible) ────────────────────────────────────
    reasoning = state.get("signals", {}).get("reasoning", "")
    if reasoning:
        with st.expander("Signal-Reasoning"):
            st.caption(reasoning)


# ══════════════════════════════════════════════════════════════════════════════
# NO-DATA SCREEN
# ══════════════════════════════════════════════════════════════════════════════

def _no_data_screen() -> None:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("## ⏳ Warte auf Daten…")
        st.markdown(
            "Das Backend `main.py` läuft noch nicht oder hat noch keine Daten "
            "gesendet. Starte es in einem anderen Terminal:"
        )
        st.code(
            "cd nq_trading_assistant\n"
            "python main.py          # yfinance Free-Mode (Standard)\n"
            "python main.py --demo   # Demo-Modus, kein Netzwerk nötig\n"
            "python main.py --live   # Rithmic L2 (Credentials erforderlich)",
            language="bash",
        )
        st.caption(
            f"Erwartete State-Datei: `{_STATE_FILE}`  "
            f"| Auto-Refresh in {_REFRESH_S}s"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_ss()

    state = _load_state()
    _sidebar(state)

    # ── No data yet ────────────────────────────────────────────────────────
    if state is None:
        _no_data_screen()
        time.sleep(_REFRESH_S)
        st.rerun()
        return

    mode     = state.get("mode", "demo")
    contract = state.get("contract", "NQ")
    price    = state.get("market", {}).get("last_price") or 0.0

    # ── Header ─────────────────────────────────────────────────────────────
    mode_badges = {"live": "🟢 Live", "free": "🟡 Free", "demo": "🔵 Demo"}
    badge       = mode_badges.get(mode, mode)
    st.markdown(
        f"<h2 style='margin:0 0 2px 0;'>"
        f"📈 NQ Trading Assistant"
        f"<span style='color:#555;font-size:.88rem;font-weight:400'>"
        f" | {contract} | {price:,.2f} | {badge}"
        f"</span></h2>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Chart Timeframe-Auswahl ─────────────────────────────────────────────
    tf = st.radio("Chart Timeframe", ["1m", "5m", "15m"],
                  horizontal=True, index=1)
    lookback = st.select_slider(
        "Zeitraum",
        options=["2h", "4h", "8h", "12h", "24h", "48h"],
        value="8h",
    )
    lookback_map = {
        "1m":  {"2h": 120, "4h": 240, "8h": 480, "12h": 720, "24h": 1440, "48h": 2880},
        "5m":  {"2h": 24,  "4h": 48,  "8h": 96,  "12h": 144, "24h": 288,  "48h": 576},
        "15m": {"2h": 8,   "4h": 16,  "8h": 32,  "12h": 48,  "24h": 96,   "48h": 192},
    }
    n_bars = lookback_map.get(tf, {}).get(lookback, 96)
    bars = state.get("bars", {}).get(tf, [])[-n_bars:]
    if bars:
        df = pd.DataFrame(bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="NQ",
            increasing_line_color="#00ff88", decreasing_line_color="#ff4444",
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
            fig.add_hline(
                y=trade_setup["entry_price"], line_color=color,
                line_width=2,
                annotation_text=f"Entry {trade_setup['entry_price']:.2f}",
            )
            fig.add_hline(
                y=trade_setup["stop_loss_price"], line_color="#ff0000",
                line_dash="dash",
                annotation_text=f"SL {trade_setup['stop_loss_price']:.2f}",
            )
            fig.add_hline(
                y=trade_setup["take_profit_1_price"], line_color="#00ff88",
                line_dash="dash",
                annotation_text=f"TP1 {trade_setup['take_profit_1_price']:.2f}",
            )
            fig.add_hline(
                y=trade_setup["take_profit_2_price"], line_color="#00ff88",
                line_dash="dot",
                annotation_text=f"TP2 {trade_setup['take_profit_2_price']:.2f}",
            )
        market = state.get("market", {})
        if market.get("session_high"):
            fig.add_hline(
                y=market["session_high"], line_color="#888888",
                line_dash="dot", annotation_text="Session High",
            )
            fig.add_hline(
                y=market["session_low"], line_color="#888888",
                line_dash="dot", annotation_text="Session Low",
            )
        first_ts = pd.to_datetime(bars[0]["timestamp"])
        last_ts  = pd.to_datetime(bars[-1]["timestamp"])
        fig.update_layout(
            template="plotly_dark", height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_rangeslider_visible=False,
        )
        fig.update_xaxes(range=[first_ts, last_ts])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Warte auf {tf} Bars...")

    # ── Konfidenz-Analyse Panel ─────────────────────────────────────────────
    with st.expander("🔍 Konfidenz-Analyse", expanded=False):
        signals_list = state.get("signals", {}).get("signals", [])
        confidence   = state.get("signals", {}).get("confidence", 0)
        st.markdown(f"### Gesamtkonfidenz: {confidence:.0%}")
        st.progress(confidence)
        if confidence >= 0.80:
            st.success("✅ SEHR HOCH — Signal wird ausgegeben")
        elif confidence >= 0.65:
            st.warning("⚡ HOCH — Signal wird ausgegeben")
        else:
            st.error("❌ ZU NIEDRIG — Kein Trade-Signal (Schwelle: 65%)")
        st.divider()
        for s in signals_list:
            s_dir = s.get("direction")
            icon  = ("🟢" if s_dir in ["BULLISH", "LONG"] else
                     "🔴" if s_dir in ["BEARISH", "SHORT"] else "⚪")
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"{icon} **{s.get('type', '?')}** — {s.get('description', '')}")
            with col_b:
                st.metric("", f"{s.get('confidence', 0):.0%}")
            st.progress(s.get("confidence", 0))
        st.caption(
            f"Berechnet: {state.get('last_update', '—')} | Nächstes Update: ~60s"
        )

    # ── Three columns ───────────────────────────────────────────────────────
    left, mid, right = st.columns([3, 4, 3])

    with left:
        _col_market(state)

    with mid:
        _col_signals(state)

    with right:
        _col_ai(state)

    # ── Footer / auto-refresh countdown ────────────────────────────────────
    now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    st.markdown(
        f'<p style="color:#333;font-size:.70rem;text-align:right;margin-top:6px;">'
        f'Auto-Refresh alle {_REFRESH_S}s | {now_utc}</p>',
        unsafe_allow_html=True,
    )

    time.sleep(_REFRESH_S)
    st.rerun()


if __name__ == "__main__":
    main()
