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
import streamlit as st
import streamlit.components.v1 as components

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
        "mnq_contracts":        5,
        "eur_usd_rate":         0.92,
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

            # Dynamic yfinance countdown (60s cycle)
            try:
                last_dt = datetime.fromisoformat(updated.replace(" UTC", ""))
                seconds_ago = int(
                    (datetime.now(timezone.utc).replace(tzinfo=None)
                     - last_dt).total_seconds()
                )
                next_update_in = max(0, 60 - seconds_ago)
                if next_update_in > 10:
                    st.metric("Nächstes yfinance Update", f"{next_update_in}s")
                else:
                    st.warning(f"⏳ Update in {next_update_in}s…")
            except Exception:
                st.caption(f"Update: {updated[-8:] if updated != '–' else '–'} UTC")

            # Dynamic realtime price countdown (30s cycle)
            realtime_last = state.get("realtime_last_update", "")
            try:
                rt_dt  = datetime.fromisoformat(realtime_last.replace(" UTC", ""))
                rt_ago = int(
                    (datetime.now(timezone.utc).replace(tzinfo=None)
                     - rt_dt).total_seconds()
                )
                rt_next = max(0, 30 - rt_ago)
                st.metric("Preis Update (Barchart)", f"{rt_next}s")
            except Exception:
                pass

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

        st.markdown("**Trade Journal**")
        st.session_state["mnq_contracts"] = st.number_input(
            "MNQ Kontrakte", min_value=1, max_value=20,
            value=st.session_state["mnq_contracts"], step=1,
            help="Anzahl MNQ Kontrakte für P&L Berechnung",
        )
        st.session_state["eur_usd_rate"] = st.number_input(
            "EUR/USD Rate", min_value=0.80, max_value=1.20,
            value=st.session_state["eur_usd_rate"], step=0.01, format="%.2f",
            help="Wechselkurs für EUR Umrechnung",
        )

        st.divider()

        # ── KI Provider stats ──────────────────────────────────────────────
        if state is not None:
            cs            = state.get("claude_cost_stats", {})
            provider      = cs.get("provider", "")
            current_model = cs.get("current_model", "")
            if cs.get("total_calls", 0) > 0:
                if "Gemini" in provider:
                    st.subheader("KI Provider")
                    if "2.5" in current_model:
                        st.success(f"🧠 {current_model}")
                    else:
                        st.warning(f"🔄 {current_model} (Fallback)")
                    failures = cs.get("model_failures", {})
                    if any(v > 0 for v in failures.values()):
                        st.caption(
                            "gemini-2.5-flash Quota heute erreicht → "
                            "gemini-2.0-flash-lite aktiv"
                        )
                    # Tages-Limit Anzeige
                    calls_today  = cs.get("calls_today", 0)
                    max_calls    = cs.get("max_calls_per_day", 10)
                    remaining    = cs.get("calls_remaining", max_calls)
                    if remaining > 5:
                        st.success(f"🧠 Gemini: {remaining}/{max_calls} Calls verfügbar")
                    elif remaining > 0:
                        st.warning(f"⚠️ Gemini: Noch {remaining} Calls heute")
                    else:
                        st.error("🔴 Gemini: Tageslimit erreicht")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Heute", calls_today)
                    with c2:
                        available = len(cs.get("models_available", []))
                        st.metric("Modelle", f"{available}/{len(cs.get('model_failures', {}))}")
                else:
                    st.subheader("Claude API")
                    hit_pct = cs.get("cache_hit_rate", 0.0) * 100
                    c1, c2  = st.columns(2)
                    with c1:
                        st.metric("Calls",   cs.get("total_calls", 0))
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

    market         = state.get("market", {})
    realtime_price = state.get("realtime_price", 0.0) or 0.0
    market_price   = market.get("last_price") or 0.0

    if realtime_price > 0:
        price        = realtime_price
        price_source = "📡 Barchart (~30s)"
    else:
        price        = market_price
        price_source = "📊 yfinance (~15min)"

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
        f'<span style="color:#888;font-size:.82rem;">NQ Futures</span><br>'
        f'<span style="color:#555;font-size:.72rem;">{price_source}</span>'
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

    # ── Limit Order Levels ─────────────────────────────────────────────────
    st.subheader("🎯 Limit Orders — Vorausschauende Entries")
    limit_orders  = state.get("limit_orders", [])
    current_price = (state.get("realtime_price") or
                     state.get("market", {}).get("last_price", 0) or 0)

    if not limit_orders:
        st.info("Warte auf klaren Bias für Limit-Order Berechnung...")
    else:
        for order in limit_orders:
            status      = order.get("status", "WAITING")
            direction   = order.get("direction", "?")
            limit_price = order.get("limit_price", 0)
            order_type  = order.get("type", "?")
            trigger_txt = order.get("trigger", "")
            valid_until = order.get("valid_until", "")
            distance    = abs(current_price - limit_price) if current_price > 0 else 0

            if status == "TRIGGERED":
                st.success(
                    f"✅ AUSGELÖST — {direction} @ {limit_price:.2f} | "
                    f"ausgelöst um {order.get('triggered_at', '?')}"
                )
            else:
                if direction == "LONG":
                    st.info("⏳ WARTE AUF KAUFLEVEL")
                else:
                    st.warning("⏳ WARTE AUF VERKAUFLEVEL")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric(
                "Limit Preis",
                f"{limit_price:.2f}",
                delta=f"{current_price - limit_price:+.2f} vom Markt"
                      if current_price > 0 else None,
            )
            col_b.metric(
                "Stop Loss",
                f"{order.get('stop_loss', 0):.2f}",
                delta=f"{order.get('stop_loss_ticks', 0)} Ticks",
            )
            col_c.metric(
                "Take Profit 1",
                f"{order.get('take_profit_1', 0):.2f}",
                delta=f"+{order.get('take_profit_1_ticks', 0)} Ticks",
            )
            st.caption(
                f"📌 {trigger_txt} | Typ: {order_type} | "
                f"Gültig bis: {valid_until} | "
                f"Abstand: {distance:.1f} Punkte | "
                f"RR: 1:{order.get('risk_reward', 1.5)}"
            )
            st.divider()

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
                tp1_ticks = ts.get("take_profit_1_ticks", 0)
                if tp1_ticks >= 200:
                    tp_badge = "🚀"
                elif tp1_ticks >= 120:
                    tp_badge = "✅"
                elif tp1_ticks >= 80:
                    tp_badge = "✅"
                else:
                    tp_badge = "⚠️"
                c3.metric(
                    "TP1",
                    f"{ts.get('take_profit_1_price', 0):.2f} "
                    f"({tp1_ticks} Ticks) {tp_badge}",
                )
                if ts.get("tp_adjusted"):
                    st.caption("📐 SL erweitert — TP1 Minimum (80 Ticks) durchgesetzt")

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

_LEARNING_RESULT  = Path(__file__).parent.parent / "logs" / "last_learning_result.json"
_LEARNING_TRIGGER = Path(__file__).parent.parent / "logs" / "run_learning.trigger"


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

    tab1, tab2, tab3 = st.tabs(["📈 Live Trading", "📊 Trade Journal", "🧠 KI Lernen"])

    with tab1:
        # ── Chart Timeframe-Auswahl ─────────────────────────────────────────
        from ui.chart_component import build_chart_html

        tf = st.radio("Chart Timeframe", ["1m", "5m", "15m", "1h"],
                      horizontal=True, index=1, key="chart_tf")

        bars        = state.get("bars", {}).get(tf, [])
        sigs_state  = state.get("signals", {})
        limit_orders = state.get("limit_orders", [])

        chart_html = build_chart_html(bars, sigs_state, limit_orders, height=480)
        components.html(chart_html, height=500, scrolling=False)

        # ── Trade Setup Karte ───────────────────────────────────────────────
        st.subheader("📋 Trade Setup")

        trade_setup = sigs_state.get("trade_setup")
        direction   = sigs_state.get("direction", "NEUTRAL")
        confidence  = sigs_state.get("confidence", 0.0)
        contracts   = st.session_state.get("mnq_contracts", 5)
        mnq_tick    = 0.50

        if not trade_setup or direction == "NEUTRAL" or confidence < 0.65:
            st.info("⏳ Warte auf Signal mit >65% Konfidenz…")
        else:
            if direction == "LONG":
                st.success(f"## ▲ BUY — LONG Setup ({confidence:.0%})")
            else:
                st.error(f"## ▼ SELL — SHORT Setup ({confidence:.0%})")

            entry  = trade_setup.get("entry_price", 0) or 0
            sl     = trade_setup.get("stop_loss_price", 0) or 0
            tp1    = trade_setup.get("take_profit_1_price", 0) or 0
            tp2    = trade_setup.get("take_profit_2_price", 0) or 0
            sl_t   = trade_setup.get("stop_loss_ticks", 0) or (
                int(abs(sl - entry) / 0.25) if sl and entry else 0)
            tp1_t  = trade_setup.get("take_profit_1_ticks", 0) or (
                int(abs(tp1 - entry) / 0.25) if tp1 and entry else 0)
            tp2_t  = trade_setup.get("take_profit_2_ticks", 0) or (
                int(abs(tp2 - entry) / 0.25) if tp2 and entry else 0)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("### 🎯 Entry")
                st.markdown(f"## **{entry:.2f}**")
                st.caption("Limit Order setzen")
            with c2:
                st.markdown("### 🛑 Stop Loss")
                st.markdown(f"## **{sl:.2f}**")
                st.markdown(f"**{sl_t} Ticks** Risiko")
                st.caption(f"${sl_t * mnq_tick * contracts:.0f} ({contracts} MNQ)")
            with c3:
                st.markdown("### 🎯 Take Profit")
                st.markdown(f"**TP1: {tp1:.2f}**")
                st.caption(f"+{tp1_t} Ticks / +${tp1_t * mnq_tick * contracts:.0f}")
                st.markdown(f"**TP2: {tp2:.2f}**")
                st.caption(f"+{tp2_t} Ticks / +${tp2_t * mnq_tick * contracts:.0f}")

            rr = tp1_t / sl_t if sl_t > 0 else 0
            st.markdown(f"**Risk/Reward: 1:{rr:.1f}**")
            st.info(
                f"📌 **Ausführung:**  "
                f"1. Limit bei **{entry:.2f}** · "
                f"2. Stop bei **{sl:.2f}** ({sl_t} Ticks) · "
                f"3. TP1 **{tp1:.2f}** / TP2 **{tp2:.2f}**"
            )

        # ── Markt-Bias Panel ───────────────────────────────────────────────
        bias = state.get("bias", {})
        if bias:
            bias_dir  = bias.get("direction", "NEUTRAL")
            bias_prob = bias.get("probability", 50)
            bull_prob = bias.get("bull_prob", 50)
            bear_prob = bias.get("bear_prob", 50)
            strength  = bias.get("strength", "WEAK")

            st.subheader("🧭 Markt-Bias")
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                if bias_dir == "LONG":
                    st.success(f"▲ BULLISH — {bias_prob:.0f}%")
                elif bias_dir == "SHORT":
                    st.error(f"▼ BEARISH — {bias_prob:.0f}%")
                else:
                    st.info(f"◆ NEUTRAL — {bias_prob:.0f}%")
                st.caption(f"Stärke: {strength}")
            with col_b2:
                st.metric("🟢 Bullish", f"{bull_prob:.0f}%")
                st.progress(bull_prob / 100)
            with col_b3:
                st.metric("🔴 Bearish", f"{bear_prob:.0f}%")
                st.progress(bear_prob / 100)
            # Premium/Discount Badge
            premium_discount = bias.get("premium_discount", "")
            pd_labels = {
                "PREMIUM":     "🔴 PREMIUM — Short bevorzugt",
                "DISCOUNT":    "🟢 DISCOUNT — Long bevorzugt",
                "EQUILIBRIUM": "⚪ EQUILIBRIUM — Neutral",
            }
            if premium_discount in pd_labels:
                st.caption(pd_labels[premium_discount])

            # 4-Ebenen Breakdown
            details = bias.get("details", {})
            with st.expander("🔍 Bias Details — 4 Ebenen"):
                level_names = {
                    "htf_bias":         "Ebene 1: 1H Struktur",
                    "session_context":  "Ebene 2: Session Kontext",
                    "key_levels":       "Ebene 3: PDH/PDL & ORB",
                    "ltf_confirmation": "Ebene 4: LTF 5m/15m",
                }
                for key, label in level_names.items():
                    level = details.get(key, {})
                    if level:
                        bull_l = level.get("bull", 0.5) * 100
                        bear_l = level.get("bear", 0.5) * 100
                        col_l, col_b, col_be = st.columns([3, 1, 1])
                        col_l.markdown(f"**{label}**")
                        col_b.metric("Bull", f"{bull_l:.0f}%")
                        col_be.metric("Bear", f"{bear_l:.0f}%")
                        for r in level.get("reasons", []):
                            st.caption(f"  • {r}")

            if bias_dir != "NEUTRAL" and bias_prob >= 65:
                st.info(
                    f"ℹ️ Nur {bias_dir}-Trades werden simuliert "
                    f"und als Signal ausgegeben."
                )

        # ── ICT Killzone & Confluence ───────────────────────────────────────
        ict = state.get("ict_signals", {})
        if ict and not ict.get("error"):
            st.subheader("⚔️ ICT Analyse")
            killzone  = ict.get("active_killzone")
            ict_score = ict.get("ict_score", 0.0)

            if killzone:
                st.success(f"🎯 **ICT Killzone aktiv: {killzone}** — Optimale Trading-Zeit!")
            else:
                st.info("⏰ Keine aktive Killzone — Nächste: NY AM (10:00–11:00 EST)")

            if ict_score > 0:
                col_ict1, col_ict2 = st.columns(2)
                with col_ict1:
                    st.metric("ICT Confluence Score", f"{ict_score:.0%}")
                    st.progress(ict_score)
                with col_ict2:
                    htf = ict.get("htf_bias", {})
                    ms  = ict.get("market_structure", {})
                    if htf:
                        st.metric(
                            "HTF Bias (1h)",
                            htf.get("type", "?"),
                            delta=htf.get("direction", ""),
                        )
                    elif ms:
                        st.metric(
                            "Market Structure (15m)",
                            ms.get("type", "?"),
                            delta=ms.get("direction", ""),
                        )
                    else:
                        ob_count  = len(ict.get("order_blocks", []))
                        fvg_count = len(ict.get("fvg_levels", []))
                        st.metric("Order Blocks / FVGs", f"{ob_count} OB / {fvg_count} FVG")

                reasons = ict.get("ict_reasons", [])
                if reasons:
                    with st.expander("📋 ICT Signal Details"):
                        for r in reasons:
                            st.markdown(f"• {r}")

        # ── Konfidenz-Analyse Panel ─────────────────────────────────────────
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

        # ── Three columns ───────────────────────────────────────────────────
        left, mid, right = st.columns([3, 4, 3])
        with left:
            _col_market(state)
        with mid:
            _col_signals(state)
        with right:
            _col_ai(state)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — Trade Journal
    # ══════════════════════════════════════════════════════════════════════

    with tab2:
        sim_stats  = state.get("sim_stats", {})
        contracts  = st.session_state.get("mnq_contracts", 5)
        eur_rate   = st.session_state.get("eur_usd_rate", 0.92)
        mnq_tick   = 0.50  # $ per tick per MNQ contract

        if sim_stats.get("total", 0) > 0:
            # ── Gesamt-Statistik (5 Spalten) ──────────────────────────────
            total_ticks = sim_stats.get("total_pnl_ticks", 0)
            total_usd   = total_ticks * mnq_tick * contracts
            total_eur   = total_usd * eur_rate

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Trades", sim_stats["total"])
            c2.metric(
                "Win-Rate", f"{sim_stats['win_rate']}%",
                delta=f"{sim_stats['win_rate'] - 65:.1f}% vs Ziel 65%",
            )
            c3.metric("Gesamt Ticks", f"{total_ticks:+}")
            c4.metric("Gesamt USD",   f"${total_usd:+.0f}")
            c5.metric("Gesamt EUR",   f"€{total_eur:+.0f}")

            # ── Offene Trades ──────────────────────────────────────────────
            open_trades = state.get("open_trades", [])
            if open_trades:
                st.subheader(f"🔄 Offene Trades ({len(open_trades)})")
                for t in open_trades:
                    entry     = t.get("entry_price", 0)
                    sl        = t.get("stop_loss", 0)
                    tp1       = t.get("take_profit_1", 0)
                    sl_t      = int(abs(sl - entry) / 0.25)  if sl  and entry else 0
                    tp1_t     = int(abs(tp1 - entry) / 0.25) if tp1 and entry else 0
                    dir_icon  = "▲" if t["direction"] == "LONG" else "▼"
                    mode      = t.get("trade_mode", "TREND")
                    mode_icon = "🔄" if mode == "REVERSAL" else "📈"

                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.markdown(f"{dir_icon} **{t['direction']}** {mode_icon}")
                    col2.metric("Entry",     f"{entry:.2f}")
                    col3.metric("SL",        f"{sl:.2f} ({sl_t}T)")
                    col4.metric("TP1",       f"{tp1:.2f} ({tp1_t}T)")
                    col5.metric("Konfidenz", f"{t.get('confidence', 0):.0%}")
                    st.caption(
                        f"Signale: {', '.join(t.get('active_signals', []))} | "
                        f"Einstieg: {t.get('timestamp_entry', '')[:16]}"
                    )
                    st.divider()

            # ── Geschlossene Trades ────────────────────────────────────────
            recent = state.get("recent_trades", [])
            if recent:
                st.subheader("📋 Letzte abgeschlossene Trades")
                for t in recent:
                    direction = t.get("direction", "?")
                    entry     = t.get("entry_price", 0)
                    exit_p    = t.get("exit_price", 0)
                    outcome   = t.get("outcome", "?")
                    exit_r    = t.get("exit_reason", "?")
                    duration  = t.get("duration_minutes", 0)

                    # Ticks aus Backend oder lokal berechnen
                    pnl_ticks = t.get("pnl_ticks") or (
                        int(((exit_p - entry) if direction == "LONG"
                             else (entry - exit_p)) / 0.25)
                        if entry and exit_p else 0
                    )
                    sl_t  = t.get("sl_ticks")  or (
                        int(abs(t.get("stop_loss", 0) - entry) / 0.25) if entry else 0
                    )
                    tp1_t = t.get("tp1_ticks") or (
                        int(abs(t.get("take_profit_1", 0) - entry) / 0.25) if entry else 0
                    )

                    pnl_usd = pnl_ticks * mnq_tick * contracts
                    pnl_eur = pnl_usd * eur_rate

                    icon = "✅" if outcome == "WIN" else ("❌" if outcome == "LOSS" else "⏱️")
                    msg  = (
                        f"{icon} {direction} | {exit_r} | "
                        f"{pnl_ticks:+} Ticks | ${pnl_usd:+.0f} / €{pnl_eur:+.0f}"
                    )
                    if outcome == "WIN":
                        st.success(msg)
                    elif outcome == "LOSS":
                        st.error(msg)
                    else:
                        st.info(msg)

                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Entry",     f"{entry:.2f}")
                    col2.metric("Exit",      f"{exit_p:.2f}" if exit_p else "offen")
                    col3.metric("SL Abstand", f"{sl_t} Ticks")
                    col4.metric("TP1 Ziel",  f"{tp1_t} Ticks")
                    col5.metric("Dauer",     f"{duration:.0f} Min")

                    sigs      = t.get("active_signals", [])
                    mode      = t.get("trade_mode", "TREND")
                    mode_icon = "🔄" if mode == "REVERSAL" else "📈"
                    st.caption(
                        f"{mode_icon} {mode} | "
                        f"Conf: {t.get('confidence', 0):.0%} | "
                        f"Signale: {', '.join(sigs[:3])}"
                    )
                    st.divider()

            # ── Killzone Performance ───────────────────────────────────────
            kz_stats = sim_stats.get("win_rate_by_killzone", {})
            if kz_stats:
                st.subheader("🎯 Killzone Performance")
                col_kz1, col_kz2 = st.columns(2)
                kz_in  = kz_stats.get("IN_KILLZONE", 0)
                kz_out = kz_stats.get("OUTSIDE_KILLZONE", 0)
                col_kz1.metric(
                    "In Killzone",
                    f"{kz_in:.1f}%",
                    delta=f"{kz_in - kz_out:+.1f}% vs. außerhalb",
                )
                col_kz2.metric("Außerhalb Killzone", f"{kz_out:.1f}%")

            # ── Signal Performance ─────────────────────────────────────────
            best_signals = sim_stats.get("best_signal_types", {})
            if best_signals:
                st.subheader("🏆 Signal Performance")
                for sig, wr in best_signals.items():
                    col_a, col_b = st.columns([3, 1])
                    col_a.markdown(f"**{sig}**")
                    col_b.metric("Win-Rate", f"{wr}%")
                    st.progress(wr / 100)
        else:
            st.info("Noch keine simulierten Trades. App muss mindestens 60 Sekunden laufen.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — KI Lernen
    # ══════════════════════════════════════════════════════════════════════

    _LEARNING_RUNNING_FLAG = Path(__file__).parent.parent / "logs" / "learning_running.flag"
    _LEARNING_START_TS     = Path(__file__).parent.parent / "logs" / "learning_start.timestamp"

    _NAMEN_MAP = {
        "MULTI_TF_BIAS":    "Trend-Richtungsanalyse (Multi-Timeframe)",
        "FAIR_VALUE_GAP":   "Preislücken-Strategie (Fair Value Gap)",
        "VIX_REGIME":       "Markt-Volatilität Filter",
        "OVERNIGHT_GAP":    "Overnight-Lücken Strategie",
        "CALENDAR_FILTER":  "Wirtschaftskalender Filter",
        "EMA_TREND":        "Gleitender Durchschnitt Trend",
        "VWAP_POSITION":    "Tagesdurchschnittspreis (VWAP)",
        "RSI_EXTREME":      "Überkauft/Überverkauft Erkennung",
        "MEAN_REVERSION":   "Rückkehr zum Mittelwert",
        "SESSION_LEVELS":   "Tages-Hochs und Tiefs",
    }

    with tab3:
        st.subheader("🧠 Tägliche KI-Lernanalyse")

        # ── Ladeanimation wenn Analyse läuft ──────────────────────────────
        trigger_running = (
            _LEARNING_TRIGGER.exists() or _LEARNING_RUNNING_FLAG.exists()
        )
        if trigger_running:
            st.info("🧠 KI analysiert gerade deine Trades...")
            progress_bar = st.progress(0.0)
            status_text  = st.empty()
            steps = [
                "📊 Lade Trade-Historie...",
                "🔍 Analysiere Gewinner-Trades...",
                "❌ Analysiere Verlierer-Trades...",
                "🧩 Erkenne Muster...",
                "⚖️ Berechne neue Gewichtungen...",
                "✅ Schreibe Empfehlungen...",
            ]
            try:
                elapsed  = time.time() - float(_LEARNING_START_TS.read_text())
                step_idx = min(int(elapsed / 5), len(steps) - 1)
                progress = min(elapsed / 30, 0.95)
                status_text.markdown(f"**{steps[step_idx]}**")
                progress_bar.progress(progress)
            except Exception:
                progress_bar.progress(0.1)
                status_text.markdown(f"**{steps[0]}**")
            time.sleep(2)
            st.rerun()

        # ── Ergebnis-Anzeige ─────────────────────────────────────────────
        else:
            try:
                last_result = json.loads(
                    _LEARNING_RESULT.read_text(encoding="utf-8")
                )

                if last_result.get("status") == "success":
                    result_data       = last_result["result"]
                    stats_snap        = last_result["stats"]
                    analyse           = result_data.get("analyse", {})
                    signal_bewertung  = result_data.get("signal_bewertung", {})
                    neue_gewichtungen = result_data.get("neue_gewichtungen", {})
                    alte_gewichtungen = last_result.get("previous_weights", {})

                    st.success(
                        f"✅ Analyse abgeschlossen — {stats_snap['total']} Trades ausgewertet"
                    )
                    st.caption(
                        f"Win-Rate: {stats_snap['win_rate']}% | "
                        f"Gesamt P&L: ${stats_snap['total_pnl_usd']:+.0f}"
                    )
                    st.divider()

                    # Zusammenfassung
                    st.markdown("### 📋 Was hat die KI herausgefunden?")
                    zusammenfassung = analyse.get("zusammenfassung", "")
                    if zusammenfassung:
                        import re as _re
                        _clean = zusammenfassung.strip()
                        if _clean.startswith(("{", "[")):
                            try:
                                _parsed = json.loads(_clean)
                                _texts: list = []

                                def _collect(_o):
                                    if isinstance(_o, str) and len(_o) > 5:
                                        _texts.append(_o)
                                    elif isinstance(_o, list):
                                        for _i in _o:
                                            _collect(_i)
                                    elif isinstance(_o, dict):
                                        for _v in _o.values():
                                            _collect(_v)

                                _collect(_parsed)
                                _clean = "\n\n".join(_texts)
                            except Exception:
                                _clean = _re.sub(r'[{}\[\]":]', " ", _clean)
                                _clean = _re.sub(r"\s+", " ", _clean).strip()
                        # Split into numbered points at known separators
                        _lines: list = []
                        for _sep in (" | ", "\n", ". "):
                            if _sep in _clean:
                                _lines = [l.strip() for l in _clean.split(_sep) if l.strip()]
                                break
                        if _lines and len(_lines) > 1:
                            for _idx, _line in enumerate(_lines[:6], 1):
                                if _line:
                                    st.markdown(f"**{_idx}.** {_line}")
                        else:
                            st.markdown(f"> {_clean[:800]}")
                    else:
                        st.info("Noch keine Analyse vorhanden.")
                    st.divider()

                    # Strategien — Herzstück
                    st.markdown("### 🔧 Welche Strategien wurden angepasst?")
                    for sig_name, bewertung in signal_bewertung.items():
                        empfehlung  = bewertung.get("empfehlung", "BEIBEHALTEN")
                        begruendung = bewertung.get("begruendung", "")
                        win_rate    = bewertung.get("win_rate", 0)
                        alte_gew    = float(alte_gewichtungen.get(sig_name, 1.0))
                        neue_gew    = float(neue_gewichtungen.get(sig_name, 1.0))
                        delta       = neue_gew - alte_gew
                        anzeige     = _NAMEN_MAP.get(sig_name, sig_name)

                        col_icon, col_info, col_change = st.columns([1, 5, 2])
                        with col_icon:
                            if empfehlung == "STAERKEN":
                                st.markdown("### 📈")
                            elif empfehlung == "REDUZIEREN":
                                st.markdown("### 📉")
                            else:
                                st.markdown("### ➡️")
                        with col_info:
                            if empfehlung == "STAERKEN":
                                st.success(f"**{anzeige}** — wird stärker gewichtet")
                            elif empfehlung == "REDUZIEREN":
                                st.error(f"**{anzeige}** — wird weniger gewichtet")
                            else:
                                st.info(f"**{anzeige}** — bleibt unverändert")
                            st.caption(f"💬 {begruendung}")
                            st.caption(f"📊 Win-Rate dieser Strategie: {win_rate:.0f}%")
                        with col_change:
                            st.metric(
                                "Stellschraube", f"{neue_gew:.2f}",
                                delta=f"{delta:+.2f}" if abs(delta) > 0.01 else "±0",
                                delta_color="normal",
                            )
                            if abs(delta) > 0.01:
                                st.caption("⬆️ mehr Einfluss" if delta > 0
                                           else "⬇️ weniger Einfluss")
                        st.divider()

                    # Erkannte Muster
                    muster = analyse.get("muster", [])
                    if muster:
                        st.markdown("### 🔍 Erkannte Handelsmuster")
                        for m in muster:
                            st.markdown(f"• {m}")
                        st.divider()

                    # Handlungsempfehlungen
                    empfehlungen = result_data.get("handlungsempfehlungen", [])
                    if empfehlungen:
                        st.markdown("### ✅ Handlungsempfehlungen")
                        import re as _re2
                        for _i, _emp in enumerate(empfehlungen[:3], 1):
                            if _emp and isinstance(_emp, str):
                                _emp_clean = _re2.sub(r'[{}\[\]":]', "", _emp).strip()
                                if _emp_clean:
                                    st.markdown(f"**{_i}.** {_emp_clean}")

                    naechste = result_data.get("naechste_analyse_in", "")
                    if naechste:
                        st.info(f"📅 Empfehlung für nächste Analyse: {naechste}")

                    # ICT-spezifische Erkenntnisse
                    ict_emp = result_data.get("ict_empfehlungen", {})
                    if ict_emp:
                        st.divider()
                        st.markdown("### 🎯 ICT Erkenntnisse")

                        kz = ict_emp.get("killzone_filter_staerken", None)
                        if kz is True:
                            st.success("✅ Killzone-Filter bestätigt — außerhalb schlechtere Performance")
                        elif kz is False:
                            st.info("ℹ️ Killzone-Filter noch kein klarer Vorteil")

                        beste_ms = ict_emp.get("beste_market_structure", "")
                        if beste_ms:
                            st.info(f"📊 Beste Market Structure: **{beste_ms}**")

                        schwelle = ict_emp.get("ict_score_schwelle") or ict_emp.get("ict_score_schwelle_empfehlung", 0)
                        if schwelle:
                            st.caption(f"🎯 Empfohlene ICT Score-Schwelle: {schwelle:.2f}")

                        if ict_emp.get("order_block_pflicht") is True:
                            st.warning("⚠️ Order Blocks stark empfohlen")

                    # Parameter-Änderungen
                    update_report     = result_data.get("update_report", {})
                    param_begruendung = result_data.get("parameter_begruendung", {})

                    if update_report and update_report.get("accepted"):
                        st.divider()
                        st.markdown("### 🔧 Angepasste Strategie-Parameter")

                        _SECTION_NAMES = {
                            "SIGNAL_GEWICHTUNGEN":  "Signal-Gewichtung",
                            "KONFIDENZ_SCHWELLEN":  "Konfidenz-Schwelle",
                            "BIAS_PARAMETER":       "Bias-Erkennung",
                            "RISK_MANAGEMENT":      "Risiko-Management",
                            "VIX_REGIME_GRENZEN":   "Volatilitäts-Filter",
                            "KONTEXT_MODIFIKATOREN": "Tageszeit-Anpassung",
                            "LIMIT_ORDER_PARAMETER": "Limit-Order Einstellung",
                        }
                        _KEY_NAMES = {
                            "FAIR_VALUE_GAP":              "Preislücken-Strategie",
                            "MULTI_TF_BIAS":               "Trend-Richtungsanalyse",
                            "EMA_TREND":                   "Gleitender Durchschnitt",
                            "VWAP_POSITION":               "VWAP-Position",
                            "RSI_EXTREME":                 "Überkauft/Überverkauft",
                            "min_confidence_normal":       "Mindest-Konfidenz (normal)",
                            "min_confidence_vix_high":     "Mindest-Konfidenz (hohe Vola)",
                            "sl_atr_multiplier":           "Stop-Loss Größe",
                            "rth_open_bonus":              "Bonus erste Handelsstunde",
                            "vix_penalty_high":            "Abzug bei hoher Volatilität",
                        }

                        for change in update_report["accepted"]:
                            param = change["param"]
                            old   = change["old"]
                            new   = change["new"]
                            delta = change["change"]
                            parts = param.split(".")
                            section_k = parts[0] if parts else param
                            key_k     = parts[-1] if len(parts) > 1 else param

                            anzeige_section = _SECTION_NAMES.get(section_k, section_k)
                            anzeige_key     = _KEY_NAMES.get(key_k, key_k)
                            begruendung     = param_begruendung.get(param, "")

                            col1, col2 = st.columns([3, 2])
                            with col1:
                                if delta > 0:
                                    st.success(f"📈 **{anzeige_section}**: {anzeige_key}")
                                else:
                                    st.warning(f"📉 **{anzeige_section}**: {anzeige_key}")
                                if begruendung:
                                    st.caption(f"💬 {begruendung}")
                            with col2:
                                st.metric(
                                    "Änderung",
                                    f"{new:.3f}",
                                    delta=f"{delta:+.3f}",
                                    delta_color="normal",
                                )
                            st.divider()

                elif last_result.get("status") == "insufficient_data":
                    st.warning(last_result["message"])

            except FileNotFoundError:
                st.info("Noch keine Lernanalyse durchgeführt.")
            except Exception:
                st.info("Noch keine Lernanalyse durchgeführt.")

        st.divider()
        sim_stats    = state.get("sim_stats", {})
        total_trades = sim_stats.get("total", 0)

        if total_trades >= 5:
            if st.button("🧠 Jetzt Lernanalyse starten", type="primary"):
                _LEARNING_TRIGGER.parent.mkdir(exist_ok=True)
                _LEARNING_TRIGGER.write_text("1")
                st.success("Analyse gestartet… Dashboard in 30 Sekunden aktualisieren.")
        else:
            st.info(f"Mindestens 5 Trades benötigt. Aktuell: {total_trades}")
            st.caption(
                "Trades werden automatisch simuliert sobald Signale > 65% Konfidenz auftreten."
            )

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
