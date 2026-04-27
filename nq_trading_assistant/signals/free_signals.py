"""
Signal generators for free (non-L2) market data.

Consumes FreeDataClient.get_snapshot() and emits Signal objects
that are compatible with the existing SignalEngine infrastructure.

Signals produced:
  MULTI_TF_BIAS    — EMA9/21 alignment on 5m + 15m, price vs VWAP
  FAIR_VALUE_GAP   — 3-candle gap on 5m chart (reuse technical.py logic)
  VIX_REGIME       — volatility regime filter (blocks at VIX > 30)
  OVERNIGHT_GAP    — gap-fill directional bias in first 90 min
  CALENDAR_FILTER  — blocks signal when high-impact event within 30 min
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from signals.order_flow import Direction, Signal, SignalType  # reuse types

logger = logging.getLogger(__name__)

# String constants for free signal types
MULTI_TF_BIAS    = "MULTI_TF_BIAS"
VIX_REGIME       = "VIX_REGIME"
OVERNIGHT_GAP    = "OVERNIGHT_GAP"
CALENDAR_FILTER  = "CALENDAR_FILTER"

# Thresholds
_VIX_LOW      = 15.0
_VIX_HIGH     = 25.0
_VIX_EXTREME  = 30.0
_GAP_MIN_PCT  = 0.30     # % gap to be considered meaningful
_GAP_WINDOW   = 90       # minutes into session gap-fill is still valid
_EMA_SHORT    = 9
_EMA_LONG     = 21


# ---------------------------------------------------------------------------
# Multi-timeframe signal analyzer
# ---------------------------------------------------------------------------

class FreeMarketAnalyzer:
    """
    Stateless analyzer that turns a FreeDataClient snapshot into
    directional Signal objects.

    Call analyze(free_snap) on every poll cycle.
    Returns a list of Signals (may be empty).
    A CALENDAR_FILTER or extreme VIX_REGIME signal signals a BLOCK —
    SignalEngine.evaluate_multi_tf() checks for these and returns None.
    """

    def analyze(self, free_snap: dict) -> List[Signal]:
        signals: List[Signal] = []

        # Hard blocks first — if any block signal is emitted, the engine
        # should return None regardless of other signals.
        signals += self._vix_regime(free_snap)
        signals += self._calendar_filter(free_snap)

        # If a block is present, stop here so we don't waste computation
        if any(s.metadata.get("block") for s in signals):
            return signals

        signals += self._multi_tf_bias(free_snap)
        signals += self._overnight_gap(free_snap)
        signals += self._fvg_5m(free_snap)

        return signals

    # ------------------------------------------------------------------
    # MULTI_TF_BIAS
    # ------------------------------------------------------------------

    def _multi_tf_bias(self, snap: dict) -> List[Signal]:
        bars_5m  = snap.get("bars_5m",  [])
        bars_15m = snap.get("bars_15m", [])
        vwap     = snap.get("session_vwap")
        price    = snap.get("last_price", 0.0)

        if len(bars_5m) < _EMA_LONG + 2 or len(bars_15m) < _EMA_LONG + 2:
            return []

        ema9_5m,  ema21_5m  = _compute_emas(bars_5m)
        ema9_15m, ema21_15m = _compute_emas(bars_15m)

        if ema9_5m is None or ema9_15m is None:
            return []

        # Count aligned conditions
        bull_conditions = 0
        bear_conditions = 0

        # 5m EMA alignment
        if ema9_5m > ema21_5m:
            bull_conditions += 1
        elif ema9_5m < ema21_5m:
            bear_conditions += 1

        # 15m EMA alignment
        if ema9_15m > ema21_15m:
            bull_conditions += 1
        elif ema9_15m < ema21_15m:
            bear_conditions += 1

        # Price vs VWAP
        if vwap and price:
            if price > vwap:
                bull_conditions += 1
            elif price < vwap:
                bear_conditions += 1

        max_bull = bull_conditions
        max_bear = bear_conditions

        if max_bull >= max_bear and max_bull >= 2:
            direction  = Direction.BULLISH
            score      = max_bull
        elif max_bear > max_bull and max_bear >= 2:
            direction  = Direction.BEARISH
            score      = max_bear
        else:
            return []   # conflicting or insufficient alignment

        confidence = 0.85 if score == 3 else 0.60

        sep_5m  = abs(ema9_5m  - ema21_5m)
        sep_15m = abs(ema9_15m - ema21_15m)

        return [_make_signal(
            MULTI_TF_BIAS, direction, confidence,
            f"{'Bull' if direction == Direction.BULLISH else 'Bear'} multi-TF bias: "
            f"{score}/3 conditions aligned "
            f"(5m EMAs Δ={sep_5m:.2f}, 15m EMAs Δ={sep_15m:.2f}, VWAP={vwap:.2f})",
            {
                "ema9_5m":    round(ema9_5m,  2),
                "ema21_5m":   round(ema21_5m, 2),
                "ema9_15m":   round(ema9_15m,  2),
                "ema21_15m":  round(ema21_15m, 2),
                "vwap":       round(vwap, 2) if vwap else None,
                "conditions": score,
            },
        )]

    # ------------------------------------------------------------------
    # VIX_REGIME
    # ------------------------------------------------------------------

    def _vix_regime(self, snap: dict) -> List[Signal]:
        vix = snap.get("vix", 0.0)
        if vix <= 0:
            return []

        regime = snap.get("vix_regime", "unknown")

        # Extreme VIX → block all signals
        if vix > _VIX_EXTREME:
            return [_make_signal(
                VIX_REGIME, Direction.NEUTRAL, 1.0,
                f"VIX={vix:.1f} EXTREME (>{_VIX_EXTREME}) — all signals blocked",
                {"vix": vix, "regime": regime, "block": True},
            )]

        # High VIX → warning, let engine reduce confidence
        if vix > _VIX_HIGH:
            return [_make_signal(
                VIX_REGIME, Direction.NEUTRAL, 0.0,
                f"VIX={vix:.1f} HIGH ({_VIX_HIGH}–{_VIX_EXTREME}) — reduced position sizing",
                {"vix": vix, "regime": regime, "reduce_size": True, "block": False},
            )]

        # Normal or low — no signal, just metadata available via snap
        return []

    # ------------------------------------------------------------------
    # OVERNIGHT_GAP
    # ------------------------------------------------------------------

    def _overnight_gap(self, snap: dict) -> List[Signal]:
        gap_pct = snap.get("overnight_gap_pct", 0.0)
        gap_dir = snap.get("overnight_gap_dir", "none")

        if abs(gap_pct) < _GAP_MIN_PCT:
            return []

        # Only valid in first 90 minutes of regular session
        now_et = _minutes_since_rth_open()
        if now_et is None or now_et > _GAP_WINDOW:
            return []

        # Gap-fill direction: gap UP → expect fill DOWN → SHORT
        #                     gap DOWN → expect fill UP → LONG
        direction = Direction.BULLISH if gap_dir == "down" else Direction.BEARISH
        confidence = min(0.55 + abs(gap_pct) / 2.0 * 0.30, 0.80)
        fill_target = snap.get("yesterday_close", 0.0)

        return [_make_signal(
            OVERNIGHT_GAP, direction, confidence,
            f"Overnight gap {'up' if gap_dir == 'up' else 'down'} {gap_pct:+.2f}% — "
            f"gap-fill target {fill_target:.2f} ({now_et:.0f} min into session)",
            {
                "gap_pct":    round(gap_pct, 3),
                "gap_dir":    gap_dir,
                "fill_target": fill_target,
                "mins_into_session": round(now_et, 1),
            },
        )]

    # ------------------------------------------------------------------
    # CALENDAR_FILTER
    # ------------------------------------------------------------------

    def _calendar_filter(self, snap: dict) -> List[Signal]:
        if not snap.get("event_window_active", False):
            return []

        mins = snap.get("minutes_to_next_event")
        upcoming = [e for e in snap.get("upcoming_events", [])
                    if e.get("impact") == "high"]
        names = ", ".join(e.get("name", "?") for e in upcoming[:3])

        desc = (
            f"High-impact event in {mins:.0f} min: {names}"
            if mins and mins > 0
            else f"High-impact event active (post-event wait): {names}"
        )

        return [_make_signal(
            CALENDAR_FILTER, Direction.NEUTRAL, 1.0, desc,
            {"block": True, "minutes_to_event": mins, "events": names},
        )]

    # ------------------------------------------------------------------
    # FAIR_VALUE_GAP on 5m chart
    # ------------------------------------------------------------------

    def _fvg_5m(self, snap: dict) -> List[Signal]:
        bars = snap.get("bars_5m", [])
        price = snap.get("last_price", 0.0)

        if len(bars) < 3 or not price:
            return []

        recent = bars[-30:]
        n = len(recent)

        for i in range(2, n):
            c0 = recent[i - 2]
            c2 = recent[i]

            # Bullish FVG: c0.low > c2.high
            if c0["low"] > c2["high"]:
                gap_low  = float(c2["high"])
                gap_high = float(c0["low"])
                if gap_low <= price <= gap_high:
                    gap_size = gap_high - gap_low
                    conf = min(0.60 + gap_size / 15.0 * 0.25, 0.85)
                    return [_make_signal(
                        "FAIR_VALUE_GAP", Direction.BULLISH, conf,
                        f"Bullish 5m FVG retest: {gap_low:.2f}–{gap_high:.2f} "
                        f"(gap={gap_size:.2f} pts)",
                        {"gap_low": gap_low, "gap_high": gap_high,
                         "gap_size": round(gap_size, 2), "timeframe": "5m"},
                    )]

            # Bearish FVG: c0.high < c2.low
            elif c0["high"] < c2["low"]:
                gap_low  = float(c0["high"])
                gap_high = float(c2["low"])
                if gap_low <= price <= gap_high:
                    gap_size = gap_high - gap_low
                    conf = min(0.60 + gap_size / 15.0 * 0.25, 0.85)
                    return [_make_signal(
                        "FAIR_VALUE_GAP", Direction.BEARISH, conf,
                        f"Bearish 5m FVG retest: {gap_low:.2f}–{gap_high:.2f} "
                        f"(gap={gap_size:.2f} pts)",
                        {"gap_low": gap_low, "gap_high": gap_high,
                         "gap_size": round(gap_size, 2), "timeframe": "5m"},
                    )]

        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_emas(bars: list) -> tuple:
    """Return (ema9_last, ema21_last) or (None, None) if insufficient data."""
    if len(bars) < _EMA_LONG + 2:
        return None, None
    closes = pd.Series([b["close"] for b in bars], dtype=float)
    ema9  = closes.ewm(span=_EMA_SHORT, adjust=False).mean()
    ema21 = closes.ewm(span=_EMA_LONG,  adjust=False).mean()
    return float(ema9.iloc[-1]), float(ema21.iloc[-1])


def _minutes_since_rth_open() -> Optional[float]:
    """
    Returns minutes elapsed since 09:30 ET today, or None if market not open.
    Approximate: does not account for holidays.
    """
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        rth_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        rth_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

        if now_et < rth_open or now_et > rth_close:
            return None  # outside RTH

        return (now_et - rth_open).total_seconds() / 60
    except Exception:
        return None


def _make_signal(
    signal_type: str,
    direction: Direction,
    confidence: float,
    description: str,
    metadata: Optional[dict] = None,
) -> Signal:
    s = Signal.__new__(Signal)
    object.__setattr__(s, "signal_type",  signal_type)
    object.__setattr__(s, "direction",    direction)
    object.__setattr__(s, "confidence",   round(confidence, 3))
    object.__setattr__(s, "description",  description)
    object.__setattr__(s, "metadata",     metadata or {})
    return s
