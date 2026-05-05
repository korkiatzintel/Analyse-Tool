"""
Technical signal analysis for NQ Futures.

Uses pandas-ta when available; falls back to native pandas/numpy
implementations of EMA, RSI, and ATR so the module works out-of-the-box.

Input: DataBuffer.get_analysis_snapshot()
Output: List[Signal]  (same Signal type as order_flow.py)
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta        # preferred
    _TA_AVAILABLE = True
except ImportError:
    ta = None                     # native fallback used below
    _TA_AVAILABLE = False

from signals.order_flow import Direction, Signal, SignalType  # reuse types

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Native TA fallbacks (pure pandas / numpy)
# ------------------------------------------------------------------

def _ema(series: pd.Series, length: int) -> pd.Series:
    if _TA_AVAILABLE:
        return ta.ema(series, length=length)
    return series.ewm(span=length, adjust=False).mean()


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    if _TA_AVAILABLE:
        return ta.rsi(series, length=length)
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=length - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=length - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    if _TA_AVAILABLE:
        return ta.atr(high, low, close, length=length)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=length - 1, adjust=False).mean()

_VWAP_WARN_POINTS    = 5.0
_SESSION_LEVEL_WARN  = 5.0
_RSI_OVERSOLD        = 30.0
_RSI_OVERBOUGHT      = 70.0
_EMA_SHORT           = 9
_EMA_LONG            = 21
_ATR_PERIOD          = 14

# Extend SignalType with technical variants via plain constants
# (avoids re-defining the Enum; signal_engine checks by string value)
VWAP_POSITION      = "VWAP_POSITION"
FAIR_VALUE_GAP     = "FAIR_VALUE_GAP"
EMA_TREND          = "EMA_TREND"
RSI_EXTREME        = "RSI_EXTREME"
SESSION_LEVELS     = "SESSION_LEVELS"


def _make_signal(
    signal_type: str,
    direction: Direction,
    confidence: float,
    description: str,
    metadata: Optional[dict] = None,
) -> Signal:
    # Wrap in the shared Signal dataclass using string signal_type
    s = Signal.__new__(Signal)
    object.__setattr__(s, "signal_type",  signal_type)
    object.__setattr__(s, "direction",    direction)
    object.__setattr__(s, "confidence",   round(confidence, 3))
    object.__setattr__(s, "description",  description)
    object.__setattr__(s, "metadata",     metadata or {})
    return s


class TechnicalAnalyzer:
    """
    Stateless technical signal generator.

    Call analyze(data_snap) on every update cycle.
    Internally builds DataFrames from minute / second bar lists.
    """

    def analyze(self, data_snap: dict, bias_direction: str = "NEUTRAL") -> List[Signal]:
        signals: List[Signal] = []

        last_price   = data_snap.get("last_price", 0.0)
        session_high = data_snap.get("session_high")
        session_low  = data_snap.get("session_low")
        vwap         = data_snap.get("vwap")
        min_bars     = data_snap.get("minute_bars", [])
        sec_bars     = data_snap.get("second_bars", [])

        if not last_price:
            return signals

        min_df = _bars_to_df(min_bars)
        sec_df = _bars_to_df(sec_bars)

        signals += self._vwap_position(last_price, vwap, bias_direction)
        signals += self._fair_value_gap(min_df, last_price)
        signals += self._ema_trend(sec_df, last_price)
        signals += self._rsi_extreme(min_df, last_price)
        signals += self._session_levels(last_price, session_high, session_low)

        return signals

    # ------------------------------------------------------------------
    # VWAP_POSITION
    # ------------------------------------------------------------------

    def _vwap_position(
        self, price: float, vwap: Optional[float], bias_direction: str = "NEUTRAL"
    ) -> List[Signal]:
        if vwap is None or vwap == 0.0:
            return []

        distance     = price - vwap
        abs_dist     = abs(distance)
        atr_approx   = _VWAP_WARN_POINTS * 2   # ~10 pts as ATR proxy
        distance_atr = abs_dist / atr_approx if atr_approx > 0 else 0

        if distance_atr < 0.5:
            # Zone 1: nahe VWAP — Trend-Signal, niedrige Konfidenz
            direction  = Direction.BULLISH if distance > 0 else Direction.BEARISH
            confidence = 0.55
            side_str   = "über" if distance > 0 else "unter"
            desc = (
                f"Preis {side_str} VWAP ({distance:+.1f} Pts) — nahe Fair Value"
            )
        elif distance_atr < 1.5:
            # Zone 2: moderat entfernt — starkes Trend-Signal
            direction  = Direction.BULLISH if distance > 0 else Direction.BEARISH
            confidence = 0.75
            side_str   = "über" if distance > 0 else "unter"
            desc = (
                f"Preis {side_str} VWAP ({distance:+.1f} Pts) — Trend bestätigt"
            )
        else:
            # Zone 3: weit entfernt — Mean-Reversion (Richtungsumkehr)
            # Preis weit unter VWAP → LONG erwarten; weit über → SHORT erwarten
            reversion_dir = Direction.BULLISH if distance < 0 else Direction.BEARISH
            rev_str       = "LONG" if distance < 0 else "SHORT"

            # Mean Reversion gegen starken Bias blockieren:
            # z.B. Preis weit über VWAP (SHORT-Reversion) aber Bias LONG → kein Trade
            if reversion_dir == Direction.BULLISH and bias_direction == "SHORT":
                logger.debug(
                    "VWAP Zone 3 LONG-Reversion blockiert: Bias ist SHORT"
                )
                return []
            if reversion_dir == Direction.BEARISH and bias_direction == "LONG":
                logger.debug(
                    "VWAP Zone 3 SHORT-Reversion blockiert: Bias ist LONG"
                )
                return []

            direction  = reversion_dir
            confidence = 0.65
            side_str   = "unter" if distance < 0 else "über"
            desc = (
                f"Preis EXTREM {side_str} VWAP ({distance:+.1f} Pts = "
                f"{distance_atr:.1f}x ATR) — "
                f"Mean Reversion {rev_str} erwartet"
            )

        return [_make_signal(
            VWAP_POSITION, direction, confidence, desc,
            {"vwap": vwap, "distance_pts": round(distance, 2),
             "distance_atr": round(distance_atr, 2)},
        )]

    # ------------------------------------------------------------------
    # FAIR_VALUE_GAP
    # ------------------------------------------------------------------

    def _fair_value_gap(self, df: pd.DataFrame, last_price: float) -> List[Signal]:
        """
        3-candle FVG pattern:
          Bullish: candle[i-2].low > candle[i].high  (gap between them)
          Bearish: candle[i-2].high < candle[i].low

        Fires a signal when current price retests inside the gap.
        Scans last 30 bars.
        """
        if df.empty or len(df) < 3:
            return []

        signals = []
        scan = df.tail(30).reset_index(drop=True)
        n = len(scan)

        for i in range(2, n):
            c0 = scan.iloc[i - 2]   # first candle
            c2 = scan.iloc[i]       # third candle

            # Bullish FVG — gap between c0.low and c2.high (c0 is the earlier one)
            if c0["low"] > c2["high"]:
                gap_low  = float(c2["high"])
                gap_high = float(c0["low"])
                if gap_low <= last_price <= gap_high:
                    gap_size = gap_high - gap_low
                    confidence = min(0.55 + gap_size / 20.0 * 0.3, 0.85)
                    signals.append(_make_signal(
                        FAIR_VALUE_GAP, Direction.BULLISH, confidence,
                        f"Bullish FVG retest: gap {gap_low:.2f}–{gap_high:.2f} "
                        f"(size={gap_size:.2f} pts)",
                        {"gap_low": gap_low, "gap_high": gap_high,
                         "gap_size": round(gap_size, 2), "type": "bullish"},
                    ))

            # Bearish FVG
            elif c0["high"] < c2["low"]:
                gap_low  = float(c0["high"])
                gap_high = float(c2["low"])
                if gap_low <= last_price <= gap_high:
                    gap_size = gap_high - gap_low
                    confidence = min(0.55 + gap_size / 20.0 * 0.3, 0.85)
                    signals.append(_make_signal(
                        FAIR_VALUE_GAP, Direction.BEARISH, confidence,
                        f"Bearish FVG retest: gap {gap_low:.2f}–{gap_high:.2f} "
                        f"(size={gap_size:.2f} pts)",
                        {"gap_low": gap_low, "gap_high": gap_high,
                         "gap_size": round(gap_size, 2), "type": "bearish"},
                    ))

        return signals[:1]  # Only most recent FVG to avoid noise

    # ------------------------------------------------------------------
    # EMA_TREND (on 30s bars built from 1s bars, or 1min bars fallback)
    # ------------------------------------------------------------------

    def _ema_trend(self, df: pd.DataFrame, last_price: float) -> List[Signal]:
        if df.empty or len(df) < _EMA_LONG + 2:
            return []

        close = df["close"]
        ema_short = _ema(close, length=_EMA_SHORT)
        ema_long  = _ema(close, length=_EMA_LONG)

        if ema_short is None or ema_long is None:
            return []

        es = ema_short.dropna()
        el = ema_long.dropna()

        if len(es) < 2 or len(el) < 2:
            return []

        prev_cross = es.iloc[-2] - el.iloc[-2]
        curr_cross = es.iloc[-1] - el.iloc[-1]

        bullish_cross = prev_cross <= 0 and curr_cross > 0
        bearish_cross = prev_cross >= 0 and curr_cross < 0

        # Also report trend continuation (no cross but aligned)
        ema9  = float(es.iloc[-1])
        ema21 = float(el.iloc[-1])
        above = ema9 > ema21

        if bullish_cross or bearish_cross:
            direction  = Direction.BULLISH if bullish_cross else Direction.BEARISH
            confidence = 0.72
            desc = (
                f"EMA{_EMA_SHORT}/EMA{_EMA_LONG} {'bullish' if bullish_cross else 'bearish'} "
                f"cross — EMA9={ema9:.2f} EMA21={ema21:.2f}"
            )
        else:
            direction  = Direction.BULLISH if above else Direction.BEARISH
            separation = abs(ema9 - ema21)
            confidence = min(0.45 + separation / 20.0 * 0.25, 0.70)
            desc = (
                f"EMA trend {'up' if above else 'down'}: "
                f"EMA9={ema9:.2f} EMA21={ema21:.2f} "
                f"(separation={separation:.2f} pts)"
            )

        return [_make_signal(
            EMA_TREND, direction, confidence, desc,
            {"ema9": round(ema9, 2), "ema21": round(ema21, 2),
             "cross": bullish_cross or bearish_cross},
        )]

    # ------------------------------------------------------------------
    # RSI_EXTREME (on 1-minute bars)
    # ------------------------------------------------------------------

    def _rsi_extreme(self, df: pd.DataFrame, last_price: float) -> List[Signal]:
        if df.empty or len(df) < 15:
            return []

        rsi_series = _rsi(df["close"], length=14)
        if rsi_series is None or rsi_series.dropna().empty:
            return []

        rsi = float(rsi_series.dropna().iloc[-1])

        if rsi > _RSI_OVERBOUGHT:
            direction  = Direction.BEARISH
            confidence = min(0.50 + (rsi - _RSI_OVERBOUGHT) / 30.0 * 0.45, 0.95)
            desc = f"RSI overbought: {rsi:.1f} (>{_RSI_OVERBOUGHT}) — exhaustion warning"
        elif rsi < _RSI_OVERSOLD:
            direction  = Direction.BULLISH
            confidence = min(0.50 + (_RSI_OVERSOLD - rsi) / 30.0 * 0.45, 0.95)
            desc = f"RSI oversold: {rsi:.1f} (<{_RSI_OVERSOLD}) — reversal watch"
        else:
            return []

        return [_make_signal(
            RSI_EXTREME, direction, confidence, desc,
            {"rsi": round(rsi, 1), "period": 14},
        )]

    # ------------------------------------------------------------------
    # SESSION_LEVELS
    # ------------------------------------------------------------------

    def _session_levels(
        self,
        price: float,
        session_high: Optional[float],
        session_low: Optional[float],
    ) -> List[Signal]:
        signals = []

        if session_high is not None:
            dist_high = abs(price - session_high)
            if dist_high < _SESSION_LEVEL_WARN:
                direction  = Direction.BEARISH  # approaching resistance
                confidence = min(0.50 + (1 - dist_high / _SESSION_LEVEL_WARN) * 0.35, 0.85)
                signals.append(_make_signal(
                    SESSION_LEVELS, direction, confidence,
                    f"Price {dist_high:.2f} pts from session high ({session_high:.2f}) — "
                    "resistance zone",
                    {"level": "session_high", "level_price": session_high,
                     "distance": round(dist_high, 2)},
                ))

        if session_low is not None:
            dist_low = abs(price - session_low)
            if dist_low < _SESSION_LEVEL_WARN:
                direction  = Direction.BULLISH  # approaching support
                confidence = min(0.50 + (1 - dist_low / _SESSION_LEVEL_WARN) * 0.35, 0.85)
                signals.append(_make_signal(
                    SESSION_LEVELS, direction, confidence,
                    f"Price {dist_low:.2f} pts from session low ({session_low:.2f}) — "
                    "support zone",
                    {"level": "session_low", "level_price": session_low,
                     "distance": round(dist_low, 2)},
                ))

        return signals


# ------------------------------------------------------------------
# Helper — build a DataFrame from a list of bar dicts
# ------------------------------------------------------------------

def _bars_to_df(bars: List[dict]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = 0.0
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def compute_atr(bars: List[dict], period: int = _ATR_PERIOD) -> Optional[float]:
    """Compute ATR from a list of bar dicts. Works with or without pandas-ta."""
    if len(bars) < period + 1:
        return None
    df = _bars_to_df(bars)
    atr_series = _atr(df["high"], df["low"], df["close"], length=period)
    if atr_series is None or atr_series.dropna().empty:
        return None
    return float(atr_series.dropna().iloc[-1])
