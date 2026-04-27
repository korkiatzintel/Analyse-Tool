"""
Rolling data buffer for NQ Futures tick and bar data.

Maintains:
  - Last 10 000 ticks
  - Last 500 1-second bars (built on-the-fly from ticks)
  - Cumulative Delta per bar
  - Footprint chart data per bar
  - Session VWAP, daily high / low

Input tick format (from rithmic_client._normalise_tick):
    {
        "type":      "tick",
        "symbol":    str,
        "price":     float,
        "volume":    int,
        "side":      str,   # "B" (buy / ask-lift) | "S" (sell / bid-hit) | ""
        "timestamp": int,   # ssboe
    }

Input bar format (from rithmic_client._normalise_bar):
    {
        "type":      "time_bar",
        "symbol":    str,
        "open":      float,
        "high":      float,
        "low":       float,
        "close":     float,
        "volume":    int,
        "timestamp": int,   # bar close ssboe
    }
"""

import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_TICK_MAXLEN = 10_000
_BAR_MAXLEN  = 500
_NQ_TICK     = 0.25      # minimum price increment for NQ


class FootprintBar:
    """
    Accumulates bid/ask volume at each price level within a 1-second window.

    footprint: {price: {"bid_vol": int, "ask_vol": int, "delta": int}}
    """

    __slots__ = (
        "open", "high", "low", "close",
        "volume", "bid_volume", "ask_volume",
        "timestamp", "footprint",
    )

    def __init__(self, first_tick: dict) -> None:
        price = first_tick["price"]
        self.open      = price
        self.high      = price
        self.low       = price
        self.close     = price
        self.volume    = 0
        self.bid_volume = 0
        self.ask_volume = 0
        self.timestamp = first_tick["timestamp"]
        self.footprint: Dict[float, Dict[str, int]] = defaultdict(
            lambda: {"bid_vol": 0, "ask_vol": 0, "delta": 0}
        )
        self._add_tick(first_tick)

    def _add_tick(self, tick: dict) -> None:
        price  = tick["price"]
        volume = tick.get("volume", 0)
        side   = str(tick.get("side", "")).upper()

        self.high  = max(self.high, price)
        self.low   = min(self.low,  price)
        self.close = price
        self.volume += volume

        if side in ("B", "BUY", "ASK"):    # aggressor buy = lift the ask
            self.ask_volume += volume
            self.footprint[price]["ask_vol"] += volume
        elif side in ("S", "SELL", "BID"): # aggressor sell = hit the bid
            self.bid_volume += volume
            self.footprint[price]["bid_vol"] += volume
        else:
            # Unknown side — split evenly (conservative)
            half = volume // 2
            self.ask_volume += half
            self.bid_volume += (volume - half)
            self.footprint[price]["ask_vol"] += half
            self.footprint[price]["bid_vol"] += (volume - half)

        self.footprint[price]["delta"] = (
            self.footprint[price]["ask_vol"] - self.footprint[price]["bid_vol"]
        )

    @property
    def cumulative_delta(self) -> int:
        return self.ask_volume - self.bid_volume

    def to_dict(self) -> dict:
        return {
            "timestamp":         self.timestamp,
            "open":              self.open,
            "high":              self.high,
            "low":               self.low,
            "close":             self.close,
            "volume":            self.volume,
            "bid_volume":        self.bid_volume,
            "ask_volume":        self.ask_volume,
            "cumulative_delta":  self.cumulative_delta,
            "footprint": {
                price: dict(levels)
                for price, levels in sorted(self.footprint.items(), reverse=True)
            },
        }

    def update_tick(self, tick: dict) -> None:
        self._add_tick(tick)


class DataBuffer:
    """
    Central rolling buffer for tick, 1s-bar, and session data.

    Thread-safe via an internal Lock so the Streamlit UI thread
    can call get_analysis_snapshot() concurrently with the async feed.

    Usage:
        buf = DataBuffer()
        # Wire into RithmicConnectionManager callbacks:
        buf.on_tick(tick_dict)
        buf.on_time_bar(bar_dict)          # optional — for exchange bars
        snapshot = buf.get_analysis_snapshot()
    """

    def __init__(self) -> None:
        self._lock = Lock()

        # Rolling tick deque
        self._ticks: deque = deque(maxlen=_TICK_MAXLEN)

        # Rolling 1-minute bars received from exchange (normalised dicts)
        self._minute_bars: deque = deque(maxlen=_BAR_MAXLEN)

        # 1-second footprint bars built from ticks
        self._second_bars: deque = deque(maxlen=_BAR_MAXLEN)
        self._current_second_bar: Optional[FootprintBar] = None
        self._current_second: int = -1   # ssboe truncated to second

        # Session state (reset at session start / midnight)
        self._session_start_ssboe: int = 0
        self._session_high:  float = float("-inf")
        self._session_low:   float = float("inf")
        self._vwap_sum_pv:   float = 0.0   # sum(price * volume)
        self._vwap_sum_v:    int   = 0      # sum(volume)

        self._session_initialized = False

    # ------------------------------------------------------------------
    # Public feed methods — call from async callbacks (use asyncio.run_coroutine_threadsafe
    # or call synchronously inside a sync wrapper)
    # ------------------------------------------------------------------

    def on_tick(self, tick: dict) -> None:
        """Process a normalised tick dict."""
        with self._lock:
            self._ticks.append(tick)
            self._update_session(tick)
            self._update_second_bar(tick)

    def on_time_bar(self, bar: dict) -> None:
        """Process a normalised 1-minute bar dict from Rithmic."""
        with self._lock:
            self._minute_bars.append(bar)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """Call at start of RTH (9:30 ET) or CME Globex open."""
        with self._lock:
            self._session_start_ssboe = 0
            self._session_high  = float("-inf")
            self._session_low   = float("inf")
            self._vwap_sum_pv   = 0.0
            self._vwap_sum_v    = 0
            self._second_bars.clear()
            self._minute_bars.clear()
            self._current_second_bar = None
            self._current_second = -1
            self._session_initialized = False
        logger.info("DataBuffer: session reset.")

    # ------------------------------------------------------------------
    # Snapshot for Signal Engine and UI
    # ------------------------------------------------------------------

    def get_analysis_snapshot(self) -> dict:
        with self._lock:
            ticks       = list(self._ticks)
            sec_bars    = [b.to_dict() for b in self._second_bars]
            min_bars    = list(self._minute_bars)

            # Flush current incomplete second bar into snapshot (read-only copy)
            if self._current_second_bar:
                sec_bars.append(self._current_second_bar.to_dict())

            last_tick = ticks[-1] if ticks else {}
            last_price = last_tick.get("price", 0.0)

            return {
                # --- Tick summary ---
                "tick_count":        len(ticks),
                "last_price":        last_price,
                "last_volume":       last_tick.get("volume", 0),
                "last_side":         last_tick.get("side", ""),
                "last_timestamp":    last_tick.get("timestamp", 0),

                # --- Session metrics ---
                "session_high":      self._session_high if self._session_initialized else None,
                "session_low":       self._session_low  if self._session_initialized else None,
                "vwap":              self._vwap(),

                # --- 1-second footprint bars ---
                "second_bars":       sec_bars,
                "second_bar_count":  len(sec_bars),

                # --- 1-minute bars ---
                "minute_bars":       min_bars,
                "minute_bar_count":  len(min_bars),

                # --- Cumulative delta ---
                "cumulative_delta_session": self._session_cumulative_delta(sec_bars),
                "cumulative_delta_last10":  self._last_n_delta(sec_bars, 10),
                "cumulative_delta_last1":   self._last_n_delta(sec_bars, 1),

                # --- Recent tick-level statistics ---
                "recent_tick_stats":  self._recent_tick_stats(ticks, n=500),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_session(self, tick: dict) -> None:
        price  = tick["price"]
        volume = tick.get("volume", 0)
        ts     = tick.get("timestamp", 0)

        if not self._session_initialized:
            self._session_start_ssboe = ts
            self._session_high = price
            self._session_low  = price
            self._session_initialized = True

        self._session_high = max(self._session_high, price)
        self._session_low  = min(self._session_low,  price)
        self._vwap_sum_pv += price * volume
        self._vwap_sum_v  += volume

    def _update_second_bar(self, tick: dict) -> None:
        ts     = tick.get("timestamp", 0)
        second = int(ts)  # ssboe is already in seconds

        if second != self._current_second:
            # Seal the previous bar
            if self._current_second_bar is not None:
                self._second_bars.append(self._current_second_bar)
            # Open a new bar
            self._current_second_bar = FootprintBar(tick)
            self._current_second     = second
        else:
            if self._current_second_bar is not None:
                self._current_second_bar.update_tick(tick)

    def _vwap(self) -> Optional[float]:
        if self._vwap_sum_v == 0:
            return None
        return round(self._vwap_sum_pv / self._vwap_sum_v, 2)

    @staticmethod
    def _session_cumulative_delta(sec_bars: List[dict]) -> int:
        return sum(b.get("cumulative_delta", 0) for b in sec_bars)

    @staticmethod
    def _last_n_delta(sec_bars: List[dict], n: int) -> int:
        return sum(b.get("cumulative_delta", 0) for b in sec_bars[-n:])

    @staticmethod
    def _recent_tick_stats(ticks: List[dict], n: int = 500) -> dict:
        recent = ticks[-n:]
        if not recent:
            return {}

        prices  = np.array([t["price"]  for t in recent], dtype=np.float64)
        volumes = np.array([t.get("volume", 0) for t in recent], dtype=np.float64)

        buy_vol  = sum(t.get("volume", 0) for t in recent
                       if str(t.get("side", "")).upper() in ("B", "BUY", "ASK"))
        sell_vol = sum(t.get("volume", 0) for t in recent
                       if str(t.get("side", "")).upper() in ("S", "SELL", "BID"))
        total_vol = buy_vol + sell_vol

        return {
            "n":            len(recent),
            "price_mean":   round(float(np.mean(prices)), 2),
            "price_std":    round(float(np.std(prices)),  2),
            "price_min":    float(np.min(prices)),
            "price_max":    float(np.max(prices)),
            "total_volume": int(np.sum(volumes)),
            "buy_volume":   buy_vol,
            "sell_volume":  sell_vol,
            "buy_ratio":    round(buy_vol / total_vol, 4) if total_vol else 0.5,
            "delta":        buy_vol - sell_vol,
        }
