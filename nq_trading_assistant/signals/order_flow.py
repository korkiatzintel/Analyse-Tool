"""
Order flow signal analysis for NQ Futures.

Consumes snapshots from OrderBook.get_snapshot() and
DataBuffer.get_analysis_snapshot() and emits typed Signal objects.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)

# Thresholds
_IMBALANCE_BULL       = 0.70
_IMBALANCE_BEAR       = 0.30
_IMBALANCE_STREAK     = 3     # consecutive updates needed
_STACKED_MIN          = 3     # consecutive footprint imbalances
_STACKED_THRESHOLD    = 0.65  # per-bar imbalance to count as "stacked"
_LARGE_ORDER_CUTOFF   = 50    # contracts
_ABSORPTION_MAX_MOVE  = 2.0   # NQ points — price "stopped" within this range
_DELTA_LOOKBACK       = 10    # bars for divergence check
_DELTA_REVERSAL_MIN   = 50    # minimum delta change to be meaningful


class SignalType(str, Enum):
    BID_ASK_IMBALANCE      = "BID_ASK_IMBALANCE"
    DELTA_DIVERGENCE       = "DELTA_DIVERGENCE"
    STACKED_IMBALANCES     = "STACKED_IMBALANCES"
    LARGE_ORDER_ABSORPTION = "LARGE_ORDER_ABSORPTION"


class Direction(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class Signal:
    signal_type: SignalType
    direction:   Direction
    confidence:  float          # 0.0 – 1.0
    description: str
    metadata:    dict = field(default_factory=dict)


class OrderFlowAnalyzer:
    """
    Stateful order-flow signal detector.

    Call analyze(book_snap, data_snap) on every update cycle.
    Returns a list of Signal objects (may be empty).

    The analyzer maintains short internal histories so streak /
    divergence logic works across multiple calls.
    """

    def __init__(self) -> None:
        # Rolling history of imbalance_ratio values for streak detection
        self._imbalance_history: deque = deque(maxlen=20)

        # Price history parallel to second_bars for divergence
        self._price_history:  deque = deque(maxlen=_DELTA_LOOKBACK + 5)
        self._delta_history:  deque = deque(maxlen=_DELTA_LOOKBACK + 5)

        # Track large-order levels we have already seen so we don't re-fire
        self._known_large_order_keys: set = set()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self, book_snap: dict, data_snap: dict) -> List[Signal]:
        signals: List[Signal] = []

        imbalance  = book_snap.get("imbalance_ratio", 0.5)
        last_price = data_snap.get("last_price", 0.0)
        sec_bars   = data_snap.get("second_bars", [])

        # Update rolling histories
        self._imbalance_history.append(imbalance)
        if last_price:
            self._price_history.append(last_price)
        if sec_bars:
            self._delta_history.append(
                data_snap.get("cumulative_delta_last1", 0)
            )

        signals += self._bid_ask_imbalance(book_snap)
        signals += self._delta_divergence(data_snap)
        signals += self._stacked_imbalances(sec_bars)
        signals += self._large_order_absorption(book_snap, last_price)

        return signals

    # ------------------------------------------------------------------
    # Signal 1 — BID_ASK_IMBALANCE
    # ------------------------------------------------------------------

    def _bid_ask_imbalance(self, book_snap: dict) -> List[Signal]:
        hist = list(self._imbalance_history)
        if len(hist) < _IMBALANCE_STREAK:
            return []

        recent = hist[-_IMBALANCE_STREAK:]
        current = hist[-1]

        bull_streak = all(v > _IMBALANCE_BULL for v in recent)
        bear_streak = all(v < _IMBALANCE_BEAR for v in recent)

        if not (bull_streak or bear_streak):
            return []

        direction = Direction.BULLISH if bull_streak else Direction.BEARISH

        # Confidence scales with how extreme and how long the streak is
        streak_len = self._streak_length(
            hist,
            lambda v: v > _IMBALANCE_BULL if bull_streak else v < _IMBALANCE_BEAR,
        )
        extreme    = abs(current - 0.5) * 2   # 0 → 1 as ratio moves away from 0.5
        confidence = min(0.40 + extreme * 0.35 + min(streak_len / 10, 0.25), 1.0)

        bid_d  = book_snap.get("bid_depth_10", 0)
        ask_d  = book_snap.get("ask_depth_10", 0)

        return [Signal(
            signal_type = SignalType.BID_ASK_IMBALANCE,
            direction   = direction,
            confidence  = round(confidence, 3),
            description = (
                f"{'Bull' if bull_streak else 'Bear'} imbalance: ratio={current:.2f} "
                f"for {streak_len} consecutive updates "
                f"(bid_depth={bid_d}, ask_depth={ask_d})"
            ),
            metadata = {
                "imbalance_ratio": current,
                "streak_length":   streak_len,
                "bid_depth_10":    bid_d,
                "ask_depth_10":    ask_d,
            },
        )]

    # ------------------------------------------------------------------
    # Signal 2 — DELTA_DIVERGENCE
    # ------------------------------------------------------------------

    def _delta_divergence(self, data_snap: dict) -> List[Signal]:
        sec_bars = data_snap.get("second_bars", [])
        if len(sec_bars) < _DELTA_LOOKBACK:
            return []

        bars    = sec_bars[-_DELTA_LOOKBACK:]
        prices  = [b["close"] for b in bars if "close" in b]
        deltas  = [b.get("cumulative_delta", 0) for b in bars]

        if len(prices) < 4:
            return []

        price_change = prices[-1] - prices[0]
        delta_change = deltas[-1] - deltas[0]

        # Need meaningful delta move to avoid noise
        if abs(delta_change) < _DELTA_REVERSAL_MIN:
            return []

        bullish_div = price_change < -1.0 and delta_change > _DELTA_REVERSAL_MIN
        bearish_div = price_change >  1.0 and delta_change < -_DELTA_REVERSAL_MIN

        if not (bullish_div or bearish_div):
            return []

        direction = Direction.BULLISH if bullish_div else Direction.BEARISH

        # Confidence from magnitude of divergence
        price_norm = min(abs(price_change) / 10.0, 1.0)
        delta_norm = min(abs(delta_change) / 500.0, 1.0)
        confidence = min(0.45 + (price_norm + delta_norm) * 0.25, 1.0)

        return [Signal(
            signal_type = SignalType.DELTA_DIVERGENCE,
            direction   = direction,
            confidence  = round(confidence, 3),
            description = (
                f"{'Bullish absorption' if bullish_div else 'Bearish distribution'}: "
                f"price {price_change:+.2f} pts, delta {delta_change:+d} "
                f"over last {_DELTA_LOOKBACK} bars"
            ),
            metadata = {
                "price_change": round(price_change, 2),
                "delta_change": delta_change,
                "lookback_bars": _DELTA_LOOKBACK,
            },
        )]

    # ------------------------------------------------------------------
    # Signal 3 — STACKED_IMBALANCES
    # ------------------------------------------------------------------

    def _stacked_imbalances(self, sec_bars: List[dict]) -> List[Signal]:
        """
        Detect N consecutive 1s-bars where each bar's buy_ratio > threshold
        (bull stack) or < 1-threshold (bear stack).
        """
        if len(sec_bars) < _STACKED_MIN:
            return []

        recent = sec_bars[-(_STACKED_MIN + 5):]

        bull_streak = self._footprint_streak(recent, bullish=True)
        bear_streak = self._footprint_streak(recent, bullish=False)

        signals = []
        for streak, direction in [
            (bull_streak, Direction.BULLISH),
            (bear_streak, Direction.BEARISH),
        ]:
            if streak < _STACKED_MIN:
                continue

            # High base confidence for stacked imbalances
            confidence = min(0.60 + (streak - _STACKED_MIN) * 0.07, 0.95)

            signals.append(Signal(
                signal_type = SignalType.STACKED_IMBALANCES,
                direction   = direction,
                confidence  = round(confidence, 3),
                description = (
                    f"Stacked {'bid' if direction == Direction.BULLISH else 'ask'} "
                    f"imbalances: {streak} consecutive bars "
                    f"(threshold={_STACKED_THRESHOLD:.0%})"
                ),
                metadata = {
                    "streak_length": streak,
                    "direction":     direction.value,
                },
            ))

        return signals

    # ------------------------------------------------------------------
    # Signal 4 — LARGE_ORDER_ABSORPTION
    # ------------------------------------------------------------------

    def _large_order_absorption(
        self, book_snap: dict, last_price: float
    ) -> List[Signal]:
        large_orders = book_snap.get("large_orders", [])
        if not large_orders or not last_price:
            return []

        signals = []
        for order in large_orders:
            size      = order.get("size", 0)
            price     = order.get("price", 0.0)
            side      = order.get("side", "")
            timestamp = order.get("timestamp", 0)

            if size < _LARGE_ORDER_CUTOFF:
                continue

            key = (round(price, 2), side, timestamp)
            if key in self._known_large_order_keys:
                continue

            distance = abs(last_price - price)
            if distance > _ABSORPTION_MAX_MOVE:
                continue

            # Price is AT or within 2 pts of a large resting order
            self._known_large_order_keys.add(key)

            # If price touched a large bid → likely support (bullish)
            # If price touched a large ask → likely resistance (bearish)
            direction  = Direction.BULLISH if side == "bid" else Direction.BEARISH
            confidence = min(0.60 + (size - _LARGE_ORDER_CUTOFF) / 500.0, 0.92)

            signals.append(Signal(
                signal_type = SignalType.LARGE_ORDER_ABSORPTION,
                direction   = direction,
                confidence  = round(confidence, 3),
                description = (
                    f"Large {side} order ({size} contracts @ {price:.2f}) "
                    f"within {distance:.2f} pts of current price {last_price:.2f} — "
                    f"{'support' if direction == Direction.BULLISH else 'resistance'} likely"
                ),
                metadata = {
                    "order_price":    price,
                    "order_size":     size,
                    "order_side":     side,
                    "price_distance": round(distance, 2),
                },
            ))

        return signals

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _streak_length(history: list, predicate) -> int:
        count = 0
        for v in reversed(history):
            if predicate(v):
                count += 1
            else:
                break
        return count

    @staticmethod
    def _footprint_streak(bars: List[dict], bullish: bool) -> int:
        count = 0
        for bar in reversed(bars):
            bv = bar.get("bid_volume", 0)
            av = bar.get("ask_volume", 0)
            total = bv + av
            if total == 0:
                break
            buy_ratio = av / total
            if bullish and buy_ratio > _STACKED_THRESHOLD:
                count += 1
            elif not bullish and buy_ratio < (1.0 - _STACKED_THRESHOLD):
                count += 1
            else:
                break
        return count
