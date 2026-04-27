"""
Signal aggregation engine for NQ Futures Day Trading.

Combines order-flow and technical signals into a single actionable
trade recommendation with entry zone, stop-loss, and targets.

Usage:
    engine = SignalEngine()
    result = engine.evaluate(book_snap, data_snap)
    # result is a TradeRecommendation or None (confidence too low)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from signals.order_flow import Direction, OrderFlowAnalyzer, Signal, SignalType
from signals.technical import (
    EMA_TREND,
    FAIR_VALUE_GAP,
    RSI_EXTREME,
    SESSION_LEVELS,
    VWAP_POSITION,
    TechnicalAnalyzer,
    compute_atr,
)

logger = logging.getLogger(__name__)

# Per-signal-type weights (sum need not equal 1 — we normalise)
_WEIGHTS: Dict[str, float] = {
    # Order flow — higher weight (direct market microstructure)
    SignalType.STACKED_IMBALANCES:     1.6,
    SignalType.LARGE_ORDER_ABSORPTION: 1.5,
    SignalType.BID_ASK_IMBALANCE:      1.2,
    SignalType.DELTA_DIVERGENCE:       1.3,
    # Technical — supporting context
    FAIR_VALUE_GAP:   1.1,
    VWAP_POSITION:    0.9,
    EMA_TREND:        0.8,
    RSI_EXTREME:      1.0,
    SESSION_LEVELS:   0.7,
}

_MIN_CONFIDENCE      = 0.65   # below this → no trade
_MIN_SIGNALS         = 2      # need at least N agreeing signals
_RISK_REWARD_T1      = 1.5    # Target 1 = 1.5 × risk
_RISK_REWARD_T2      = 2.5    # Target 2 = 2.5 × risk
_FALLBACK_ATR        = 8.0    # NQ points — used when ATR unavailable
_LIQUIDITY_CLUSTER_N = 5      # top N price levels for cluster entry zone


@dataclass
class TradeRecommendation:
    direction:  str               # "LONG" | "SHORT" | "NEUTRAL"
    confidence: float             # 0.0 – 1.0
    signals:    List[dict]        # contributing signals
    entry_zone: Dict[str, float]  # {"low": ..., "high": ...}
    stop_loss:  float
    target_1:   float
    target_2:   float
    atr:        Optional[float]
    reasoning:  str
    raw_score:  float             # weighted sum before normalisation


class SignalEngine:
    """
    Aggregates OrderFlowAnalyzer and TechnicalAnalyzer signals.

    Single public method: evaluate(book_snap, data_snap) → TradeRecommendation | None
    """

    def __init__(self) -> None:
        self._of_analyzer = OrderFlowAnalyzer()
        self._ta_analyzer  = TechnicalAnalyzer()

    def evaluate(
        self,
        book_snap: dict,
        data_snap: dict,
    ) -> Optional[TradeRecommendation]:
        # Collect raw signals
        of_signals = self._of_analyzer.analyze(book_snap, data_snap)
        ta_signals  = self._ta_analyzer.analyze(data_snap)
        all_signals = of_signals + ta_signals

        if not all_signals:
            return None

        # Weighted directional vote
        bull_score, bear_score, bull_sigs, bear_sigs = self._score_signals(all_signals)
        net_score    = bull_score - bear_score
        total_weight = bull_score + bear_score if (bull_score + bear_score) > 0 else 1.0

        direction, winning_sigs = (
            ("LONG",  bull_sigs) if net_score > 0 else
            ("SHORT", bear_sigs)
        )

        if abs(net_score) == 0:
            return None

        # Normalised confidence
        confidence = min(abs(net_score) / total_weight, 1.0)

        # Agreement check — need at least _MIN_SIGNALS in winning direction
        if len(winning_sigs) < _MIN_SIGNALS:
            return None

        if confidence < _MIN_CONFIDENCE:
            return None

        # Market price and ATR
        last_price = data_snap.get("last_price", 0.0)
        sec_bars   = data_snap.get("second_bars", [])
        min_bars   = data_snap.get("minute_bars", [])
        atr = compute_atr(min_bars) or compute_atr(sec_bars) or _FALLBACK_ATR

        # Entry zone from L2 liquidity clusters
        entry_zone = self._liquidity_entry_zone(book_snap, direction, last_price)

        # Risk levels
        entry_ref  = (entry_zone["low"] + entry_zone["high"]) / 2
        stop_loss, target_1, target_2 = self._risk_levels(
            direction, entry_ref, atr
        )

        reasoning = self._build_reasoning(
            direction, confidence, winning_sigs, entry_zone,
            stop_loss, target_1, target_2, atr, last_price
        )

        return TradeRecommendation(
            direction   = direction,
            confidence  = round(confidence, 3),
            signals     = [_signal_to_dict(s) for s in winning_sigs],
            entry_zone  = entry_zone,
            stop_loss   = stop_loss,
            target_1    = target_1,
            target_2    = target_2,
            atr         = round(atr, 2) if atr else None,
            reasoning   = reasoning,
            raw_score   = round(abs(net_score), 3),
        )

    def to_dict(self, rec: Optional[TradeRecommendation]) -> dict:
        if rec is None:
            return {
                "direction":  "NEUTRAL",
                "confidence": 0.0,
                "signals":    [],
                "entry_zone": {},
                "stop_loss":  None,
                "target_1":   None,
                "target_2":   None,
                "atr":        None,
                "reasoning":  "No high-confidence signal.",
                "raw_score":  0.0,
            }
        return {
            "direction":  rec.direction,
            "confidence": rec.confidence,
            "signals":    rec.signals,
            "entry_zone": rec.entry_zone,
            "stop_loss":  rec.stop_loss,
            "target_1":   rec.target_1,
            "target_2":   rec.target_2,
            "atr":        rec.atr,
            "reasoning":  rec.reasoning,
            "raw_score":  rec.raw_score,
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_signals(self, signals: List[Signal]):
        bull_score = 0.0
        bear_score = 0.0
        bull_sigs  = []
        bear_sigs  = []

        for sig in signals:
            sig_type = (
                sig.signal_type.value
                if hasattr(sig.signal_type, "value")
                else str(sig.signal_type)
            )
            weight = _WEIGHTS.get(sig_type, 1.0)
            weighted = sig.confidence * weight

            if sig.direction in (Direction.BULLISH, "BULLISH"):
                bull_score += weighted
                bull_sigs.append(sig)
            elif sig.direction in (Direction.BEARISH, "BEARISH"):
                bear_score += weighted
                bear_sigs.append(sig)

        return bull_score, bear_score, bull_sigs, bear_sigs

    # ------------------------------------------------------------------
    # Entry zone from L2 liquidity clusters
    # ------------------------------------------------------------------

    def _liquidity_entry_zone(
        self, book_snap: dict, direction: str, last_price: float
    ) -> Dict[str, float]:
        """
        For LONG:  find the densest bid cluster just below current price.
        For SHORT: find the densest ask cluster just above current price.
        Returns {low, high} representing the entry zone.
        """
        if direction == "LONG":
            ladder = book_snap.get("bid_ladder", [])
            # Filter levels at or below last_price
            candidates = [
                lv for lv in ladder if lv["price"] <= last_price + 1.0
            ]
        else:
            ladder = book_snap.get("ask_ladder", [])
            candidates = [
                lv for lv in ladder if lv["price"] >= last_price - 1.0
            ]

        if not candidates:
            # Fallback: ±0.5 NQ point around last price
            return {
                "low":  round(last_price - 0.5, 2),
                "high": round(last_price + 0.5, 2),
            }

        # Sort by size descending, take top N
        top = sorted(candidates, key=lambda x: x["size"], reverse=True)[
            :_LIQUIDITY_CLUSTER_N
        ]
        prices = [lv["price"] for lv in top]

        return {
            "low":  round(min(prices), 2),
            "high": round(max(prices), 2),
        }

    # ------------------------------------------------------------------
    # Risk / reward levels
    # ------------------------------------------------------------------

    def _risk_levels(
        self, direction: str, entry: float, atr: float
    ):
        risk = atr  # 1 ATR = risk unit

        if direction == "LONG":
            stop    = round(entry - risk, 2)
            target1 = round(entry + risk * _RISK_REWARD_T1, 2)
            target2 = round(entry + risk * _RISK_REWARD_T2, 2)
        else:
            stop    = round(entry + risk, 2)
            target1 = round(entry - risk * _RISK_REWARD_T1, 2)
            target2 = round(entry - risk * _RISK_REWARD_T2, 2)

        return stop, target1, target2

    # ------------------------------------------------------------------
    # Reasoning string
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reasoning(
        direction: str,
        confidence: float,
        signals: List[Signal],
        entry_zone: dict,
        stop_loss: float,
        target_1: float,
        target_2: float,
        atr: float,
        last_price: float,
    ) -> str:
        sig_names = ", ".join(
            str(s.signal_type.value if hasattr(s.signal_type, "value") else s.signal_type)
            for s in signals
        )
        risk_pts    = abs(((entry_zone["low"] + entry_zone["high"]) / 2) - stop_loss)
        reward1_pts = abs(target_1 - ((entry_zone["low"] + entry_zone["high"]) / 2))

        return (
            f"{direction} at {last_price:.2f} | confidence={confidence:.0%} | "
            f"entry {entry_zone['low']:.2f}–{entry_zone['high']:.2f} | "
            f"SL={stop_loss:.2f} (−{risk_pts:.1f} pts, 1 ATR={atr:.1f}) | "
            f"T1={target_1:.2f} (+{reward1_pts:.1f} pts) | "
            f"T2={target_2:.2f} | "
            f"Signals: [{sig_names}]"
        )


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _signal_to_dict(sig: Signal) -> dict:
    return {
        "type":        str(sig.signal_type.value if hasattr(sig.signal_type, "value")
                           else sig.signal_type),
        "direction":   str(sig.direction.value   if hasattr(sig.direction,   "value")
                           else sig.direction),
        "confidence":  sig.confidence,
        "description": sig.description,
        "metadata":    sig.metadata,
    }
