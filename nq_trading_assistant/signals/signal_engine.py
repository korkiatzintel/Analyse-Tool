"""
Signal aggregation engine for NQ Futures Day Trading.

Combines order-flow and technical signals into a single actionable
trade recommendation with entry zone, stop-loss, and targets.

Usage:
    engine = SignalEngine()
    result = engine.evaluate(book_snap, data_snap)
    # result is a TradeRecommendation or None (confidence too low)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
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
from signals.free_signals import (
    MULTI_TF_BIAS,
    VIX_REGIME,
    OVERNIGHT_GAP,
    CALENDAR_FILTER,
    FreeMarketAnalyzer,
    _VIX_HIGH,
)
from signals.ict_engine import ICTSignalEngine

logger = logging.getLogger(__name__)

_WEIGHTS_FILE = Path(__file__).parent.parent / "logs" / "signal_weights.json"

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
_MIN_SIGNALS         = 2      # need at least N agreeing signals (L2 mode)
_MIN_SIGNALS_FREE    = 3      # confluence: need 3 for free-data mode
_RISK_REWARD_T1      = 1.5    # Target 1 = 1.5 × risk
_RISK_REWARD_T2      = 2.5    # Target 2 = 2.5 × risk
_FALLBACK_ATR        = 8.0    # NQ points — used when ATR unavailable
_LIQUIDITY_CLUSTER_N = 5      # top N price levels for cluster entry zone
_MIN_TP1_TICKS       = 80     # minimum TP1 ticks (80 × 0.25 = 20 pts = $400 NQ)
_MIN_TP1_POINTS      = _MIN_TP1_TICKS * 0.25


class TradeSetup:
    """NQ/MNQ contract specs + precise entry/SL/TP calculation in ticks and dollars."""

    TICK_SIZE      = 0.25
    NQ_TICK_VALUE  = 5.00
    MNQ_TICK_VALUE = 0.50

    def calculate(
        self,
        direction: str,
        entry_price: float,
        atr: float,
        vwap: Optional[float] = None,
        session_high: Optional[float] = None,
        session_low: Optional[float] = None,
    ) -> dict:
        tick = self.TICK_SIZE

        # Snap entry to nearest tick
        entry = round(round(entry_price / tick) * tick, 2)

        # SL distance: 0.5 × ATR, minimum 8 ticks (2 points)
        sl_distance = round(round(max(atr * 0.5, 2.0) / tick) * tick, 2)

        if direction == "LONG":
            stop_loss     = round(entry - sl_distance, 2)
            take_profit_1 = round(entry + sl_distance * 1.5, 2)
            take_profit_2 = round(entry + sl_distance * 2.5, 2)
        else:
            stop_loss     = round(entry + sl_distance, 2)
            take_profit_1 = round(entry - sl_distance * 1.5, 2)
            take_profit_2 = round(entry - sl_distance * 2.5, 2)

        tp1_ticks = round(abs(take_profit_1 - entry) / tick)

        # Ensure TP1 >= minimum ticks; expand SL symmetrically if needed
        tp_adjusted = False
        if tp1_ticks < _MIN_TP1_TICKS:
            min_sl = round(round(max(_MIN_TP1_POINTS / _RISK_REWARD_T1,
                                     atr * 0.5) / tick) * tick, 2)
            sl_distance = min_sl
            if direction == "LONG":
                stop_loss     = round(entry - sl_distance, 2)
                take_profit_1 = round(entry + sl_distance * _RISK_REWARD_T1, 2)
                take_profit_2 = round(entry + sl_distance * _RISK_REWARD_T2, 2)
            else:
                stop_loss     = round(entry + sl_distance, 2)
                take_profit_1 = round(entry - sl_distance * _RISK_REWARD_T1, 2)
                take_profit_2 = round(entry - sl_distance * _RISK_REWARD_T2, 2)
            tp1_ticks   = round(abs(take_profit_1 - entry) / tick)
            tp_adjusted = True

        # Reject setup if TP1 still below minimum (ATR too small)
        if tp1_ticks < _MIN_TP1_TICKS:
            return None

        sl_ticks  = round(sl_distance  / tick)
        tp1_ticks = round(abs(take_profit_1 - entry) / tick)
        tp2_ticks = round(abs(take_profit_2 - entry) / tick)

        setup = {
            "direction":             direction,
            "entry_price":           entry,

            "stop_loss_price":       stop_loss,
            "stop_loss_ticks":       sl_ticks,
            "stop_loss_points":      sl_distance,
            "stop_loss_usd_nq":      round(sl_ticks  * self.NQ_TICK_VALUE,  2),
            "stop_loss_usd_mnq":     round(sl_ticks  * self.MNQ_TICK_VALUE, 2),

            "take_profit_1_price":   take_profit_1,
            "take_profit_1_ticks":   tp1_ticks,
            "take_profit_1_usd_nq":  round(tp1_ticks * self.NQ_TICK_VALUE,  2),
            "take_profit_1_usd_mnq": round(tp1_ticks * self.MNQ_TICK_VALUE, 2),

            "take_profit_2_price":   take_profit_2,
            "take_profit_2_ticks":   tp2_ticks,
            "take_profit_2_usd_nq":  round(tp2_ticks * self.NQ_TICK_VALUE,  2),
            "take_profit_2_usd_mnq": round(tp2_ticks * self.MNQ_TICK_VALUE, 2),

            "risk_reward_tp1":       _RISK_REWARD_T1,
            "risk_reward_tp2":       _RISK_REWARD_T2,
            "atr_used":              round(atr, 2),
            "meets_min_ticks":       True,
        }
        if tp_adjusted:
            setup["tp_adjusted"] = True
            setup["tp_adjustment_reason"] = (
                f"TP1 war unter {_MIN_TP1_TICKS} Ticks — auf Minimum angepasst"
            )
        return setup


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
    reasoning:   str
    raw_score:   float            # weighted sum before normalisation
    trade_setup: Optional[dict]  = None


class SignalEngine:
    """
    Aggregates OrderFlowAnalyzer and TechnicalAnalyzer signals.

    Single public method: evaluate(book_snap, data_snap) → TradeRecommendation | None
    """

    # Per-signal weights for free-data mode
    _FREE_WEIGHTS: Dict[str, float] = {
        MULTI_TF_BIAS:   1.6,   # primary directional signal
        OVERNIGHT_GAP:   1.3,
        FAIR_VALUE_GAP:  1.2,
        EMA_TREND:       0.9,
        RSI_EXTREME:     1.0,
        VWAP_POSITION:   0.8,
        SESSION_LEVELS:  0.7,
        # Blocking signals — weight not used but listed for completeness
        VIX_REGIME:      0.0,
        CALENDAR_FILTER: 0.0,
    }

    def __init__(self, strategy_config=None) -> None:
        self._of_analyzer   = OrderFlowAnalyzer()
        self._ta_analyzer   = TechnicalAnalyzer()
        self._free_analyzer = FreeMarketAnalyzer()
        self._ict           = ICTSignalEngine()
        self.trade_setup    = TradeSetup()
        self._cfg           = strategy_config
        self.threshold      = (
            strategy_config.get("KONFIDENZ_SCHWELLEN", "min_confidence_normal", _MIN_CONFIDENCE)
            if strategy_config else _MIN_CONFIDENCE
        )
        self._learned_weights: Dict[str, float] = {}
        self._weights_mtime: float = 0.0
        self._reload_weights()

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

        trade_setup = TradeSetup().calculate(
            direction    = direction,
            entry_price  = entry_ref,
            atr          = atr,
            vwap         = data_snap.get("vwap"),
            session_high = data_snap.get("session_high"),
            session_low  = data_snap.get("session_low"),
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
            trade_setup = trade_setup,
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
                "trade_setup": None,
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
            "trade_setup": rec.trade_setup,
        }

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


    # ==================================================================
    # Free-data path (yfinance + FRED + calendar)
    # ==================================================================

    def evaluate_multi_tf(
        self, free_snap: dict
    ) -> Optional[TradeRecommendation]:
        """
        Evaluate signals using multi-timeframe yfinance data.

        Blocking conditions checked first:
          - VIX > 30   → return None
          - High-impact event within 30 min → return None

        CONFLUENCE_SCORE: requires _MIN_SIGNALS_FREE (3) agreeing signals.
        """
        all_signals = self._free_analyzer.analyze(free_snap)

        # Hard block: any signal with block=True stops the pipeline
        for sig in all_signals:
            if sig.metadata.get("block"):
                logger.info(
                    "Signal pipeline blocked: %s — %s",
                    sig.signal_type, sig.description,
                )
                return None

        # Apply VIX dampening: if VIX is high (25–30), lower all confidences by 20%
        vix = free_snap.get("vix", 0.0)
        if vix > _VIX_HIGH:
            all_signals = _dampen_confidence(all_signals, factor=0.80)

        # Add technical signals using 5m bars
        data_snap_5m = _free_snap_to_data_snap(free_snap)
        ta_signals   = self._ta_analyzer.analyze(data_snap_5m)
        all_signals  = all_signals + ta_signals

        if not all_signals:
            return None

        # Weighted directional vote using free-data weights
        bull_score, bear_score, bull_sigs, bear_sigs = self._score_signals(
            all_signals, weights=self._FREE_WEIGHTS
        )
        net_score    = bull_score - bear_score
        total_weight = bull_score + bear_score if (bull_score + bear_score) > 0 else 1.0

        if abs(net_score) == 0:
            return None

        direction, winning_sigs = (
            ("LONG",  bull_sigs) if net_score > 0 else
            ("SHORT", bear_sigs)
        )

        # CONFLUENCE_SCORE: need at least 3 distinct signal types
        unique_types = {
            str(s.signal_type.value if hasattr(s.signal_type, "value") else s.signal_type)
            for s in winning_sigs
        }
        if len(unique_types) < _MIN_SIGNALS_FREE:
            return None

        confidence = min(abs(net_score) / total_weight, 1.0)
        if confidence < _MIN_CONFIDENCE:
            return None

        last_price = free_snap.get("last_price", 0.0)
        bars_5m    = free_snap.get("bars_5m", [])
        atr        = compute_atr(bars_5m) or _FALLBACK_ATR

        entry_zone = self._swing_entry_zone(free_snap, direction, last_price)
        entry_ref  = (entry_zone["low"] + entry_zone["high"]) / 2
        stop_loss, target_1, target_2 = self._risk_levels(direction, entry_ref, atr)

        trade_setup = TradeSetup().calculate(
            direction    = direction,
            entry_price  = entry_ref,
            atr          = atr,
            vwap         = free_snap.get("session_vwap"),
            session_high = free_snap.get("session_high"),
            session_low  = free_snap.get("session_low"),
        )

        reasoning = self._build_reasoning(
            direction, confidence, winning_sigs, entry_zone,
            stop_loss, target_1, target_2, atr, last_price,
        )

        return TradeRecommendation(
            direction   = direction,
            confidence  = round(confidence, 3),
            signals     = [_signal_to_dict(s) for s in winning_sigs],
            entry_zone  = entry_zone,
            stop_loss   = stop_loss,
            target_1    = target_1,
            target_2    = target_2,
            atr         = round(atr, 2),
            reasoning   = reasoning,
            raw_score   = round(abs(net_score), 3),
            trade_setup = trade_setup,
        )

    def _swing_entry_zone(
        self, free_snap: dict, direction: str, last_price: float
    ) -> Dict[str, float]:
        """Entry zone derived from recent swing high / low on 5m bars."""
        bars = free_snap.get("bars_5m", [])
        vwap = free_snap.get("session_vwap") or last_price

        if not bars or not last_price:
            return {"low": round(last_price - 2.0, 2), "high": round(last_price + 2.0, 2)}

        recent = bars[-12:]   # last 60 min of 5m bars

        if direction == "LONG":
            lows = [b["low"] for b in recent if b["low"] < last_price]
            support    = max(lows) if lows else last_price - 2.0
            entry_low  = min(support, last_price)
            entry_high = max(support + 1.0, last_price)
            return {"low": round(entry_low, 2), "high": round(entry_high, 2)}
        else:
            highs = [b["high"] for b in recent if b["high"] > last_price]
            resistance = min(highs) if highs else last_price + 2.0
            entry_low  = min(last_price, resistance - 1.0)
            entry_high = max(last_price, resistance)
            return {"low": round(entry_low, 2), "high": round(entry_high, 2)}

    def _score_signals(self, signals: List[Signal], weights: Optional[Dict] = None):
        """Weighted directional vote. Uses _WEIGHTS (L2) or caller-supplied weights."""
        w = weights if weights is not None else _WEIGHTS
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
            weight   = w.get(sig_type, 1.0)
            weighted = sig.confidence * weight

            if sig.direction in (Direction.BULLISH, "BULLISH"):
                bull_score += weighted
                bull_sigs.append(sig)
            elif sig.direction in (Direction.BEARISH, "BEARISH"):
                bear_score += weighted
                bear_sigs.append(sig)

        return bull_score, bear_score, bull_sigs, bear_sigs

    # ==================================================================
    # Learned weight management
    # ==================================================================

    def _reload_weights(self) -> None:
        """Re-read signal_weights.json if the file has changed since last load."""
        try:
            mtime = _WEIGHTS_FILE.stat().st_mtime
        except OSError:
            return
        if mtime <= self._weights_mtime:
            return
        try:
            data = json.loads(_WEIGHTS_FILE.read_text(encoding="utf-8"))
            self._learned_weights = {k: float(v) for k, v in data.items()
                                     if isinstance(v, (int, float))}
            self._weights_mtime   = mtime
            logger.info("Signal weights reloaded from %s", _WEIGHTS_FILE.name)
        except Exception as e:
            logger.warning("Could not reload signal weights: %s", e)

    def _apply_learned_weights(
        self, signals: List[dict], ctx: dict
    ) -> List[dict]:
        """Apply learned per-signal weights and context modifiers in-place."""
        if not self._learned_weights:
            return signals

        vix_regime   = ctx.get("vix_regime", "normal")
        hour         = __import__("datetime").datetime.utcnow().hour
        time_of_day  = (
            "RTH_OPEN"  if 13 <= hour < 14 else
            "RTH_MID"   if 14 <= hour < 19 else
            "RTH_CLOSE" if 19 <= hour < 21 else
            "PREMARKET"
        )

        result = []
        for s in signals:
            sig_type   = s.get("type", "")
            base_conf  = s.get("confidence", 0.0)
            weight     = self._learned_weights.get(sig_type, 1.0)
            new_conf   = base_conf * weight

            # Context modifiers
            if vix_regime in ("high", "HIGH"):
                new_conf *= self._learned_weights.get("_vix_high_penalty", 0.8)
            elif vix_regime in ("extreme", "EXTREME"):
                new_conf *= self._learned_weights.get("_vix_extreme_penalty", 0.5)

            if time_of_day == "RTH_OPEN":
                new_conf *= self._learned_weights.get("_time_open_bonus", 1.2)
            elif time_of_day == "RTH_CLOSE":
                new_conf *= self._learned_weights.get("_time_close_penalty", 0.9)

            result.append({**s, "confidence": round(min(0.99, new_conf), 3)})
        return result

    # ==================================================================
    # Trade Scanner — always returns 3 candidates
    # ==================================================================

    def scan_trades(self, ctx: dict) -> dict:
        """
        Evaluates all strategies and always returns 3 candidates sorted by
        confidence (highest first).  A candidate is marked as "signal" when
        confidence >= self.threshold.
        """
        self._reload_weights()

        # Dynamic threshold from config (VIX-aware)
        vix_regime = ctx.get("vix_regime", "normal")
        if self._cfg:
            if vix_regime in ("extreme", "EXTREME"):
                self.threshold = self._cfg.get("KONFIDENZ_SCHWELLEN", "min_confidence_vix_extreme", 0.80)
            elif vix_regime in ("high", "HIGH"):
                self.threshold = self._cfg.get("KONFIDENZ_SCHWELLEN", "min_confidence_vix_high", 0.72)
            else:
                time_of_day = ctx.get("time_of_day", "RTH_MID")
                if time_of_day == "RTH_OPEN":
                    self.threshold = self._cfg.get("KONFIDENZ_SCHWELLEN", "min_confidence_rth_open", 0.60)
                elif time_of_day == "RTH_CLOSE":
                    self.threshold = self._cfg.get("KONFIDENZ_SCHWELLEN", "min_confidence_rth_close", 0.70)
                else:
                    self.threshold = self._cfg.get("KONFIDENZ_SCHWELLEN", "min_confidence_normal", 0.65)

        # PREMARKET / LUNCH — blockiere alle Signale außerhalb RTH
        time_of_day = ctx.get("time_of_day", "RTH_MID")
        if time_of_day in ("PREMARKET", "LUNCH"):
            blocked_reason = (
                "PREMARKET — Trades nur während RTH (09:30-16:00 ET)"
                if time_of_day == "PREMARKET"
                else "LUNCH (13:00-14:00 ET) — kein Trading"
            )
            dummy = [
                {"rank": i + 1, "direction": d, "confidence": 0.0,
                 "signals": [], "trade_setup": None, "is_signal": False,
                 "blocked_reason": blocked_reason}
                for i, d in enumerate(["LONG", "SHORT", "LONG"])
            ]
            return {
                "candidates":  dummy,
                "best_signal": dummy[0],
                "any_signal":  False,
                "threshold":   self.threshold,
                "ict_signals": {},
                "reversal":    {"reversal_type": None, "direction": None,
                                "confidence": 0.0, "signals": []},
            }

        price  = ctx.get("last_price", 0) or 0
        atr    = compute_atr(ctx.get("bars_5m", [])) or _FALLBACK_ATR
        vwap   = ctx.get("session_vwap") or None
        s_high = ctx.get("session_high") or None
        s_low  = ctx.get("session_low")  or None

        candidates = []

        # Candidate 1: LONG
        long_signals    = self._evaluate_direction(ctx, "LONG")
        long_confidence = self._calculate_confidence(long_signals)
        long_setup = self.trade_setup.calculate(
            "LONG", price, atr, vwap, s_high, s_low,
        ) if long_confidence > 0.3 and price > 0 else None

        candidates.append({
            "rank":       None,
            "direction":  "LONG",
            "confidence": long_confidence,
            "signals":    long_signals,
            "trade_setup": long_setup,
            "is_signal":  long_confidence >= self.threshold,
            "reasoning":  self._build_scan_reasoning("LONG", long_signals, long_confidence),
        })

        # Candidate 2: SHORT
        short_signals    = self._evaluate_direction(ctx, "SHORT")
        short_confidence = self._calculate_confidence(short_signals)
        short_setup = self.trade_setup.calculate(
            "SHORT", price, atr, vwap, s_high, s_low,
        ) if short_confidence > 0.3 and price > 0 else None

        candidates.append({
            "rank":       None,
            "direction":  "SHORT",
            "confidence": short_confidence,
            "signals":    short_signals,
            "trade_setup": short_setup,
            "is_signal":  short_confidence >= self.threshold,
            "reasoning":  self._build_scan_reasoning("SHORT", short_signals, short_confidence),
        })

        # Candidate 3: Special setups (Mean Reversion, Gap Fill)
        special_ctx = {
            "price":      price,
            "atr":        atr,
            "vwap":       vwap or 0,
            "session_high": s_high or 0,
            "session_low":  s_low  or 0,
            "bars_5m":    ctx.get("bars_5m", []),
            "prev_close": ctx.get("yesterday_close", 0) or 0,
        }
        neutral_signals    = self._evaluate_special_setups(special_ctx)
        neutral_confidence = self._calculate_confidence(neutral_signals)
        neutral_direction  = "LONG"
        if neutral_signals:
            first_dir = neutral_signals[0].get("direction", "")
            neutral_direction = "LONG" if first_dir in ["BULLISH", "LONG"] else "SHORT"

        candidates.append({
            "rank":       None,
            "direction":  neutral_direction,
            "confidence": neutral_confidence,
            "signals":    neutral_signals,
            "trade_setup": None,
            "is_signal":  neutral_confidence >= self.threshold,
            "label":      "Mean Reversion / Special Setup",
            "reasoning":  self._build_scan_reasoning(
                neutral_direction, neutral_signals, neutral_confidence
            ),
        })

        # Bias filter — suppress signals against the prevailing bias
        bias           = ctx.get("bias", {})
        bias_direction = bias.get("direction", "NEUTRAL")
        bias_prob      = bias.get("probability", 50)
        if bias_direction != "NEUTRAL" and bias_prob >= 65:
            for candidate in candidates:
                if candidate["direction"] != bias_direction:
                    candidate["is_signal"]       = False
                    candidate["blocked_reason"]  = (
                        f"Gegen Markt-Bias "
                        f"({bias_direction} {bias_prob:.0f}%)"
                    )

        # ICT analysis — FVG / Order Blocks / Market Structure / Killzones / HTF Bias
        ict_signals = self._ict.analyze(
            ctx.get("bars_5m",  []),
            ctx.get("bars_15m", []),
            ctx.get("bars_1h",  []),
        )
        ict_score       = ict_signals.get("ict_score", 0.0)
        killzone_bonus  = ict_signals.get("killzone_bonus", 1.0)
        killzone        = ict_signals.get("active_killzone")

        # Apply killzone bonus and inject ICT_CONFLUENCE signal
        for candidate in candidates:
            if candidate.get("is_signal") or candidate["confidence"] > 0.3:
                candidate["confidence"] = round(
                    min(0.99, candidate["confidence"] * killzone_bonus), 3
                )
                if candidate["confidence"] >= self.threshold:
                    candidate["is_signal"] = True

            if ict_score > 0.3:
                candidate.setdefault("signals", []).append({
                    "type":        "ICT_CONFLUENCE",
                    "direction":   candidate["direction"],
                    "confidence":  round(ict_score, 3),
                    "description": " | ".join(ict_signals.get("ict_reasons", [])),
                    "metadata": {
                        "killzone":         killzone,
                        "market_structure": ict_signals.get("market_structure"),
                        "order_blocks":     len(ict_signals.get("order_blocks", [])),
                        "fvg_count":        len(ict_signals.get("fvg_levels", [])),
                    },
                })

        # Outside killzones: require ≥80% confidence to fire a signal
        if not killzone:
            for candidate in candidates:
                if candidate.get("is_signal") and candidate["confidence"] < 0.80:
                    candidate["is_signal"]      = False
                    candidate["blocked_reason"] = (
                        "Außerhalb ICT Killzone — Konfidenz < 80% benötigt"
                    )

        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        for i, c in enumerate(candidates):
            c["rank"] = i + 1

        best = candidates[0]
        return {
            "candidates":   candidates,
            "best_signal":  best,
            "any_signal":   any(c["is_signal"] for c in candidates),
            "threshold":    self.threshold,
            "ict_signals":  ict_signals,
        }

    def _get_all_signals(self, ctx: dict) -> List[dict]:
        """Return all non-blocking signals as dicts, VIX dampening + learned weights applied."""
        vix_high_thresh = (
            self._cfg.get("VIX_REGIME_GRENZEN", "vix_normal_threshold", _VIX_HIGH)
            if self._cfg else _VIX_HIGH
        )
        vix_penalty = (
            self._cfg.get("VIX_REGIME_GRENZEN", "vix_penalty_high", 0.80)
            if self._cfg else 0.80
        )
        free_signals = self._free_analyzer.analyze(ctx)
        if ctx.get("vix", 0.0) > vix_high_thresh:
            free_signals = _dampen_confidence(free_signals, factor=vix_penalty)
        ta_signals = self._ta_analyzer.analyze(_free_snap_to_data_snap(ctx))
        signals = [
            _signal_to_dict(s)
            for s in (free_signals + ta_signals)
            if not s.metadata.get("block")
        ]
        signals = self._apply_strategy_weights(signals)
        return self._apply_learned_weights(signals, ctx)

    def _apply_strategy_weights(self, signals: List[dict]) -> List[dict]:
        """Apply strategy_config signal weights if available."""
        if not self._cfg:
            return signals
        result = []
        for s in signals:
            sig_type = s.get("type", "")
            weight   = self._cfg.get("SIGNAL_GEWICHTUNGEN", sig_type, 1.0)
            new_conf = round(min(0.99, s.get("confidence", 0.0) * weight), 3)
            result.append({**s, "confidence": new_conf})
        return result

    def _deduplicate_signals(self, signals: List[dict]) -> List[dict]:
        """Remove duplicate signal types — keep highest confidence per type+direction."""
        seen: dict = {}
        for s in signals:
            key = f"{s.get('type', '')}_{s.get('direction', '')}"
            if key not in seen or s.get("confidence", 0) > seen[key].get("confidence", 0):
                seen[key] = s
        return list(seen.values())

    def _evaluate_direction(self, ctx: dict, direction: str) -> List[dict]:
        """Return all signals that agree with the given direction."""
        target_dirs = ["BULLISH", "LONG"] if direction == "LONG" else ["BEARISH", "SHORT"]
        raw = [s for s in self._get_all_signals(ctx) if s.get("direction") in target_dirs]
        return self._deduplicate_signals(raw)

    def _evaluate_special_setups(self, ctx: dict) -> List[dict]:
        """Detect Mean Reversion and Gap Fill setups."""
        signals = []
        price = ctx.get("price", 0)
        vwap  = ctx.get("vwap", 0)

        if vwap > 0 and price > 0:
            distance = abs(price - vwap)
            if distance > ctx.get("atr", _FALLBACK_ATR) * 0.8:
                direction = "BULLISH" if price < vwap else "BEARISH"
                signals.append({
                    "type":        "MEAN_REVERSION",
                    "direction":   direction,
                    "confidence":  min(0.75, distance / (ctx.get("atr", _FALLBACK_ATR) * 2)),
                    "description": (
                        f"Preis {distance:.1f} Punkte von VWAP entfernt "
                        f"— Rückkehr wahrscheinlich"
                    ),
                })

        bars = ctx.get("bars_5m", [])
        prev_close = ctx.get("prev_close", 0)
        if len(bars) > 3 and prev_close > 0:
            gap = bars[0].get("open", 0) - prev_close
            if abs(gap) > 10:
                direction = "BEARISH" if gap > 0 else "BULLISH"
                signals.append({
                    "type":        "GAP_FILL",
                    "direction":   direction,
                    "confidence":  0.65,
                    "description": f"Gap von {gap:.1f} Punkten noch nicht gefüllt",
                })

        return signals

    @staticmethod
    def _calculate_confidence(signals: List[dict]) -> float:
        if not signals:
            return 0.0
        return min(0.95, sum(s.get("confidence", 0) for s in signals) / max(len(signals), 1))

    @staticmethod
    def _build_scan_reasoning(direction: str, signals: List[dict], confidence: float) -> str:
        if not signals:
            return f"Keine {direction} Signale aktiv"
        parts = [f"{s['type']} ({s.get('confidence', 0):.0%})" for s in signals]
        return f"{direction} {confidence:.0%} | " + " · ".join(parts)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _dampen_confidence(signals: List[Signal], factor: float) -> List[Signal]:
    """Return a new list with every signal's confidence multiplied by factor."""
    dampened = []
    for sig in signals:
        d = Signal.__new__(Signal)
        object.__setattr__(d, "signal_type",  sig.signal_type)
        object.__setattr__(d, "direction",    sig.direction)
        object.__setattr__(d, "confidence",   round(sig.confidence * factor, 3))
        object.__setattr__(d, "description",  sig.description)
        object.__setattr__(d, "metadata",     sig.metadata)
        dampened.append(d)
    return dampened


def _free_snap_to_data_snap(free_snap: dict) -> dict:
    """
    Translate a FreeDataClient snapshot into the data_snap format that
    TechnicalAnalyzer expects, using 5m bars as the "minute_bars" series.
    """
    return {
        "last_price":   free_snap.get("last_price", 0.0),
        "vwap":         free_snap.get("session_vwap"),
        "session_high": free_snap.get("session_high"),
        "session_low":  free_snap.get("session_low"),
        "minute_bars":  free_snap.get("bars_5m", []),   # 5m → TechnicalAnalyzer
        "second_bars":  free_snap.get("bars_1m", []),   # 1m → EMA base
        "second_bar_count": len(free_snap.get("bars_1m", [])),
        "minute_bar_count": len(free_snap.get("bars_5m", [])),
        "cumulative_delta_session": 0,
        "cumulative_delta_last10":  0,
        "cumulative_delta_last1":   0,
        "recent_tick_stats": {},
        "tick_count": 0,
    }


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
