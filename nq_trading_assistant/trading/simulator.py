import json
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime

TRADES_FILE  = Path(__file__).parent.parent / "logs" / "simulated_trades.json"
WEIGHTS_FILE = Path(__file__).parent.parent / "logs" / "signal_weights.json"

TICK_SIZE      = 0.25
TICK_VALUE_NQ  = 5.00


@dataclass
class SimulatedTrade:
    trade_id:          str
    timestamp_entry:   str
    direction:         str
    entry_price:       float
    stop_loss:         float
    take_profit_1:     float
    take_profit_2:     float
    confidence:        float
    active_signals:    list
    signal_details:    list
    vix:               float
    vwap:              float
    session_high:      float
    session_low:       float
    atr:               float
    vix_regime:        str
    time_of_day:       str

    timestamp_exit:          Optional[str]   = None
    exit_price:              Optional[float] = None
    exit_reason:             Optional[str]   = None
    duration_minutes:        Optional[float] = None
    pnl_points:              Optional[float] = None
    pnl_usd_nq:              Optional[float] = None
    outcome:                 Optional[str]   = None
    max_adverse_excursion:   Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    bias_direction:          str             = "NEUTRAL"
    bias_probability:        float           = 50.0
    # ICT context at entry
    killzone:                Optional[str]   = None
    ict_score:               Optional[float] = None
    order_blocks_active:     Optional[int]   = None
    market_structure:        Optional[str]   = None
    liquidity_levels:        Optional[int]   = None
    bias_1h:                 Optional[str]   = None
    fvg_count:               Optional[int]   = None


class TradeSimulator:
    def __init__(self, strategy_config=None):
        self._cfg = strategy_config
        TRADES_FILE.parent.mkdir(exist_ok=True)
        self._trades  = self._load_trades()
        self._weights = self._load_weights()

    def _load_trades(self):
        try:
            return json.loads(TRADES_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _load_weights(self):
        defaults = {
            "MULTI_TF_BIAS":        1.0,
            "FAIR_VALUE_GAP":       1.0,
            "VIX_REGIME":           1.0,
            "OVERNIGHT_GAP":        1.0,
            "CALENDAR_FILTER":      1.0,
            "EMA_TREND":            1.0,
            "VWAP_POSITION":        1.0,
            "RSI_EXTREME":          1.0,
            "MEAN_REVERSION":       1.0,
            "SESSION_LEVELS":       1.0,
            "_vix_high_penalty":    0.8,
            "_vix_extreme_penalty": 0.5,
            "_time_open_bonus":     1.2,
            "_time_close_penalty":  0.9,
            "_last_updated":        None,
            "_update_count":        0,
            "_total_trades_analyzed": 0,
        }
        try:
            saved = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
            defaults.update(saved)
        except Exception:
            pass
        return defaults

    def _save_weights(self):
        WEIGHTS_FILE.write_text(
            json.dumps(self._weights, indent=2),
            encoding="utf-8",
        )

    def _save_trades(self):
        TRADES_FILE.write_text(
            json.dumps(self._trades, indent=2, default=str),
            encoding="utf-8",
        )

    def _get_time_of_day(self) -> str:
        hour = datetime.utcnow().hour
        if 13 <= hour < 14:    return "RTH_OPEN"
        elif 14 <= hour < 19:  return "RTH_MID"
        elif 19 <= hour < 21:  return "RTH_CLOSE"
        else:                  return "PREMARKET"

    def open_trade(self, signal: dict, ctx: dict) -> Optional[str]:
        confidence = signal.get("confidence", 0)
        direction  = signal.get("direction", "NEUTRAL")

        min_conf = (
            self._cfg.get("KONFIDENZ_SCHWELLEN", "min_confidence_normal", 0.65)
            if self._cfg else 0.65
        )
        if confidence < min_conf or direction == "NEUTRAL":
            return None

        ts = signal.get("trade_setup") or {}
        if not ts:
            return None

        # Minimum TP1 filter
        tp1_ticks = ts.get("take_profit_1_ticks", 0)
        min_tp1   = (
            self._cfg.get("RISK_MANAGEMENT", "min_tp1_ticks", 80)
            if self._cfg else 80
        )
        if tp1_ticks < min_tp1:
            import logging
            logging.getLogger(__name__).debug(
                "Trade abgelehnt: TP1 nur %d Ticks (Minimum: %d Ticks = %d Punkte)",
                tp1_ticks, min_tp1, min_tp1 // 4,
            )
            return None

        # Confidence boost for large TP setups
        bonus_t1 = (
            self._cfg.get("RISK_MANAGEMENT", "tp_bonus_threshold_1", 120)
            if self._cfg else 120
        )
        bonus_t2 = (
            self._cfg.get("RISK_MANAGEMENT", "tp_bonus_threshold_2", 200)
            if self._cfg else 200
        )
        factor_1 = (
            self._cfg.get("RISK_MANAGEMENT", "tp_bonus_factor_1", 1.05)
            if self._cfg else 1.05
        )
        factor_2 = (
            self._cfg.get("RISK_MANAGEMENT", "tp_bonus_factor_2", 1.10)
            if self._cfg else 1.10
        )
        if tp1_ticks >= bonus_t2:
            confidence = min(0.95, confidence * factor_2)
        elif tp1_ticks >= bonus_t1:
            confidence = min(0.95, confidence * factor_1)

        # Bias filter — no trades against the prevailing bias
        bias      = ctx.get("bias") or {}
        bias_dir  = bias.get("direction", "NEUTRAL")
        bias_prob = bias.get("probability", 50)
        min_bias  = (
            self._cfg.get("BIAS_PARAMETER", "min_bias_probability", 0.65) * 100
            if self._cfg else 65
        )
        if bias_dir != "NEUTRAL" and bias_prob >= min_bias:
            if direction != bias_dir:
                return None

        # Max 1 open trade at a time; no duplicate direction
        open_trades = self.get_open_trades()
        if len(open_trades) >= 1:
            return None

        ict = ctx.get("ict_signals", {})
        ms  = ict.get("market_structure", {})
        ms_str = (
            f"{ms.get('type','?')}_{ms.get('direction','?')}" if ms else None
        )

        trade = SimulatedTrade(
            trade_id        = str(uuid.uuid4())[:8],
            timestamp_entry = datetime.utcnow().isoformat(),
            direction       = direction,
            entry_price     = ts.get("entry_price", ctx.get("price", 0)),
            stop_loss       = ts.get("stop_loss_price", 0),
            take_profit_1   = ts.get("take_profit_1_price", 0),
            take_profit_2   = ts.get("take_profit_2_price", 0),
            confidence      = confidence,
            active_signals  = [s.get("type") for s in signal.get("signals", [])],
            signal_details  = signal.get("signals", []),
            vix             = ctx.get("vix", 0),
            vwap            = ctx.get("session_vwap") or ctx.get("vwap", 0),
            session_high    = ctx.get("session_high", 0) or 0,
            session_low     = ctx.get("session_low", 0)  or 0,
            atr             = signal.get("atr", 0) or ts.get("atr_used", 0),
            vix_regime      = ctx.get("vix_regime", "normal"),
            time_of_day     = self._get_time_of_day(),
            bias_direction  = bias_dir,
            bias_probability = bias_prob,
            killzone         = ict.get("active_killzone"),
            ict_score        = ict.get("ict_score", 0),
            order_blocks_active = len(ict.get("order_blocks", [])),
            market_structure = ms_str,
            liquidity_levels = len(ict.get("liquidity_levels", [])),
            fvg_count        = len(ict.get("fvg_levels", [])),
            bias_1h          = (ctx.get("ict_signals", {})
                                .get("htf_bias", {})
                                .get("direction", "NEUTRAL")),
        )

        self._trades.append(asdict(trade))
        self._save_trades()
        return trade.trade_id

    def update_open_trades(self, current_price: float):
        now     = datetime.utcnow()
        updated = False

        for trade in self._trades:
            if trade.get("outcome") is not None:
                continue

            entry     = trade["entry_price"]
            sl        = trade["stop_loss"]
            tp1       = trade["take_profit_1"]
            tp2       = trade["take_profit_2"]
            direction = trade["direction"]

            try:
                entry_time = datetime.fromisoformat(trade["timestamp_entry"])
                duration   = (now - entry_time).total_seconds() / 60
            except Exception:
                duration = 0

            exit_reason = None
            exit_price  = None

            if direction == "LONG":
                if current_price <= sl:
                    exit_reason, exit_price = "SL",      sl
                elif current_price >= tp2:
                    exit_reason, exit_price = "WIN_TP2",  tp2
                elif current_price >= tp1:
                    exit_reason, exit_price = "WIN_TP1",  tp1
            else:
                if current_price >= sl:
                    exit_reason, exit_price = "SL",      sl
                elif current_price <= tp2:
                    exit_reason, exit_price = "WIN_TP2",  tp2
                elif current_price <= tp1:
                    exit_reason, exit_price = "WIN_TP1",  tp1

            timeout_min = (
                self._cfg.get("KERNREGELN", "trade_timeout_minutes", 240)
                if self._cfg else 240
            )
            if duration > timeout_min and exit_reason is None:
                exit_reason, exit_price = "TIMEOUT", current_price

            if exit_reason:
                pnl_points = (exit_price - entry) if direction == "LONG" \
                             else (entry - exit_price)
                ticks = pnl_points / TICK_SIZE

                trade["timestamp_exit"]  = now.isoformat()
                trade["exit_price"]      = exit_price
                trade["exit_reason"]     = exit_reason
                trade["duration_minutes"] = round(duration, 1)
                trade["pnl_points"]      = round(pnl_points, 2)
                trade["pnl_usd_nq"]      = round(ticks * TICK_VALUE_NQ, 2)
                trade["outcome"] = (
                    "WIN"     if "WIN"     in exit_reason else
                    "TIMEOUT" if exit_reason == "TIMEOUT" else
                    "LOSS"
                )
                updated = True

        if updated:
            self._save_trades()

    def get_open_trades(self) -> list:
        return [t for t in self._trades if t.get("outcome") is None]

    def get_closed_trades(self, limit: int = 50) -> list:
        closed = [t for t in self._trades if t.get("outcome") is not None]
        return sorted(
            closed,
            key=lambda x: x.get("timestamp_exit", ""),
            reverse=True,
        )[:limit]

    def get_stats(self) -> dict:
        closed = [t for t in self._trades if t.get("outcome") is not None]
        if not closed:
            return {"total": 0, "win_rate": 0, "avg_pnl": 0}

        wins     = [t for t in closed if t["outcome"] == "WIN"]
        losses   = [t for t in closed if t["outcome"] == "LOSS"]
        timeouts = [t for t in closed if t["outcome"] == "TIMEOUT"]

        return {
            "total":            len(closed),
            "open":             len(self.get_open_trades()),
            "wins":             len(wins),
            "losses":           len(losses),
            "timeouts":         len(timeouts),
            "win_rate":         round(len(wins) / len(closed) * 100, 1),
            "avg_pnl_points":   round(sum(t["pnl_points"] for t in closed) / len(closed), 2),
            "avg_pnl_usd":      round(sum(t["pnl_usd_nq"] for t in closed) / len(closed), 2),
            "total_pnl_usd":    round(sum(t["pnl_usd_nq"] for t in closed), 2),
            "best_signal_types":         self._best_signals(closed),
            "worst_signal_types":        self._worst_signals(closed),
            "win_rate_by_regime":        self._stats_by_regime(closed),
            "win_rate_by_time":          self._stats_by_time(closed),
            "win_rate_by_killzone":      self._stats_by_killzone(closed),
            "win_rate_by_market_structure": self._stats_by_market_structure(closed),
            "win_rate_by_ict_score":     self._stats_by_ict_score(closed),
            "win_rate_by_order_blocks":  self._stats_by_order_blocks(closed),
        }

    def _best_signals(self, trades: list) -> dict:
        signal_stats: dict = {}
        for t in trades:
            for sig in t.get("active_signals", []):
                if sig not in signal_stats:
                    signal_stats[sig] = {"wins": 0, "total": 0}
                signal_stats[sig]["total"] += 1
                if t["outcome"] == "WIN":
                    signal_stats[sig]["wins"] += 1
        result = {
            sig: round(s["wins"] / s["total"] * 100, 1)
            for sig, s in signal_stats.items()
            if s["total"] >= 3
        }
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def _worst_signals(self, trades: list) -> dict:
        best = self._best_signals(trades)
        return dict(sorted(best.items(), key=lambda x: x[1]))

    def _stats_by_regime(self, trades: list) -> dict:
        regimes: dict = {}
        for t in trades:
            r = t.get("vix_regime", "normal")
            if r not in regimes:
                regimes[r] = {"wins": 0, "total": 0}
            regimes[r]["total"] += 1
            if t["outcome"] == "WIN":
                regimes[r]["wins"] += 1
        return {r: round(s["wins"] / s["total"] * 100, 1)
                for r, s in regimes.items() if s["total"] > 0}

    def _stats_by_time(self, trades: list) -> dict:
        times: dict = {}
        for t in trades:
            tod = t.get("time_of_day", "RTH_MID")
            if tod not in times:
                times[tod] = {"wins": 0, "total": 0}
            times[tod]["total"] += 1
            if t["outcome"] == "WIN":
                times[tod]["wins"] += 1
        return {tod: round(s["wins"] / s["total"] * 100, 1)
                for tod, s in times.items() if s["total"] > 0}

    def _stats_by_killzone(self, trades: list) -> dict:
        kz: dict = {
            "IN_KILLZONE":      {"wins": 0, "total": 0},
            "OUTSIDE_KILLZONE": {"wins": 0, "total": 0},
        }
        for t in trades:
            key = "IN_KILLZONE" if t.get("killzone") else "OUTSIDE_KILLZONE"
            kz[key]["total"] += 1
            if t["outcome"] == "WIN":
                kz[key]["wins"] += 1
        return {
            k: round(v["wins"] / v["total"] * 100, 1)
            for k, v in kz.items() if v["total"] > 0
        }

    def _stats_by_market_structure(self, trades: list) -> dict:
        ms: dict = {}
        for t in trades:
            key = t.get("market_structure") or "NONE"
            if key not in ms:
                ms[key] = {"wins": 0, "total": 0}
            ms[key]["total"] += 1
            if t["outcome"] == "WIN":
                ms[key]["wins"] += 1
        return {
            k: round(v["wins"] / v["total"] * 100, 1)
            for k, v in ms.items() if v["total"] > 0
        }

    def _stats_by_ict_score(self, trades: list) -> dict:
        ranges: dict = {
            "HIGH (>0.6)":      {"wins": 0, "total": 0},
            "MEDIUM (0.3-0.6)": {"wins": 0, "total": 0},
            "LOW (<0.3)":       {"wins": 0, "total": 0},
            "NO_ICT":           {"wins": 0, "total": 0},
        }
        for t in trades:
            score = t.get("ict_score") or 0
            if score == 0:
                key = "NO_ICT"
            elif score >= 0.6:
                key = "HIGH (>0.6)"
            elif score >= 0.3:
                key = "MEDIUM (0.3-0.6)"
            else:
                key = "LOW (<0.3)"
            ranges[key]["total"] += 1
            if t["outcome"] == "WIN":
                ranges[key]["wins"] += 1
        return {
            k: round(v["wins"] / v["total"] * 100, 1)
            for k, v in ranges.items() if v["total"] > 0
        }

    def _stats_by_order_blocks(self, trades: list) -> dict:
        ob: dict = {
            "MIT_OB":  {"wins": 0, "total": 0},
            "OHNE_OB": {"wins": 0, "total": 0},
        }
        for t in trades:
            key = "MIT_OB" if (t.get("order_blocks_active") or 0) > 0 else "OHNE_OB"
            ob[key]["total"] += 1
            if t["outcome"] == "WIN":
                ob[key]["wins"] += 1
        return {
            k: round(v["wins"] / v["total"] * 100, 1)
            for k, v in ob.items() if v["total"] > 0
        }

    def apply_weights(self, weights_update: dict):
        self._weights.update(weights_update)
        self._weights["_last_updated"]  = datetime.utcnow().isoformat()
        self._weights["_update_count"]  = self._weights.get("_update_count", 0) + 1
        self._save_weights()

    def get_weights(self) -> dict:
        return self._weights
