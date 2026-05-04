from datetime import datetime, timedelta

import pandas as pd


class LimitOrderEngine:

    TICK_SIZE = 0.25

    def __init__(self, strategy_config=None):
        self._cfg = strategy_config

    def _p(self, section: str, key: str, default):
        if self._cfg:
            return self._cfg.get(section, key, default)
        return default

    def calculate_limit_levels(self, ctx: dict, bias: dict) -> list:
        """
        Berechne 1-3 vorausschauende Limit-Order-Level.
        Gibt Liste von Order-Setups sortiert nach Qualität zurück.
        """
        # Support both key naming conventions
        price        = ctx.get("price") or ctx.get("last_price", 0) or 0
        vwap         = ctx.get("vwap")  or ctx.get("session_vwap", 0) or 0
        bars_5m      = ctx.get("bars_5m",  [])
        bars_15m     = ctx.get("bars_15m", [])
        session_high = ctx.get("session_high") or 0
        session_low  = ctx.get("session_low")  or 0
        atr          = ctx.get("atr", 10) or 10
        bias_dir     = bias.get("direction", "NEUTRAL")

        vwap_offset   = self._p("LIMIT_ORDER_PARAMETER", "vwap_entry_offset",          0.50)
        ema_offset    = self._p("LIMIT_ORDER_PARAMETER", "ema_entry_offset",            0.25)
        min_dist_atr  = self._p("LIMIT_ORDER_PARAMETER", "min_distance_to_trigger_atr", 0.30)
        validity_min  = self._p("LIMIT_ORDER_PARAMETER", "order_validity_minutes",       30)
        fvg_min_size  = self._p("FILTER_PARAMETER",      "fvg_min_size_points",         2.0)
        fvg_min_qual  = self._p("LIMIT_ORDER_PARAMETER", "fvg_min_quality_score",       0.70)

        if bias_dir == "NEUTRAL" or price <= 0:
            return []

        candidates: list = []

        # ── LEVEL 1: VWAP Retest ─────────────────────────────────────────────
        if vwap > 0:
            distance_to_vwap = abs(price - vwap)
            if bias_dir == "LONG" and price > vwap and distance_to_vwap > atr * min_dist_atr:
                limit_price = self._round_tick(vwap + vwap_offset)
                candidates.append({
                    "type":          "VWAP_PULLBACK",
                    "direction":     "LONG",
                    "limit_price":   limit_price,
                    "trigger":       f"Pullback zu VWAP ({vwap:.2f})",
                    "quality":       0.75,
                    "distance_pts":  round(price - limit_price, 2),
                    "requires_move": round(price - limit_price, 2),
                })
            elif bias_dir == "SHORT" and price < vwap and distance_to_vwap > atr * min_dist_atr:
                limit_price = self._round_tick(vwap - vwap_offset)
                candidates.append({
                    "type":          "VWAP_RALLY",
                    "direction":     "SHORT",
                    "limit_price":   limit_price,
                    "trigger":       f"Rally zu VWAP ({vwap:.2f})",
                    "quality":       0.75,
                    "distance_pts":  round(limit_price - price, 2),
                    "requires_move": round(limit_price - price, 2),
                })

        # ── LEVEL 2: EMA21 Retest auf 5m ─────────────────────────────────────
        ema_min_dist = self._p("FILTER_PARAMETER", "ema_min_distance_atr", 0.20)
        if len(bars_5m) >= 21:
            df5   = pd.DataFrame(bars_5m[-50:])
            ema21 = self._round_tick(df5["close"].ewm(span=21).mean().iloc[-1])
            dist  = abs(price - ema21)

            if bias_dir == "LONG" and price > ema21 and dist > atr * ema_min_dist:
                candidates.append({
                    "type":          "EMA21_RETEST",
                    "direction":     "LONG",
                    "limit_price":   self._round_tick(ema21 + ema_offset),
                    "trigger":       f"Retest EMA21 auf 5m ({ema21:.2f})",
                    "quality":       0.70,
                    "distance_pts":  round(price - ema21, 2),
                    "requires_move": round(price - ema21, 2),
                })
            elif bias_dir == "SHORT" and price < ema21 and dist > atr * ema_min_dist:
                candidates.append({
                    "type":          "EMA21_RETEST",
                    "direction":     "SHORT",
                    "limit_price":   self._round_tick(ema21 - ema_offset),
                    "trigger":       f"Retest EMA21 auf 5m ({ema21:.2f})",
                    "quality":       0.70,
                    "distance_pts":  round(ema21 - price, 2),
                    "requires_move": round(ema21 - price, 2),
                })

        # ── LEVEL 3: Fair Value Gap Retest ────────────────────────────────────
        for fvg in self._find_fvg_levels(bars_5m, bias_dir, fvg_min_size, fvg_min_qual)[:2]:
            fvg["distance_pts"]  = round(abs(price - fvg["limit_price"]), 2)
            fvg["requires_move"] = fvg["distance_pts"]
            candidates.append(fvg)

        # ── LEVEL 4: Session Level ────────────────────────────────────────────
        if bias_dir == "LONG" and session_low > 0 and abs(price - session_low) > atr * 0.5:
            sl_level = self._round_tick(session_low + 0.25)
            candidates.append({
                "type":          "SESSION_LOW_RETEST",
                "direction":     "LONG",
                "limit_price":   sl_level,
                "trigger":       f"Retest Session Low ({session_low:.2f})",
                "quality":       0.80,
                "distance_pts":  round(price - sl_level, 2),
                "requires_move": round(price - sl_level, 2),
            })
        elif bias_dir == "SHORT" and session_high > 0 and abs(price - session_high) > atr * 0.5:
            sh_level = self._round_tick(session_high - 0.25)
            candidates.append({
                "type":          "SESSION_HIGH_RETEST",
                "direction":     "SHORT",
                "limit_price":   sh_level,
                "trigger":       f"Retest Session High ({session_high:.2f})",
                "quality":       0.80,
                "distance_pts":  round(sh_level - price, 2),
                "requires_move": round(sh_level - price, 2),
            })

        # Berechne SL/TP und füge Metadaten hinzu
        valid_until = (datetime.utcnow() + timedelta(minutes=validity_min)).strftime("%H:%M UTC")
        for c in candidates:
            c.update(self._calculate_sl_tp(c, atr))
            c["valid_until"] = valid_until
            c["status"]      = "WAITING"

        candidates.sort(key=lambda x: x["quality"], reverse=True)
        return candidates[:3]

    def _find_fvg_levels(
        self, bars_5m: list, bias_dir: str,
        fvg_min_size: float = 2.0, fvg_min_qual: float = 0.70
    ) -> list:
        """Finde offene Fair Value Gaps als Entry Levels."""
        if len(bars_5m) < 3:
            return []

        fvgs: list = []
        bars = bars_5m[-30:]

        for i in range(1, len(bars) - 1):
            prev = bars[i - 1]
            nxt  = bars[i + 1]

            if bias_dir == "LONG":
                gap_low  = prev.get("high", 0)
                gap_high = nxt.get("low", 0)
                if gap_high > gap_low and (gap_high - gap_low) >= fvg_min_size:
                    fvgs.append({
                        "type":        "FVG_BULLISH",
                        "direction":   "LONG",
                        "limit_price": self._round_tick(gap_low + 0.25),
                        "trigger":     f"Bullish FVG Retest ({gap_low:.2f}–{gap_high:.2f})",
                        "quality":     max(fvg_min_qual, 0.78),
                        "gap_low":     gap_low,
                        "gap_high":    gap_high,
                    })
            elif bias_dir == "SHORT":
                gap_low  = nxt.get("high", 0)
                gap_high = prev.get("low", 0)
                if gap_high > gap_low and (gap_high - gap_low) >= fvg_min_size:
                    fvgs.append({
                        "type":        "FVG_BEARISH",
                        "direction":   "SHORT",
                        "limit_price": self._round_tick(gap_high - 0.25),
                        "trigger":     f"Bearish FVG Retest ({gap_low:.2f}–{gap_high:.2f})",
                        "quality":     0.78,
                        "gap_low":     gap_low,
                        "gap_high":    gap_high,
                    })

        return fvgs[-2:] if fvgs else []

    _MIN_TP1_TICKS  = 80
    _MIN_TP1_POINTS = 20.0

    def _calculate_sl_tp(self, candidate: dict, atr: float) -> dict:
        entry     = candidate["limit_price"]
        direction = candidate["direction"]
        sl_mult   = self._p("RISK_MANAGEMENT", "sl_atr_multiplier", 0.5)
        sl_min    = self._p("RISK_MANAGEMENT", "sl_min_points",     2.0)
        tp1_mult  = self._p("RISK_MANAGEMENT", "tp1_rr_multiplier", 1.5)
        tp2_mult  = self._p("RISK_MANAGEMENT", "tp2_rr_multiplier", 2.5)
        sl_dist   = self._round_tick(max(atr * sl_mult, sl_min))

        min_tp1   = self._p("RISK_MANAGEMENT", "min_tp1_ticks",  self._MIN_TP1_TICKS)
        min_pts   = self._p("RISK_MANAGEMENT", "min_tp1_points", self._MIN_TP1_POINTS)

        if direction == "LONG":
            stop_loss  = entry - sl_dist
            tp1        = entry + sl_dist * tp1_mult
            tp2        = entry + sl_dist * tp2_mult
        else:
            stop_loss  = entry + sl_dist
            tp1        = entry - sl_dist * tp1_mult
            tp2        = entry - sl_dist * tp2_mult

        tp1_ticks = int(abs(tp1 - entry) / self.TICK_SIZE)

        tp_adjusted = False
        if tp1_ticks < min_tp1:
            # Expand SL so that TP1 = SL × tp1_mult ≥ min_pts
            required_sl = self._round_tick(max(min_pts / tp1_mult, sl_dist))
            if required_sl > sl_dist:
                sl_dist = required_sl
                if direction == "LONG":
                    stop_loss = entry - sl_dist
                    tp1       = entry + sl_dist * tp1_mult
                    tp2       = entry + sl_dist * tp2_mult
                else:
                    stop_loss = entry + sl_dist
                    tp1       = entry - sl_dist * tp1_mult
                    tp2       = entry - sl_dist * tp2_mult
                tp_adjusted = True

        sl_ticks  = int(sl_dist / self.TICK_SIZE)
        tp1_ticks = int(abs(tp1 - entry) / self.TICK_SIZE)

        return {
            "stop_loss":              self._round_tick(stop_loss),
            "take_profit_1":          self._round_tick(tp1),
            "take_profit_2":          self._round_tick(tp2),
            "stop_loss_ticks":        sl_ticks,
            "stop_loss_usd_nq":       sl_ticks * 5,
            "take_profit_1_ticks":    tp1_ticks,
            "take_profit_1_usd_nq":   tp1_ticks * 5,
            "risk_reward":            1.5,
            "atr_used":               round(atr, 2),
            "tp_adjusted":            tp_adjusted,
            "meets_min_ticks":        tp1_ticks >= min_tp1,
        }

    def _round_tick(self, price: float) -> float:
        return round(round(price / self.TICK_SIZE) * self.TICK_SIZE, 2)

    def check_triggered(self, orders: list, current_price: float) -> list:
        """Prüfe ob wartende Limit Orders durch den aktuellen Preis getriggert wurden."""
        now_str = datetime.utcnow().strftime("%H:%M UTC")

        for order in orders:
            if order.get("status") != "WAITING":
                continue
            limit     = order["limit_price"]
            direction = order["direction"]

            if direction == "LONG"  and current_price <= limit:
                order["status"]          = "TRIGGERED"
                order["triggered_at"]    = now_str
                order["triggered_price"] = current_price
            elif direction == "SHORT" and current_price >= limit:
                order["status"]          = "TRIGGERED"
                order["triggered_at"]    = now_str
                order["triggered_price"] = current_price

        return orders
