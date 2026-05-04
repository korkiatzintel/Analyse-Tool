import pandas as pd


class ReversalEngine:
    """
    Erkennt potenzielle Wendepunkte via:
    1. Liquidity Sweep + CHoCH Sequenz (stärkstes Signal)
    2. CHoCH + BOS Bestätigung (sicherster Einstieg)
    3. Extreme VWAP Abweichung + RSI Divergenz
    """

    MIN_SWEEP_BARS      = 5
    SWEEP_CONFIRM_BARS  = 3

    def analyze(self, bars_5m: list, bars_15m: list, ctx: dict) -> dict:
        if len(bars_5m) < 20:
            return self._no_reversal()

        df5  = pd.DataFrame(bars_5m[-50:]).reset_index(drop=True)
        df15 = (
            pd.DataFrame(bars_15m[-30:]).reset_index(drop=True)
            if len(bars_15m) >= 10 else df5
        )

        price        = ctx.get("price", 0) or ctx.get("last_price", 0)
        vwap         = ctx.get("vwap", 0) or ctx.get("session_vwap", 0)
        vix          = ctx.get("vix", 20) or 20
        atr          = ctx.get("atr", 10) or 10
        session_high = ctx.get("session_high", 0) or 0
        session_low  = ctx.get("session_low",  0) or 0

        signals:            list  = []
        reversal_score:     float = 0.0
        reversal_direction        = None

        # ── Signal 1: Liquidity Sweep ────────────────────────────────────
        sweep = self._detect_liquidity_sweep(df5)
        if sweep:
            signals.append({
                "type":        "LIQUIDITY_SWEEP",
                "direction":   sweep["direction"],
                "description": sweep["description"],
                "weight":      0.35,
            })
            reversal_score     += 0.35
            reversal_direction  = sweep["reversal_direction"]

        # ── Signal 2: CHoCH ───────────────────────────────────────────────
        choch = self._detect_choch(df5)
        if choch:
            signals.append({
                "type":        "CHOCH",
                "direction":   choch["direction"],
                "description": choch["description"],
                "weight":      0.30,
            })
            reversal_score += 0.30
            if reversal_direction is None:
                reversal_direction = choch["direction"]
            elif reversal_direction == choch["direction"]:
                reversal_score += 0.15  # Bonus: Sweep + CHoCH in gleicher Richtung

        # ── Signal 3: Extreme VWAP Abweichung ────────────────────────────
        if vwap > 0 and atr > 0 and price > 0:
            distance = abs(price - vwap)
            if distance > atr * 1.5:
                direction = "LONG" if price < vwap else "SHORT"
                signals.append({
                    "type":        "EXTREME_VWAP_DEVIATION",
                    "direction":   direction,
                    "description": (
                        f"Preis {distance:.1f} Pts von VWAP "
                        f"({distance / atr:.1f}x ATR) — "
                        f"Mean Reversion wahrscheinlich"
                    ),
                    "weight": 0.20,
                })
                reversal_score += 0.20
                if reversal_direction is None:
                    reversal_direction = direction

        # ── Signal 4: RSI Divergenz ───────────────────────────────────────
        rsi_div = self._detect_rsi_divergence(df5)
        if rsi_div:
            signals.append({
                "type":        "RSI_DIVERGENCE",
                "direction":   rsi_div["direction"],
                "description": rsi_div["description"],
                "weight":      0.15,
            })
            reversal_score += 0.15

        # ── Signal 5: Session Extreme ─────────────────────────────────────
        if session_high > 0 and session_low > 0 and price > 0 and atr > 0:
            if price >= session_high - atr * 0.3:
                signals.append({
                    "type":        "SESSION_HIGH_EXTREME",
                    "direction":   "SHORT",
                    "description": (
                        f"Preis an Session High ({session_high:.2f}) — "
                        f"Widerstand erwartet"
                    ),
                    "weight": 0.15,
                })
                reversal_score    += 0.15
                reversal_direction = "SHORT"
            elif price <= session_low + atr * 0.3:
                signals.append({
                    "type":        "SESSION_LOW_EXTREME",
                    "direction":   "LONG",
                    "description": (
                        f"Preis an Session Low ({session_low:.2f}) — "
                        f"Support erwartet"
                    ),
                    "weight": 0.15,
                })
                reversal_score    += 0.15
                reversal_direction = "LONG"

        # ── VIX Dämpfung ──────────────────────────────────────────────────
        if vix > 30:
            reversal_score *= 0.70
        elif vix > 25:
            reversal_score *= 0.80

        # ── Mindest-Score + Richtung ──────────────────────────────────────
        if reversal_score < 0.35 or not reversal_direction:
            return self._no_reversal()

        # Konsistenz-Check: >70% des Gewichts muss in eine Richtung zeigen
        direction_votes: dict = {}
        for s in signals:
            d = s.get("direction", "")
            direction_votes[d] = direction_votes.get(d, 0) + s["weight"]

        total_weight = sum(direction_votes.values())
        max_dir      = max(direction_votes, key=direction_votes.get) if direction_votes else None
        max_weight   = direction_votes.get(max_dir, 0)

        if total_weight > 0 and max_weight / total_weight < 0.70:
            return self._no_reversal()

        reversal_direction = max_dir

        # ── Klassifizierung ───────────────────────────────────────────────
        has_sweep = any(s["type"] == "LIQUIDITY_SWEEP" for s in signals)
        has_choch = any(s["type"] == "CHOCH"           for s in signals)

        if has_sweep and has_choch:
            reversal_type  = "CONFIRMED"
            tp_multiplier  = 2.0
            sl_multiplier  = 0.70
        elif has_choch or reversal_score >= 0.50:
            reversal_type  = "POTENTIAL"
            tp_multiplier  = 1.50
            sl_multiplier  = 0.85
        else:
            reversal_type  = "WEAK"
            tp_multiplier  = 1.20
            sl_multiplier  = 0.95

        return {
            "reversal_type":  reversal_type,
            "direction":      reversal_direction,
            "confidence":     round(min(reversal_score, 0.95), 3),
            "signals":        signals,
            "tp_multiplier":  tp_multiplier,
            "sl_multiplier":  sl_multiplier,
            "description":    (
                f"{reversal_type} Wendepunkt {reversal_direction} "
                f"({reversal_score:.0%})"
            ),
        }

    # ── Detection helpers ─────────────────────────────────────────────────

    def _detect_liquidity_sweep(self, df: pd.DataFrame) -> dict | None:
        if len(df) < self.MIN_SWEEP_BARS + 2:
            return None

        recent    = df.tail(self.MIN_SWEEP_BARS + 2)
        last      = recent.iloc[-1]
        prev_bars = recent.iloc[:-1]

        swing_high = prev_bars["high"].max()
        swing_low  = prev_bars["low"].min()

        if (last["high"] > swing_high
                and last["close"] < swing_high
                and last["close"] < last["open"]):
            return {
                "direction":          "SHORT",
                "reversal_direction": "SHORT",
                "level":              swing_high,
                "description": (
                    f"Bearish Liquidity Sweep: Wick über {swing_high:.2f}, "
                    f"Close darunter — Stop Hunts abgeschlossen"
                ),
            }

        if (last["low"] < swing_low
                and last["close"] > swing_low
                and last["close"] > last["open"]):
            return {
                "direction":          "LONG",
                "reversal_direction": "LONG",
                "level":              swing_low,
                "description": (
                    f"Bullish Liquidity Sweep: Wick unter {swing_low:.2f}, "
                    f"Close darüber — Stop Hunts abgeschlossen"
                ),
            }

        return None

    def _detect_choch(self, df: pd.DataFrame) -> dict | None:
        if len(df) < 10:
            return None

        highs: list = []
        lows:  list = []

        for i in range(2, len(df) - 2):
            h = df.iloc[i]["high"]
            if (h > df.iloc[i - 1]["high"] and h > df.iloc[i - 2]["high"]
                    and h > df.iloc[i + 1]["high"] and h > df.iloc[i + 2]["high"]):
                highs.append((i, h))

            l = df.iloc[i]["low"]
            if (l < df.iloc[i - 1]["low"] and l < df.iloc[i - 2]["low"]
                    and l < df.iloc[i + 1]["low"] and l < df.iloc[i + 2]["low"]):
                lows.append((i, l))

        if not highs or not lows:
            return None

        last_close     = df.iloc[-1]["close"]
        last_swing_high = highs[-1][1]
        last_swing_low  = lows[-1][1]

        if (last_close > last_swing_high
                and len(lows) >= 2
                and lows[-1][1] > lows[-2][1]):
            return {
                "direction": "LONG",
                "level":     last_swing_high,
                "description": (
                    f"Bullish CHoCH: Bruch über {last_swing_high:.2f} — "
                    f"Struktur dreht nach oben"
                ),
            }

        if (last_close < last_swing_low
                and len(highs) >= 2
                and highs[-1][1] < highs[-2][1]):
            return {
                "direction": "SHORT",
                "level":     last_swing_low,
                "description": (
                    f"Bearish CHoCH: Bruch unter {last_swing_low:.2f} — "
                    f"Struktur dreht nach unten"
                ),
            }

        return None

    def _detect_rsi_divergence(self, df: pd.DataFrame) -> dict | None:
        if len(df) < 15:
            return None

        closes = df["close"].values
        period = 14
        if len(closes) <= period:
            return None

        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains  = [max(d, 0)        for d in deltas]
        losses = [abs(min(d, 0))   for d in deltas]

        avg_gain = sum(gains[-period:])  / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return None

        rsi = 100 - (100 / (1 + avg_gain / avg_loss))

        if rsi < 35 and closes[-1] >= min(closes[-5:-1]):
            return {
                "direction":   "LONG",
                "description": f"Bullish RSI Divergenz: RSI {rsi:.0f} — Momentum dreht",
            }

        if rsi > 65 and closes[-1] <= max(closes[-5:-1]):
            return {
                "direction":   "SHORT",
                "description": f"Bearish RSI Divergenz: RSI {rsi:.0f} — Momentum dreht",
            }

        return None

    def _no_reversal(self) -> dict:
        return {
            "reversal_type": None,
            "direction":     None,
            "confidence":    0.0,
            "signals":       [],
            "tp_multiplier": 1.0,
            "sl_multiplier": 1.0,
            "description":   "Kein Wendepunkt erkannt",
        }
