import logging

import pandas as pd

logger = logging.getLogger(__name__)


class MarketBiasEngine:
    """
    Berechnet den übergeordneten Markt-Bias (LONG/SHORT/NEUTRAL).
    Nur Trades in Bias-Richtung werden zugelassen.
    """

    def calculate_bias(self, ctx: dict) -> dict:
        bars_1m  = ctx.get("bars_1m",  [])
        bars_5m  = ctx.get("bars_5m",  [])
        bars_15m = ctx.get("bars_15m", [])
        # Support both key conventions
        price = ctx.get("price") or ctx.get("last_price", 0) or 0
        vwap  = ctx.get("vwap")  or ctx.get("session_vwap", 0) or 0
        session_high = ctx.get("session_high") or 0
        session_low  = ctx.get("session_low")  or 0
        vix          = ctx.get("vix", 20)
        vix_regime   = ctx.get("vix_regime", "normal")

        bullish_score = 0.0
        bearish_score = 0.0
        reasons: list = []

        # 1. VWAP Position (Gewicht: 2.0)
        if vwap > 0:
            if price > vwap * 1.001:
                bullish_score += 2.0
                reasons.append(f"Preis über VWAP (+{price - vwap:.1f} Pts)")
            elif price < vwap * 0.999:
                bearish_score += 2.0
                reasons.append(f"Preis unter VWAP ({price - vwap:.1f} Pts)")

        # 2. EMA Trend auf 15m (Gewicht: 3.0)
        if len(bars_15m) >= 21:
            df15    = pd.DataFrame(bars_15m[-50:])
            ema9_15  = df15["close"].ewm(span=9).mean().iloc[-1]
            ema21_15 = df15["close"].ewm(span=21).mean().iloc[-1]
            ema50_15 = (df15["close"].ewm(span=50).mean().iloc[-1]
                        if len(bars_15m) >= 50 else ema21_15)

            if ema9_15 > ema21_15 > ema50_15:
                bullish_score += 3.0
                reasons.append("15m EMA Stack bullish (9>21>50)")
            elif ema9_15 < ema21_15 < ema50_15:
                bearish_score += 3.0
                reasons.append("15m EMA Stack bearish (9<21<50)")
            elif ema9_15 > ema21_15:
                bullish_score += 1.5
                reasons.append("15m EMA9 > EMA21 bullish")
            else:
                bearish_score += 1.5
                reasons.append("15m EMA9 < EMA21 bearish")

        # 3. EMA Trend auf 5m (Gewicht: 2.0)
        if len(bars_5m) >= 21:
            df5    = pd.DataFrame(bars_5m[-50:])
            ema9_5  = df5["close"].ewm(span=9).mean().iloc[-1]
            ema21_5 = df5["close"].ewm(span=21).mean().iloc[-1]

            if ema9_5 > ema21_5:
                bullish_score += 2.0
                reasons.append("5m EMA9 > EMA21 bullish")
            else:
                bearish_score += 2.0
                reasons.append("5m EMA9 < EMA21 bearish")

        # 4. Session High/Low Position (Gewicht: 1.5)
        if session_high > 0 and session_low > 0:
            session_range = session_high - session_low
            if session_range > 0:
                position = (price - session_low) / session_range
                if position > 0.6:
                    bullish_score += 1.5
                    reasons.append(
                        f"Preis im oberen Drittel der Session ({position:.0%})"
                    )
                elif position < 0.4:
                    bearish_score += 1.5
                    reasons.append(
                        f"Preis im unteren Drittel der Session ({position:.0%})"
                    )

        # 5. Momentum der letzten 5 1m-Bars (Gewicht: 1.0)
        if len(bars_1m) >= 5:
            closes   = [b["close"] for b in bars_1m[-5:]]
            momentum = closes[-1] - closes[0]
            if momentum > 2:
                bullish_score += 1.0
                reasons.append(f"1m Momentum bullish (+{momentum:.1f} Pts)")
            elif momentum < -2:
                bearish_score += 1.0
                reasons.append(f"1m Momentum bearish ({momentum:.1f} Pts)")

        # 6. VIX Regime — abschwächend bei extremen Werten
        if vix_regime in ("extreme", "EXTREME"):
            bullish_score *= 0.5
            bearish_score *= 0.5
            reasons.append("⚠️ VIX extrem — Bias abgeschwächt")

        total_score = bullish_score + bearish_score
        if total_score == 0:
            return {
                "direction":    "NEUTRAL",
                "probability":  50.0,
                "strength":     "WEAK",
                "bullish_score": 0.0,
                "bearish_score": 0.0,
                "bull_prob":    50.0,
                "bear_prob":    50.0,
                "reasons":      ["Kein klarer Bias erkennbar"],
            }

        bull_prob = (bullish_score / total_score) * 100
        bear_prob = (bearish_score / total_score) * 100

        if bull_prob >= 70:
            direction   = "LONG"
            probability = bull_prob
            strength    = "STRONG" if bull_prob >= 80 else "MODERATE"
        elif bear_prob >= 70:
            direction   = "SHORT"
            probability = bear_prob
            strength    = "STRONG" if bear_prob >= 80 else "MODERATE"
        else:
            direction   = "NEUTRAL"
            probability = max(bull_prob, bear_prob)
            strength    = "WEAK"

        return {
            "direction":    direction,
            "probability":  round(probability, 1),
            "strength":     strength,
            "bullish_score": round(bullish_score, 2),
            "bearish_score": round(bearish_score, 2),
            "bull_prob":    round(bull_prob, 1),
            "bear_prob":    round(bear_prob, 1),
            "reasons":      reasons,
        }
