import logging

import pandas as pd

logger = logging.getLogger(__name__)


class MarketBiasEngine:
    """
    Berechnet den übergeordneten Markt-Bias (LONG/SHORT/NEUTRAL).
    Nur Trades in Bias-Richtung werden zugelassen.
    """

    def __init__(self, strategy_config=None):
        self._cfg = strategy_config

    def _w(self, key: str, default: float) -> float:
        if self._cfg:
            return self._cfg.get("BIAS_PARAMETER", key, default)
        return default

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

        w_vwap    = self._w("vwap_weight",             2.0)
        w_ema15m  = self._w("ema15m_weight",           3.0)
        w_ema5m   = self._w("ema5m_weight",            2.0)
        w_session = self._w("session_position_weight", 1.5)
        w_mom     = self._w("momentum_weight",         1.0)

        # 1. VWAP Position
        if vwap > 0:
            if price > vwap * 1.001:
                bullish_score += w_vwap
                reasons.append(f"Preis über VWAP (+{price - vwap:.1f} Pts)")
            elif price < vwap * 0.999:
                bearish_score += w_vwap
                reasons.append(f"Preis unter VWAP ({price - vwap:.1f} Pts)")

        # 2. EMA Trend auf 15m
        if len(bars_15m) >= 21:
            df15    = pd.DataFrame(bars_15m[-50:])
            ema9_15  = df15["close"].ewm(span=9).mean().iloc[-1]
            ema21_15 = df15["close"].ewm(span=21).mean().iloc[-1]
            ema50_15 = (df15["close"].ewm(span=50).mean().iloc[-1]
                        if len(bars_15m) >= 50 else ema21_15)

            if ema9_15 > ema21_15 > ema50_15:
                bullish_score += w_ema15m
                reasons.append("15m EMA Stack bullish (9>21>50)")
            elif ema9_15 < ema21_15 < ema50_15:
                bearish_score += w_ema15m
                reasons.append("15m EMA Stack bearish (9<21<50)")
            elif ema9_15 > ema21_15:
                bullish_score += w_ema15m * 0.5
                reasons.append("15m EMA9 > EMA21 bullish")
            else:
                bearish_score += w_ema15m * 0.5
                reasons.append("15m EMA9 < EMA21 bearish")

        # 3. EMA Trend auf 5m
        if len(bars_5m) >= 21:
            df5    = pd.DataFrame(bars_5m[-50:])
            ema9_5  = df5["close"].ewm(span=9).mean().iloc[-1]
            ema21_5 = df5["close"].ewm(span=21).mean().iloc[-1]

            if ema9_5 > ema21_5:
                bullish_score += w_ema5m
                reasons.append("5m EMA9 > EMA21 bullish")
            else:
                bearish_score += w_ema5m
                reasons.append("5m EMA9 < EMA21 bearish")

        # 4. Session High/Low Position
        if session_high > 0 and session_low > 0:
            session_range = session_high - session_low
            if session_range > 0:
                position = (price - session_low) / session_range
                if position > 0.6:
                    bullish_score += w_session
                    reasons.append(
                        f"Preis im oberen Drittel der Session ({position:.0%})"
                    )
                elif position < 0.4:
                    bearish_score += w_session
                    reasons.append(
                        f"Preis im unteren Drittel der Session ({position:.0%})"
                    )

        # 5. Momentum der letzten 5 1m-Bars
        if len(bars_1m) >= 5:
            closes   = [b["close"] for b in bars_1m[-5:]]
            momentum = closes[-1] - closes[0]
            if momentum > 2:
                bullish_score += w_mom
                reasons.append(f"1m Momentum bullish (+{momentum:.1f} Pts)")
            elif momentum < -2:
                bearish_score += w_mom
                reasons.append(f"1m Momentum bearish ({momentum:.1f} Pts)")

        # 6. VIX Regime — abschwächend bei extremen Werten
        if vix_regime in ("extreme", "EXTREME"):
            vix_extreme_penalty = (
                self._cfg.get("VIX_REGIME_GRENZEN", "vix_penalty_extreme", 0.50)
                if self._cfg else 0.50
            )
            bullish_score *= vix_extreme_penalty
            bearish_score *= vix_extreme_penalty
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
