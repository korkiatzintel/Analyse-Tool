import logging
from datetime import datetime, timezone, timedelta

import pandas as pd


class MarketBiasEngine:
    """
    Professionelle 4-Ebenen Bias Engine für NQ Futures.

    Architektur (HTF → LTF):
    Ebene 1: 1H Struktur     — Institutioneller Kontext  (35%)
    Ebene 2: Session Kontext — VWAP, Premium/Discount    (25%)
    Ebene 3: Key Levels      — PDH/PDL, ORB              (20%)
    Ebene 4: LTF Bestätigung — 5m/15m Ausrichtung        (20%)

    Nur wenn alle 4 Ebenen übereinstimmen → hoher Score.
    """

    def __init__(self, strategy_config=None):
        self._log = logging.getLogger(__name__)
        self._cfg = strategy_config

    def calculate_bias(self, ctx: dict) -> dict:
        bars_1m  = ctx.get("bars_1m",  [])
        bars_5m  = ctx.get("bars_5m",  [])
        bars_15m = ctx.get("bars_15m", [])
        bars_1h  = ctx.get("bars_1h",  [])

        price        = ctx.get("price") or ctx.get("last_price", 0) or 0
        vwap         = ctx.get("vwap")  or ctx.get("session_vwap", 0) or 0
        session_high = ctx.get("session_high") or 0
        session_low  = ctx.get("session_low")  or 0
        vix          = ctx.get("vix", 20)

        if price == 0:
            return self._neutral_bias("Kein Preis verfügbar")

        bullish = 0.0
        bearish = 0.0
        details = {}

        # ══ EBENE 1: 1H STRUKTUR-BIAS (Gewicht: 35%) ══════════════════════
        e1 = self._level1_htf_bias(bars_1h, price, vwap)
        bullish += e1["bull"] * 0.35
        bearish += e1["bear"] * 0.35
        details["htf_bias"] = e1

        # ══ EBENE 2: SESSION KONTEXT (Gewicht: 25%) ════════════════════════
        e2 = self._level2_session_context(
            bars_5m, price, session_high, session_low, vwap)
        bullish += e2["bull"] * 0.25
        bearish += e2["bear"] * 0.25
        details["session_context"] = e2

        # ══ EBENE 3: KEY LEVELS — PDH/PDL & ORB (Gewicht: 20%) ═══════════
        e3 = self._level3_key_levels(
            bars_1h, bars_5m, price, session_high, session_low)
        bullish += e3["bull"] * 0.20
        bearish += e3["bear"] * 0.20
        details["key_levels"] = e3

        # ══ EBENE 4: LTF BESTÄTIGUNG (Gewicht: 20%) ═══════════════════════
        e4 = self._level4_ltf_confirmation(bars_5m, bars_15m, price, vwap)
        bullish += e4["bull"] * 0.20
        bearish += e4["bear"] * 0.20
        details["ltf_confirmation"] = e4

        # ══ VIX ANPASSUNG ══════════════════════════════════════════════════
        if vix > 30:
            bullish *= 0.70
            bearish *= 0.70
        elif vix > 25:
            bullish *= 0.85
            bearish *= 0.85

        # ══ SCORE BERECHNUNG ═══════════════════════════════════════════════
        total = bullish + bearish
        if total == 0:
            return self._neutral_bias("Keine Signale")

        bull_pct = (bullish / total) * 100
        bear_pct = (bearish / total) * 100

        premium_discount = self._premium_discount(price, session_high, session_low)

        if bull_pct >= 70:
            direction   = "LONG"
            probability = bull_pct
            strength    = "STRONG" if bull_pct >= 80 else "MODERATE"
        elif bear_pct >= 70:
            direction   = "SHORT"
            probability = bear_pct
            strength    = "STRONG" if bear_pct >= 80 else "MODERATE"
        else:
            direction   = "NEUTRAL"
            probability = max(bull_pct, bear_pct)
            strength    = "WEAK"

        reasons = []
        for level_data in details.values():
            reasons.extend(level_data.get("reasons", []))

        return {
            "direction":        direction,
            "probability":      round(probability, 1),
            "strength":         strength,
            "bull_prob":        round(bull_pct, 1),
            "bear_prob":        round(bear_pct, 1),
            "bullish_score":    round(bullish, 3),
            "bearish_score":    round(bearish, 3),
            "premium_discount": premium_discount,
            "reasons":          reasons,
            "details":          details,
            "vix_adjusted":     vix > 25,
            "timestamp":        datetime.utcnow().isoformat(),
        }

    # ── Ebene 1: 1H Struktur ───────────────────────────────────────────────

    def _level1_htf_bias(self, bars_1h: list, price: float, vwap: float) -> dict:
        bull = 0.0
        bear = 0.0
        reasons = []

        if len(bars_1h) < 5:
            return {"bull": 0.5, "bear": 0.5,
                    "reasons": ["1H Bars nicht verfügbar"],
                    "name": "HTF Bias (neutral)"}

        df1h = pd.DataFrame(bars_1h[-50:])

        # EMA Stack auf 1H (9/21/50)
        if len(df1h) >= 50:
            ema9  = df1h["close"].ewm(span=9).mean().iloc[-1]
            ema21 = df1h["close"].ewm(span=21).mean().iloc[-1]
            ema50 = df1h["close"].ewm(span=50).mean().iloc[-1]

            if ema9 > ema21 > ema50:
                bull += 1.0
                reasons.append(
                    f"1H EMA Stack bullish: "
                    f"9({ema9:.0f}) > 21({ema21:.0f}) > 50({ema50:.0f})")
            elif ema9 < ema21 < ema50:
                bear += 1.0
                reasons.append(
                    f"1H EMA Stack bearish: "
                    f"9({ema9:.0f}) < 21({ema21:.0f}) < 50({ema50:.0f})")
            elif ema9 > ema21:
                bull += 0.5
                reasons.append(f"1H EMA9({ema9:.0f}) > EMA21({ema21:.0f})")
            else:
                bear += 0.5
                reasons.append(f"1H EMA9({ema9:.0f}) < EMA21({ema21:.0f})")

        # Höhere Hochs / Höhere Tiefs auf 1H (HH/HL = bullish)
        if len(df1h) >= 10:
            recent_highs = df1h["high"].tail(10).values
            recent_lows  = df1h["low"].tail(10).values

            hh = recent_highs[-1] > recent_highs[-5]
            hl = recent_lows[-1]  > recent_lows[-5]
            lh = recent_highs[-1] < recent_highs[-5]
            ll = recent_lows[-1]  < recent_lows[-5]

            if hh and hl:
                bull += 1.0
                reasons.append("1H Struktur: HH + HL — bullische Marktstruktur")
            elif lh and ll:
                bear += 1.0
                reasons.append("1H Struktur: LH + LL — bearische Marktstruktur")
            elif hh:
                bull += 0.4
                reasons.append("1H: Neues Hoch gebildet")
            elif ll:
                bear += 0.4
                reasons.append("1H: Neues Tief gebildet")

        # 1H Momentum (letzte 3 Kerzen)
        if len(df1h) >= 3:
            last3    = df1h["close"].tail(3).values
            momentum = last3[-1] - last3[0]
            if momentum > 0:
                bull += 0.3
                reasons.append(f"1H Momentum: +{momentum:.0f} Punkte")
            else:
                bear += 0.3
                reasons.append(f"1H Momentum: {momentum:.0f} Punkte")

        total = bull + bear
        return {
            "bull":    bull / total if total > 0 else 0.5,
            "bear":    bear / total if total > 0 else 0.5,
            "reasons": reasons,
            "name":    "1H Struktur-Bias",
        }

    # ── Ebene 2: Session Kontext ───────────────────────────────────────────

    def _level2_session_context(self, bars_5m: list, price: float,
                                session_high: float, session_low: float,
                                vwap: float) -> dict:
        bull = 0.0
        bear = 0.0
        reasons = []

        # VWAP Position
        if vwap > 0:
            vwap_dist = price - vwap
            vwap_pct  = (vwap_dist / vwap) * 100
            if price > vwap * 1.001:
                bull += 1.0
                reasons.append(
                    f"Preis über VWAP: +{vwap_dist:.1f} Punkte "
                    f"({vwap_pct:+.2f}%) — Käufer dominieren")
            elif price < vwap * 0.999:
                bear += 1.0
                reasons.append(
                    f"Preis unter VWAP: {vwap_dist:.1f} Punkte "
                    f"({vwap_pct:+.2f}%) — Verkäufer dominieren")
            else:
                reasons.append(f"Preis nahe VWAP ({vwap_dist:+.1f} Pts)")

        # Premium/Discount innerhalb der Session
        if session_high > 0 and session_low > 0:
            session_range = session_high - session_low
            if session_range > 0:
                position = (price - session_low) / session_range
                if position > 0.65:
                    bear += 0.8
                    reasons.append(
                        f"Preis in Premium Zone ({position:.0%} der Session-Range) — "
                        f"Short bevorzugt")
                elif position < 0.35:
                    bull += 0.8
                    reasons.append(
                        f"Preis in Discount Zone ({position:.0%} der Session-Range) — "
                        f"Long bevorzugt")
                else:
                    reasons.append(f"Preis in Equilibrium ({position:.0%} der Range)")

        # Tageszeit
        hour = datetime.now(timezone.utc).hour
        if 13 <= hour < 14:
            if vwap > 0:
                bull += 0.3 if price > vwap else 0.0
                bear += 0.3 if price < vwap else 0.0
            reasons.append("RTH Open (09:30 EST) — Hohe Volatilität")
        elif 14 <= hour < 17:
            reasons.append("NY Morning Prime Time — Beste Trading-Qualität")
        elif 17 <= hour < 18:
            bull *= 0.8
            bear *= 0.8
            reasons.append("NY Lunch (12:00-13:00 EST) — Reduzierte Liquidität")
        elif 18 <= hour < 20:
            reasons.append("NY PM Session (14:00-16:00 EST)")
        else:
            bull *= 0.7
            bear *= 0.7
            reasons.append("Außerhalb RTH — Vorsicht bei Signalen")

        total = bull + bear
        return {
            "bull":    bull / total if total > 0 else 0.5,
            "bear":    bear / total if total > 0 else 0.5,
            "reasons": reasons,
            "name":    "Session Kontext",
        }

    # ── Ebene 3: Key Levels ────────────────────────────────────────────────

    def _level3_key_levels(self, bars_1h: list, bars_5m: list,
                           price: float, session_high: float,
                           session_low: float) -> dict:
        bull = 0.0
        bear = 0.0
        reasons = []

        if not bars_1h or len(bars_1h) < 2:
            return {"bull": 0.5, "bear": 0.5,
                    "reasons": ["Keine 1H Daten für Key Levels"],
                    "name": "Key Levels (neutral)"}

        df1h = pd.DataFrame(bars_1h)

        # Previous Day High/Low
        try:
            now         = datetime.now(timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            if "timestamp" in df1h.columns:
                df1h["ts"] = pd.to_datetime(df1h["timestamp"])
                yesterday  = df1h[df1h["ts"] < today_start]

                if len(yesterday) >= 8:
                    pdh = yesterday["high"].max()
                    pdl = yesterday["low"].min()

                    if price > pdh:
                        bull += 0.8
                        reasons.append(
                            f"Preis über PDH ({pdh:.2f}) — Breakout bullish")
                    elif price < pdl:
                        bear += 0.8
                        reasons.append(
                            f"Preis unter PDL ({pdl:.2f}) — Breakdown bearish")
                    elif price > (pdh + pdl) / 2:
                        bull += 0.4
                        reasons.append(
                            f"Preis über PDH/PDL Mittelpunkt ({(pdh+pdl)/2:.2f}) — leicht bullish")
                    else:
                        bear += 0.4
                        reasons.append(
                            f"Preis unter PDH/PDL Mittelpunkt — leicht bearish")
        except Exception as exc:
            reasons.append(f"PDH/PDL: {str(exc)[:30]}")

        # Opening Range Breakout (erste 30 Minuten RTH = 13:30 UTC)
        try:
            if bars_5m and len(bars_5m) >= 6:
                df5 = pd.DataFrame(bars_5m)
                if "timestamp" in df5.columns:
                    df5["ts"] = pd.to_datetime(df5["timestamp"])
                    now             = datetime.now(timezone.utc)
                    today_rth_start = now.replace(
                        hour=13, minute=30, second=0, microsecond=0)
                    today_rth_end   = today_rth_start + timedelta(minutes=30)

                    orb_bars = df5[
                        (df5["ts"] >= today_rth_start) &
                        (df5["ts"] <= today_rth_end)
                    ]

                    if len(orb_bars) >= 3:
                        orb_high = orb_bars["high"].max()
                        orb_low  = orb_bars["low"].min()

                        if price > orb_high:
                            bull += 1.0
                            reasons.append(
                                f"Preis über ORB High ({orb_high:.2f}) — Bullischer Breakout")
                        elif price < orb_low:
                            bear += 1.0
                            reasons.append(
                                f"Preis unter ORB Low ({orb_low:.2f}) — Bearischer Breakout")
                        else:
                            midpoint = (orb_high + orb_low) / 2
                            if price > midpoint:
                                bull += 0.3
                            else:
                                bear += 0.3
                            reasons.append(
                                f"Preis innerhalb ORB ({orb_low:.2f}-{orb_high:.2f})")
        except Exception as exc:
            reasons.append(f"ORB: {str(exc)[:30]}")

        # Session Extremes
        if session_high > 0 and session_low > 0:
            atr_approx = (session_high - session_low) * 0.1
            if price >= session_high - atr_approx:
                bear += 0.6
                reasons.append(
                    f"Preis nahe Session High ({session_high:.2f}) — Widerstand erwartet")
            elif price <= session_low + atr_approx:
                bull += 0.6
                reasons.append(
                    f"Preis nahe Session Low ({session_low:.2f}) — Support erwartet")

        total = bull + bear
        return {
            "bull":    bull / total if total > 0 else 0.5,
            "bear":    bear / total if total > 0 else 0.5,
            "reasons": reasons,
            "name":    "Key Levels (PDH/PDL, ORB)",
        }

    # ── Ebene 4: LTF Bestätigung ───────────────────────────────────────────

    def _level4_ltf_confirmation(self, bars_5m: list, bars_15m: list,
                                  price: float, vwap: float) -> dict:
        bull = 0.0
        bear = 0.0
        reasons = []

        # 5m EMA
        if len(bars_5m) >= 21:
            df5     = pd.DataFrame(bars_5m[-50:])
            ema9_5  = df5["close"].ewm(span=9).mean().iloc[-1]
            ema21_5 = df5["close"].ewm(span=21).mean().iloc[-1]

            if ema9_5 > ema21_5:
                bull += 0.8
                reasons.append(
                    f"5m EMA9({ema9_5:.0f}) > EMA21({ema21_5:.0f}) — Short-Term bullish")
            else:
                bear += 0.8
                reasons.append(
                    f"5m EMA9({ema9_5:.0f}) < EMA21({ema21_5:.0f}) — Short-Term bearish")

        # 15m EMA
        if len(bars_15m) >= 21:
            df15     = pd.DataFrame(bars_15m[-50:])
            ema9_15  = df15["close"].ewm(span=9).mean().iloc[-1]
            ema21_15 = df15["close"].ewm(span=21).mean().iloc[-1]

            if ema9_15 > ema21_15:
                bull += 0.8
                reasons.append(
                    f"15m EMA9({ema9_15:.0f}) > EMA21({ema21_15:.0f}) — Mid-Term bullish")
            else:
                bear += 0.8
                reasons.append(
                    f"15m EMA9({ema9_15:.0f}) < EMA21({ema21_15:.0f}) — Mid-Term bearish")

        # 5m Volumen-Momentum
        if len(bars_5m) >= 10:
            df5v = pd.DataFrame(bars_5m[-10:])
            if "volume" in df5v.columns:
                recent_vol = df5v["volume"].tail(5).mean()
                prev_vol   = df5v["volume"].head(5).mean()
                if recent_vol > prev_vol * 1.2:
                    last_close = df5v["close"].iloc[-1]
                    prev_close = df5v["close"].iloc[-5]
                    if last_close > prev_close:
                        bull += 0.4
                        reasons.append("5m: Erhöhtes Kaufvolumen")
                    else:
                        bear += 0.4
                        reasons.append("5m: Erhöhtes Verkaufsvolumen")

        total = bull + bear
        return {
            "bull":    bull / total if total > 0 else 0.5,
            "bear":    bear / total if total > 0 else 0.5,
            "reasons": reasons,
            "name":    "LTF Bestätigung (5m/15m)",
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _premium_discount(self, price: float,
                          session_high: float, session_low: float) -> str:
        if session_high == 0 or session_low == 0:
            return "UNBEKANNT"
        session_range = session_high - session_low
        if session_range == 0:
            return "EQUILIBRIUM"
        position = (price - session_low) / session_range
        if position > 0.618:
            return "PREMIUM"
        elif position < 0.382:
            return "DISCOUNT"
        return "EQUILIBRIUM"

    def _neutral_bias(self, reason: str) -> dict:
        return {
            "direction":        "NEUTRAL",
            "probability":      50.0,
            "strength":         "WEAK",
            "bull_prob":        50.0,
            "bear_prob":        50.0,
            "bullish_score":    0.0,
            "bearish_score":    0.0,
            "premium_discount": "UNBEKANNT",
            "reasons":          [reason],
            "details":          {},
            "vix_adjusted":     False,
            "timestamp":        datetime.utcnow().isoformat(),
        }
