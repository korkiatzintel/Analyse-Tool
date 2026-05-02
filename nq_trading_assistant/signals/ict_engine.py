from datetime import datetime, timezone

import pandas as pd
from smartmoneyconcepts import smc


class ICTSignalEngine:
    """
    ICT-basierte Signale für NQ Futures.
    Ergänzt die bestehende Signal Engine um FVG, Order Blocks,
    Market Structure (BOS/CHoCH) und Killzone-Filter.
    """

    KILLZONES = {
        "LONDON_OPEN": (8,  0,  9,  0),
        "NY_AM":       (14, 0, 15, 0),
        "NY_PM":       (18, 0, 19, 0),
    }

    def get_active_killzone(self) -> str | None:
        now = datetime.now(timezone.utc)
        current_minutes = now.hour * 60 + now.minute
        for name, (sh, sm, eh, em) in self.KILLZONES.items():
            if sh * 60 + sm <= current_minutes <= eh * 60 + em:
                return name
        return None

    def analyze(self, bars_5m: list, bars_15m: list, bars_1h: list | None = None) -> dict:
        """
        Berechne ICT Signale aus OHLCV Daten.
        bars_1h wird für HTF Bias (Higher Timeframe) genutzt.
        Gibt immer ein dict zurück — Fehler einzelner Indikatoren
        unterbrechen die anderen nicht.
        """
        if len(bars_5m) < 10:
            killzone = self.get_active_killzone()
            return {
                "error":           "Zu wenige Bars",
                "active_killzone": killzone,
                "killzone_bonus":  1.3 if killzone else 1.0,
                "ict_score":       0.0,
                "ict_reasons":     [],
            }

        df5  = self._to_df(bars_5m[-100:])
        df15 = self._to_df(bars_15m[-50:]) if len(bars_15m or []) >= 10 else None
        df1h = self._to_df(bars_1h[-200:]) if len(bars_1h or []) >= 20 else None

        signals: dict = {}

        # ── 1. Fair Value Gaps (5m) ───────────────────────────────────────
        try:
            fvg_5m = smc.fvg(df5, join_consecutive=True)
            if fvg_5m is not None and not fvg_5m.empty:
                recent_fvg = fvg_5m[fvg_5m["FVG"].notna()].tail(3)
                signals["fvg_levels"] = recent_fvg.to_dict("records")
        except Exception as e:
            signals["fvg_error"] = str(e)

        # ── 2. Order Blocks (5m) ──────────────────────────────────────────
        try:
            swing_hl = smc.swing_highs_lows(df5, swing_length=10)
            ob = smc.ob(df5, swing_hl)
            if ob is not None and not ob.empty:
                active_obs = ob[ob["OB"].notna()].tail(3)
                signals["order_blocks"] = active_obs.to_dict("records")
        except Exception as e:
            signals["ob_error"] = str(e)

        # ── 3. Market Structure BOS / CHoCH (15m) ────────────────────────
        if df15 is not None:
            try:
                swing_hl_15 = smc.swing_highs_lows(df15, swing_length=10)
                bos = smc.bos_choch(df15, swing_hl_15)
                if bos is not None and not bos.empty:
                    recent_bos = bos[bos["BOS"].notna() | bos["CHOCH"].notna()].tail(2)
                    if not recent_bos.empty:
                        last = recent_bos.iloc[-1]
                        is_bos   = pd.notna(last.get("BOS"))
                        is_choch = pd.notna(last.get("CHOCH"))
                        if is_bos or is_choch:
                            val = last.get("BOS", last.get("CHOCH", 0))
                            signals["market_structure"] = {
                                "type":      "BOS" if is_bos else "CHOCH",
                                "direction": "BULLISH" if (val or 0) > 0 else "BEARISH",
                                "level":     float(last.get("Level", 0) or 0),
                            }
            except Exception as e:
                signals["bos_error"] = str(e)

        # ── 4. HTF Bias (1h) — BOS/CHoCH + FVG auf Stunden-Chart ─────────
        if df1h is not None:
            try:
                swing_hl_1h = smc.swing_highs_lows(df1h, swing_length=10)
                bos_1h = smc.bos_choch(df1h, swing_hl_1h)
                if bos_1h is not None and not bos_1h.empty:
                    recent_1h = bos_1h[bos_1h["BOS"].notna() | bos_1h["CHOCH"].notna()].tail(3)
                    if not recent_1h.empty:
                        last_1h = recent_1h.iloc[-1]
                        is_bos_1h   = pd.notna(last_1h.get("BOS"))
                        is_choch_1h = pd.notna(last_1h.get("CHOCH"))
                        if is_bos_1h or is_choch_1h:
                            val_1h = last_1h.get("BOS", last_1h.get("CHOCH", 0))
                            signals["htf_bias"] = {
                                "type":      "BOS" if is_bos_1h else "CHOCH",
                                "direction": "BULLISH" if (val_1h or 0) > 0 else "BEARISH",
                                "level":     float(last_1h.get("Level", 0) or 0),
                                "timeframe": "1h",
                            }
            except Exception as e:
                signals["htf_bias_error"] = str(e)

            try:
                fvg_1h = smc.fvg(df1h, join_consecutive=True)
                if fvg_1h is not None and not fvg_1h.empty:
                    recent_fvg_1h = fvg_1h[fvg_1h["FVG"].notna()].tail(3)
                    signals["fvg_1h"] = recent_fvg_1h.to_dict("records")
            except Exception as e:
                signals["fvg_1h_error"] = str(e)

        # ── 5. Liquidity Levels (5m) ──────────────────────────────────────
        try:
            swing_hl_5 = smc.swing_highs_lows(df5, swing_length=5)
            liq = smc.liquidity(df5, swing_hl_5)
            if liq is not None and not liq.empty:
                active_liq = liq[liq["Liquidity"].notna()].tail(5)
                signals["liquidity_levels"] = active_liq.to_dict("records")
        except Exception as e:
            signals["liq_error"] = str(e)

        # ── 6. Killzone ───────────────────────────────────────────────────
        killzone = self.get_active_killzone()
        signals["active_killzone"] = killzone
        signals["killzone_bonus"]  = 1.3 if killzone else 1.0

        # ── 7. ICT Score ──────────────────────────────────────────────────
        score   = 0.0
        reasons = []

        if killzone:
            score += 0.25
            reasons.append(f"✅ Aktive Killzone: {killzone} (+25%)")

        ms = signals.get("market_structure", {})
        if ms.get("type") == "CHOCH":
            score += 0.30
            reasons.append(f"✅ CHoCH {ms.get('direction')} erkannt (+30%)")
        elif ms.get("type") == "BOS":
            score += 0.20
            reasons.append(f"✅ BOS {ms.get('direction')} erkannt (+20%)")

        htf = signals.get("htf_bias", {})
        if htf:
            score += 0.20
            reasons.append(
                f"✅ HTF 1h Bias: {htf.get('type')} {htf.get('direction')} (+20%)"
            )

        obs = signals.get("order_blocks", [])
        if obs:
            score += 0.15
            reasons.append(f"✅ {len(obs)} Order Block(s) aktiv (+15%)")

        fvgs = signals.get("fvg_levels", [])
        if fvgs:
            score += 0.10
            reasons.append(f"✅ {len(fvgs)} FVG(s) 5m erkannt (+10%)")

        fvgs_1h = signals.get("fvg_1h", [])
        if fvgs_1h:
            score += 0.10
            reasons.append(f"✅ {len(fvgs_1h)} FVG(s) 1h erkannt (+10%)")

        signals["ict_score"]   = min(score, 1.0)
        signals["ict_reasons"] = reasons
        return signals

    @staticmethod
    def _to_df(bars: list) -> pd.DataFrame:
        df = pd.DataFrame(bars)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        return df.reset_index(drop=True)
