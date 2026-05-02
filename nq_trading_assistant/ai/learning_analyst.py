import json
import logging
from pathlib import Path
from datetime import datetime


class LearningAnalyst:
    """
    Wird manuell einmal täglich aufgerufen.
    Analysiert simulated_trades.json und gibt neue Signal-Gewichtungen zurück.
    """

    _SYSTEM_PROMPT = """Du bist ein quantitativer Trading-Analyst spezialisiert auf \
NQ Futures (E-Mini Nasdaq-100). Du analysierst simulierte Trade-Daten um \
Signal-Gewichtungen zu optimieren.

Antworte AUSSCHLIESSLICH in folgendem JSON-Format ohne Markdown oder Erklärungen:
{
  "analyse": {
    "zusammenfassung": "2-3 Sätze Gesamtbewertung",
    "staerken": ["Stärke 1", "Stärke 2"],
    "schwaechen": ["Schwäche 1", "Schwäche 2"],
    "muster": ["Erkanntes Muster 1", "Erkanntes Muster 2"]
  },
  "signal_bewertung": {
    "SIGNAL_NAME": {
      "win_rate": 0.0,
      "empfehlung": "STAERKEN/REDUZIEREN/BEIBEHALTEN",
      "begruendung": "Kurze Begründung"
    }
  },
  "neue_gewichtungen": {
    "SIGNAL_NAME": 1.0
  },
  "kontext_anpassungen": {
    "_vix_high_penalty": 0.8,
    "_vix_extreme_penalty": 0.5,
    "_time_open_bonus": 1.2,
    "_time_close_penalty": 0.9
  },
  "handlungsempfehlungen": [
    "Konkrete Empfehlung 1",
    "Konkrete Empfehlung 2",
    "Konkrete Empfehlung 3"
  ],
  "naechste_analyse_in": "Empfehlung wann nächste Analyse sinnvoll ist"
}

Gewichtungen: 0.5 = stark reduzieren, 1.0 = neutral, 1.5 = stark bevorzugen.
Maximale Änderung pro Signal pro Analyse: ±0.3 (keine extremen Sprünge)."""

    def __init__(self, analyst_instance):
        self._analyst       = analyst_instance
        self._log           = logging.getLogger(__name__)
        self._analysis_file = Path(__file__).parent.parent / "logs" / "learning_analysis.json"

    def run_daily_analysis(self, simulator, strategy_config=None) -> dict:
        stats           = simulator.get_stats()
        closed_trades   = simulator.get_closed_trades(limit=15)
        current_weights = simulator.get_weights()
        self._strategy_config = strategy_config

        if stats["total"] < 5:
            return {
                "status":  "insufficient_data",
                "message": f"Zu wenige Trades ({stats['total']}). Mindestens 5 benötigt.",
            }

        # Compact trade lines: one per row to minimise token count
        trade_summary = [
            f"{t['direction']} {t.get('confidence', 0):.0%} "
            f"→ {t.get('outcome', '?')} "
            f"({', '.join(t.get('active_signals', [])[:2])})"
            for t in closed_trades[:15]
        ]

        weights_simple = {
            k: v for k, v in current_weights.items()
            if not k.startswith("_") and isinstance(v, (int, float))
        }

        # ── CALL 1: Diagnose (text, max 400 tokens) ───────────────────────
        prompt_1 = (
            f"NQ Futures Trading System Analyse.\n\n"
            f"Stats: {stats['total']} Trades, Win-Rate {stats['win_rate']}%\n"
            f"Win nach Zeit: {stats.get('win_rate_by_time', {})}\n"
            f"Win nach VIX: {stats.get('win_rate_by_regime', {})}\n"
            f"Beste Signale: {stats.get('best_signal_types', {})}\n\n"
            f"Letzte 15 Trades (Richtung/Konfidenz/Ergebnis/Signale):\n"
            + "\n".join(trade_summary)
            + "\n\nSchreibe 3 kurze Stichpunkte was verbessert werden muss. Auf Deutsch."
        )
        diagnose = self._call_analyst(prompt_1, max_tokens=400)

        # ── CALL 2: Gewichtungen (JSON-only, max 300 tokens) ──────────────
        weights_list = "\n".join(f"- {k}: {v:.2f}" for k, v in weights_simple.items())
        best_signals = stats.get("best_signal_types", {})
        worst_3      = dict(sorted(best_signals.items(), key=lambda x: x[1])[:3])

        prompt_2 = (
            f"Basierend auf: Win-Rate {stats['win_rate']}%, "
            f"beste Signale: {best_signals}, "
            f"schlechteste: {worst_3}, "
            f"Diagnose: {diagnose[:300]}\n\n"
            f"Aktuelle Gewichtungen:\n{weights_list}\n\n"
            "Gib NUR ein JSON-Objekt zurück (kein Text davor/danach):\n"
            '{"MULTI_TF_BIAS": 1.0, "FAIR_VALUE_GAP": 1.0, "VIX_REGIME": 1.0, '
            '"OVERNIGHT_GAP": 1.0, "EMA_TREND": 1.0, "VWAP_POSITION": 1.0, '
            '"RSI_EXTREME": 1.0, "MEAN_REVERSION": 1.0, "SESSION_LEVELS": 1.0}\n\n'
            "Ändere nur Werte die laut den Daten angepasst werden müssen. "
            "Range: 0.5–1.5. Max Änderung: 0.25 pro Signal."
        )
        weights_raw = self._call_analyst(prompt_2, max_tokens=300)

        # ── CALL 3: Handlungsempfehlungen (text, max 300 tokens) ──────────
        # Killzone win-rate aus Trade-Daten ableiten
        killzone_trades = [
            t for t in closed_trades
            if "ICT_CONFLUENCE" in t.get("active_signals", [])
        ]
        kz_total = len(killzone_trades)
        kz_wins  = sum(1 for t in killzone_trades if t.get("outcome") == "WIN")
        kz_info  = (
            f"ICT_CONFLUENCE Trades: {kz_total}, Win-Rate: "
            f"{round(kz_wins/kz_total*100,1) if kz_total else 'n/a'}%"
        )

        prompt_3 = (
            f"NQ Trading System, Win-Rate {stats['win_rate']}%.\n"
            f"Diagnose: {diagnose[:200]}\n"
            f"ICT Killzone-Daten: {kz_info}\n\n"
            "Gib genau 3 konkrete Handlungsempfehlungen auf Deutsch.\n"
            "Bewerte auch ob ICT Killzone-Filter strenger oder lockerer "
            "sein soll (Trades innerhalb vs. außerhalb Killzones).\n"
            "Format: Nummerierte Liste, je max 1 Satz."
        )
        empfehlungen_raw = self._call_analyst(prompt_3, max_tokens=300)

        # ── Parse Gewichtungen ────────────────────────────────────────────
        neue_gewichtungen: dict = {}
        try:
            w = weights_raw.strip().replace("```json", "").replace("```", "").strip()
            start = w.find("{")
            end   = w.rfind("}") + 1
            if start >= 0 and end > start:
                w = w[start:end]
            neue_gewichtungen = json.loads(w)
        except Exception as e:
            self._log.error("Gewichtungen Parse Error: %s | raw: %s", e, weights_raw[:200])

        # ── Parse Empfehlungen ────────────────────────────────────────────
        empfehlungen = []
        for line in empfehlungen_raw.strip().splitlines():
            clean = line.strip().lstrip("0123456789.-) ").strip()
            if clean:
                empfehlungen.append(clean)

        # ── Signal-Bewertung aus Stats ────────────────────────────────────
        signal_bewertung: dict = {}
        for sig, wr in best_signals.items():
            signal_bewertung[sig] = {
                "win_rate":    wr,
                "empfehlung":  "STAERKEN" if wr >= 60 else ("REDUZIEREN" if wr <= 40 else "BEIBEHALTEN"),
                "begruendung": f"Win-Rate {wr}% basierend auf {stats['total']} Trades",
            }

        # ── Baue finales Result ───────────────────────────────────────────
        result = {
            "analyse": {
                "zusammenfassung": diagnose[:500],
                "staerken":        [],
                "schwaechen":      [],
                "muster":          [],
            },
            "signal_bewertung":      signal_bewertung,
            "neue_gewichtungen":     neue_gewichtungen,
            "kontext_anpassungen":   {},
            "handlungsempfehlungen": empfehlungen[:3],
            "naechste_analyse_in":   "Nach 20–30 weiteren Trades",
        }

        # ── Wende Gewichtungen an ─────────────────────────────────────────
        if neue_gewichtungen:
            simulator.apply_weights(neue_gewichtungen)
            self._log.info(
                "Gewichtungen aktualisiert: %d Signale angepasst",
                len(neue_gewichtungen),
            )

        # ── Speichere Analyse ─────────────────────────────────────────────
        analysis_record = {
            "timestamp":       datetime.utcnow().isoformat(),
            "stats_snapshot":  stats,
            "result":          result,
            "trades_analyzed": stats["total"],
            "previous_weights": weights_simple,
        }

        history: list = []
        try:
            history = json.loads(self._analysis_file.read_text(encoding="utf-8"))
        except Exception:
            pass
        history.append(analysis_record)
        self._analysis_file.write_text(
            json.dumps(history[-10:], indent=2, default=str),
            encoding="utf-8",
        )

        self._log.info(
            "Learning Analysis fertig: %d Trades, Win-Rate %s%%, "
            "%d Gewichtungen aktualisiert",
            stats["total"], stats["win_rate"], len(neue_gewichtungen),
        )

        return {"status": "success", "result": result, "stats": stats}

    def _call_analyst(self, prompt: str, max_tokens: int) -> str:
        """Route to the available KI provider."""
        if hasattr(self._analyst, "run_daily_analysis"):
            return self._analyst.run_daily_analysis(prompt, max_tokens=max_tokens)
        if hasattr(self._analyst, "_anthropic"):
            resp = self._analyst._anthropic.messages.create(
                model      = "claude-sonnet-4-6",
                max_tokens = max_tokens,
                system     = "Du bist ein NQ Futures Trading-Analyst. Antworte auf Deutsch.",
                messages   = [{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        raise Exception("Kein kompatibler KI-Client verfügbar")

    def _run_diagnose(self, stats: dict, trades: list) -> str:
        """Call 1: short German text diagnosis, max 500 tokens."""
        trade_summary = json.dumps(
            [
                {
                    "richtung":  t["direction"],
                    "ergebnis":  t["outcome"],
                    "pnl":       t.get("pnl_points", 0),
                    "signale":   t["active_signals"],
                    "vix":       t.get("vix_regime", "normal"),
                    "zeit":      t.get("time_of_day", "RTH_MID"),
                }
                for t in trades
            ],
            indent=2,
            ensure_ascii=False,
        )
        prompt = (
            f"Analysiere diese NQ Futures Trade-Daten kurz:\n\n"
            f"Win-Rate: {stats['win_rate']}% | Trades: {stats['total']} | "
            f"Gesamt P&L: ${stats['total_pnl_usd']:+.0f}\n\n"
            f"Letzte {len(trades)} Trades:\n{trade_summary}\n\n"
            f"Antworte in max 200 Wörtern auf Deutsch was gut/schlecht läuft."
        )
        if hasattr(self._analyst, "run_daily_analysis"):
            return self._analyst.run_daily_analysis(prompt, max_tokens=500)
        elif hasattr(self._analyst, "_anthropic"):
            resp = self._analyst._anthropic.messages.create(
                model      = "claude-sonnet-4-6",
                max_tokens = 500,
                system     = "Du bist ein NQ Futures Trading-Analyst. Antworte auf Deutsch.",
                messages   = [{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        return "Keine Diagnose verfügbar."

    def _build_params_prompt(
        self,
        diagnose:     str,
        stats:        dict,
        weights_json: str,
        signals_json: str,
        time_json:    str,
        regime_json:  str,
    ) -> str:
        """Call 2: prompt for pure JSON parameter-update response."""
        return (
            f"Basierend auf dieser Analyse:\n{diagnose}\n\n"
            f"Und diesen Stats:\n"
            f"Win-Rate: {stats['win_rate']}%\n"
            f"Beste Signale:\n{signals_json}\n"
            f"Win-Rate nach Zeit:\n{time_json}\n"
            f"Win-Rate nach VIX:\n{regime_json}\n\n"
            f"Aktuelle Gewichtungen:\n{weights_json}\n\n"
            "Gib NUR dieses JSON zurück (kein anderer Text):\n"
            "{\n"
            '  "analyse": {\n'
            '    "zusammenfassung": "...",\n'
            '    "staerken": ["..."],\n'
            '    "schwaechen": ["..."],\n'
            '    "muster": ["..."]\n'
            "  },\n"
            '  "signal_bewertung": {\n'
            '    "SIGNAL_NAME": {\n'
            '      "win_rate": 0.0,\n'
            '      "empfehlung": "STAERKEN/REDUZIEREN/BEIBEHALTEN",\n'
            '      "begruendung": "..."\n'
            "    }\n"
            "  },\n"
            '  "neue_gewichtungen": {"SIGNAL_NAME": 1.0},\n'
            '  "kontext_anpassungen": {},\n'
            '  "handlungsempfehlungen": ["..."],\n'
            '  "naechste_analyse_in": "..."\n'
            "}"
        )

    def _build_analysis_prompt(self, stats: dict, trades: list, weights: dict) -> str:
        cfg = getattr(self, "_strategy_config", None)
        params_json = (
            json.dumps(cfg.get_all_for_gemini(), indent=2, ensure_ascii=False)
            if cfg else "{}"
        )

        trade_details = [
            {
                "id":            t["trade_id"],
                "richtung":      t["direction"],
                "konfidenz":     f"{t['confidence']:.0%}",
                "aktive_signale": t["active_signals"],
                "ergebnis":      t["outcome"],
                "pnl_punkte":    t.get("pnl_points", 0),
                "dauer_minuten": t.get("duration_minutes", 0),
                "vix_regime":    t.get("vix_regime", "normal"),
                "tageszeit":     t.get("time_of_day", "RTH_MID"),
                "exit_grund":    t.get("exit_reason", ""),
                "bias_richtung": t.get("bias_direction", ""),
                "bias_staerke":  t.get("bias_probability", 0),
            }
            for t in trades[:30]
        ]

        # Pre-compute all JSON serializations before entering the f-string
        trade_details_json = json.dumps(trade_details, indent=2, ensure_ascii=False)
        stats_signals_json = json.dumps(stats.get("best_signal_types", {}), indent=2, ensure_ascii=False)
        stats_regime_json  = json.dumps(stats.get("win_rate_by_regime", {}), indent=2, ensure_ascii=False)
        stats_time_json    = json.dumps(stats.get("win_rate_by_time",   {}), indent=2, ensure_ascii=False)
        weights_clean      = {k: v for k, v in weights.items() if not k.startswith("_")}
        weights_json       = json.dumps(weights_clean, indent=2)
        n_trades           = len(trade_details)

        # ICT stats
        ict_killzone_json = json.dumps(stats.get("win_rate_by_killzone",         {}), indent=2, ensure_ascii=False)
        ict_ms_json       = json.dumps(stats.get("win_rate_by_market_structure", {}), indent=2, ensure_ascii=False)
        ict_score_json    = json.dumps(stats.get("win_rate_by_ict_score",        {}), indent=2, ensure_ascii=False)
        ict_ob_json       = json.dumps(stats.get("win_rate_by_order_blocks",     {}), indent=2, ensure_ascii=False)

        w_multi    = weights.get("MULTI_TF_BIAS",        1.0)
        w_fvg      = weights.get("FAIR_VALUE_GAP",       1.0)
        w_vix      = weights.get("VIX_REGIME",           1.0)
        w_gap      = weights.get("OVERNIGHT_GAP",        1.0)
        w_ema      = weights.get("EMA_TREND",            1.0)
        w_vwap     = weights.get("VWAP_POSITION",        1.0)
        w_rsi      = weights.get("RSI_EXTREME",          1.0)
        w_mean     = weights.get("MEAN_REVERSION",       1.0)
        w_vhp      = weights.get("_vix_high_penalty",    0.8)
        w_vep      = weights.get("_vix_extreme_penalty", 0.5)
        w_tob      = weights.get("_time_open_bonus",     1.2)
        w_tcp      = weights.get("_time_close_penalty",  0.9)

        total      = stats["total"]
        win_rate   = stats["win_rate"]
        avg_pts    = stats["avg_pnl_points"]
        avg_usd    = stats["avg_pnl_usd"]
        total_usd  = stats["total_pnl_usd"]
        wins       = stats["wins"]
        losses     = stats["losses"]
        timeouts   = stats["timeouts"]

        return f"""Du bist ein erfahrener quantitativer NQ Futures Trading-Analyst.
Du analysierst ein automatisches Trading-System und optimierst dessen \
Signal-Gewichtungen basierend auf realen Simulationsdaten.

════════════════════════════════════════
VOLLSTÄNDIGE STRATEGIEBESCHREIBUNG
════════════════════════════════════════

Das System handelt E-Mini Nasdaq-100 Futures (NQ) im Day-Trading.
Tick-Größe: 0.25 Punkte = $5 pro Tick (NQ) / $0.50 (MNQ)

HANDELSSYSTEM-LOGIK:
1. Zuerst wird ein übergeordneter MARKT-BIAS berechnet (LONG/SHORT/NEUTRAL)
   basierend auf: VWAP-Position, EMA Stack 15m/5m, Session-Position, Momentum

2. Nur Trades IN Bias-Richtung werden akzeptiert

3. Signal Engine bewertet 8 verschiedene Signaltypen (siehe unten)

4. Trades werden nur eröffnet wenn Konfidenz >= 65% UND mindestens
   2 Signale übereinstimmen

5. Risk Management: SL = 0.5 ATR, TP1 = 1.5x Risiko, TP2 = 2.5x Risiko
   Timeout: Trade wird nach 4 Stunden geschlossen

SIGNAL-TYPEN (Beschreibung für deine Analyse):

MULTI_TF_BIAS:
- Prüft ob EMA9 > EMA21 auf BEIDEN Zeitrahmen (5min UND 15min)
- Bullish wenn beide EMAs ausgerichtet sind + Preis über VWAP
- Stärkster Trend-Indikator im System
- Gewichtung aktuell: {w_multi:.2f}

FAIR_VALUE_GAP (FVG):
- 3-Kerzen-Muster: mittlere Kerze springt so stark dass eine
  Preislücke entsteht (Institutioneller Footprint)
- Signal bei Retest dieser Lücke (Preis kehrt zurück)
- Sehr verlässlich in trendfolgenden Märkten
- Gewichtung aktuell: {w_fvg:.2f}

VIX_REGIME:
- VIX < 15: Ruhiger Markt (Trend-Setups bevorzugen)
- VIX 15-25: Normaler Markt (alle Setups)
- VIX 25-30: Volatiler Markt (nur High-Konfidenz)
- VIX > 30: Extremer Markt (keine Signale)
- Gewichtung aktuell: {w_vix:.2f}

OVERNIGHT_GAP:
- Gap > 0.3% zwischen gestrigem Close und heutigem Open
- Trade in Richtung Gap-Fill in ersten 90 Minuten
- Funktioniert gut bei klaren Richtungs-Gaps
- Gewichtung aktuell: {w_gap:.2f}

EMA_TREND:
- EMA9 vs EMA21 Kreuzung auf 30-Sekunden-Bars
- Kurzfristiger Trend-Indikator
- Bestätigt oder widerspricht dem übergeordneten Bias
- Gewichtung aktuell: {w_ema:.2f}

VWAP_POSITION:
- VWAP = Volume Weighted Average Price (institutioneller Fairwert)
- Preis über VWAP = Käufer dominieren
- Preis unter VWAP = Verkäufer dominieren
- Abstand zum VWAP zeigt Momentum-Stärke
- Gewichtung aktuell: {w_vwap:.2f}

RSI_EXTREME:
- RSI < 30 = Überverkauft (Long-Setup)
- RSI > 70 = Überkauft (Short-Setup)
- Kontra-Trend Signal — funktioniert besonders bei hohem VIX
- Gewichtung aktuell: {w_rsi:.2f}

MEAN_REVERSION:
- Preis weit von VWAP entfernt (> 0.8 ATR)
- Erwartung der Rückkehr zum Mittelwert
- Funktioniert in seitwärts tendierenden Märkten gut
- Gewichtung aktuell: {w_mean:.2f}

KONTEXT-MODIFIKATOREN:
- VIX hoch Abzug: {w_vhp:.2f} (Konfidenz × dieser Faktor bei VIX 25-30)
- VIX extrem Abzug: {w_vep:.2f} (bei VIX > 30)
- RTH Open Bonus: {w_tob:.2f} (erste 30 Min mehr Gewicht)
- RTH Close Abzug: {w_tcp:.2f} (letzte 30 Min weniger)

NQ-SPEZIFISCHES MARKTWISSEN:
- NQ ist volatiler als ES (höherer ATR)
- Beste Trading-Zeit: 09:30-11:00 ET und 13:30-15:00 ET
- FOMC, CPI, NFP Tage: Vermeidung empfohlen
- NQ reagiert stark auf Tech-Sentiment und Zinsen
- Typische Win-Rate bei guten Setups: 60-65%
- Profitable Systeme brauchen mindestens 1:1.5 Risk/Reward

════════════════════════════════════════
AKTUELLE PERFORMANCE-DATEN
════════════════════════════════════════

GESAMTSTATISTIK:
- Trades gesamt: {total}
- Win-Rate: {win_rate}% (Ziel: >62%)
- Avg P&L: {avg_pts:+.2f} Punkte / ${avg_usd:+.0f} pro Trade
- Gesamt P&L: ${total_usd:+.0f}
- Wins: {wins} | Losses: {losses} | Timeouts: {timeouts}

WIN-RATE NACH SIGNAL-TYP:
{stats_signals_json}

WIN-RATE NACH VIX-REGIME:
{stats_regime_json}

WIN-RATE NACH TAGESZEIT:
{stats_time_json}

════════════════════════════════════════
ICT PERFORMANCE ANALYSE
════════════════════════════════════════

WIN-RATE IN KILLZONE vs. AUSSERHALB:
{ict_killzone_json}
→ Idealerweise sollte IN_KILLZONE deutlich höher sein als OUTSIDE_KILLZONE.
  Wenn nicht: Killzone-Filter muss verstärkt werden.

WIN-RATE NACH MARKET STRUCTURE BEIM ENTRY:
{ict_ms_json}
→ CHoCH Entries sollten höhere Win-Rate haben als BOS oder kein Signal.

WIN-RATE NACH ICT CONFLUENCE SCORE:
{ict_score_json}
→ Höherer ICT Score sollte mit höherer Win-Rate korrelieren.
  Wenn nicht: ICT Gewichtung überdenken.

WIN-RATE MIT vs. OHNE ORDER BLOCKS:
{ict_ob_json}
→ Trades mit aktivem Order Block sollten besser performen.

AKTUELLE SIGNAL-GEWICHTUNGEN:
{weights_json}

LETZTE {n_trades} TRADES (detailliert):
{trade_details_json}

════════════════════════════════════════
ANPASSBARE STRATEGIE-PARAMETER
════════════════════════════════════════

Du darfst folgende Parameter anpassen um die Strategie zu verbessern.
Beachte die angegebenen Ranges und maximalen Änderungen pro Update.
Begründe JEDE Änderung mit konkreten Daten aus den Trades.

{params_json}

WICHTIGE REGELN FÜR PARAMETER-ÄNDERUNGEN:
1. Ändere nie mehr als 3-4 Parameter pro Analyse
2. Fokussiere auf die Parameter mit dem größten Hebel
3. Sei konservativ — lieber kleine sichere Schritte
4. Begründe jeden Schritt mit den Trade-Daten
5. KERNREGELN dürfen NICHT geändert werden

════════════════════════════════════════
DEINE ANALYSE-AUFGABE
════════════════════════════════════════

Analysiere die Daten und beantworte:

1. Welche Signale performen gut/schlecht und WARUM
   (beziehe NQ-Marktlogik ein)

2. Gibt es Muster? (z.B. "FVG funktioniert nur bei normalem VIX",
   "Trades am RTH Open haben höhere Win-Rate")

3. Welche Gewichtungen sollen angepasst werden?
   Sei konservativ: max ±0.25 Änderung pro Signal pro Analyse
   Begründe jede Änderung mit konkreten Daten aus den Trades

4. Was soll der Trader konkret anders machen?

5. ICT-SPEZIFISCHE FRAGEN (beantworte alle vier):
   a) Lohnt sich der Killzone-Filter? (Win-Rate Differenz > 10%?)
   b) Welche Market Structure hat die höchste Win-Rate?
   c) Korreliert ICT Score mit Win-Rate?
   d) Sind Order Blocks ein verlässlicher Zusatzfilter?

WICHTIG — Antworte NUR in diesem JSON-Format, ohne Markdown:
{{
  "analyse": {{
    "zusammenfassung": "2-3 verständliche Sätze für den Trader",
    "staerken": ["Konkrete Stärke 1", "Konkrete Stärke 2"],
    "schwaechen": ["Konkrete Schwäche 1", "Konkrete Schwäche 2"],
    "muster": [
      "Beispiel: FVG-Setups gewinnen zu 75% wenn VIX unter 20",
      "Beispiel: Trades am RTH_OPEN haben 20% höhere Win-Rate"
    ]
  }},
  "signal_bewertung": {{
    "SIGNAL_NAME": {{
      "win_rate": 65.0,
      "empfehlung": "STAERKEN",
      "begruendung": "Konkrete Begründung mit Bezug auf die Trade-Daten"
    }}
  }},
  "neue_gewichtungen": {{
    "MULTI_TF_BIAS": 1.15,
    "FAIR_VALUE_GAP": 0.90
  }},
  "kontext_anpassungen": {{
    "_vix_high_penalty": 0.75,
    "_time_open_bonus": 1.25
  }},
  "handlungsempfehlungen": [
    "Konkrete, verständliche Empfehlung 1",
    "Konkrete, verständliche Empfehlung 2",
    "Konkrete, verständliche Empfehlung 3"
  ],
  "naechste_analyse_in": "Nach weiteren X Trades oder in Y Tagen",
  "parameter_updates": {{
    "SIGNAL_GEWICHTUNGEN": {{
      "FAIR_VALUE_GAP": 1.15,
      "EMA_TREND": 0.85
    }},
    "KONFIDENZ_SCHWELLEN": {{
      "min_confidence_vix_high": 0.75
    }},
    "RISK_MANAGEMENT": {{
      "sl_atr_multiplier": 0.55
    }}
  }},
  "parameter_begruendung": {{
    "SIGNAL_GEWICHTUNGEN.FAIR_VALUE_GAP": "FVG hat 78% Win-Rate bei VIX < 20, sollte stärker gewichtet werden",
    "KONFIDENZ_SCHWELLEN.min_confidence_vix_high": "Bei VIX > 25 verlieren wir zu viele Trades, höhere Schwelle nötig"
  }},
  "ict_empfehlungen": {{
    "killzone_filter_staerken": true,
    "beste_market_structure": "CHOCH_BULLISH",
    "ict_score_schwelle_empfehlung": 0.3,
    "order_block_pflicht": false
  }}
}}"""
