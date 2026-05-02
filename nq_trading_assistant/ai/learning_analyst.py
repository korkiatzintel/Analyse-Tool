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
        closed_trades   = simulator.get_closed_trades(limit=100)
        current_weights = simulator.get_weights()
        self._strategy_config = strategy_config

        if stats["total"] < 5:
            return {
                "status":       "insufficient_data",
                "message":      f"Zu wenige Trades ({stats['total']}). Mindestens 5 benötigt.",
                "min_required": 5,
            }

        prompt = self._build_analysis_prompt(stats, closed_trades, current_weights)

        try:
            if hasattr(self._analyst, "run_daily_analysis"):
                # GeminiAnalyst — nutzt automatisches Modell-Fallback
                raw = self._analyst.run_daily_analysis(prompt)
            elif hasattr(self._analyst, "_anthropic"):
                response = self._analyst._anthropic.messages.create(
                    model      = "claude-sonnet-4-6",
                    max_tokens = 2000,
                    system     = self._SYSTEM_PROMPT,
                    messages   = [{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text.strip()
            else:
                raise Exception("Kein kompatibler KI-Client verfügbar")

            raw = raw.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()

            # Truncate to last complete JSON object if response was cut off
            if not raw.endswith("}"):
                last_brace = raw.rfind("}")
                if last_brace > 0:
                    raw = raw[:last_brace + 1]

            try:
                result = json.loads(raw)
            except json.JSONDecodeError as e:
                self._log.error("JSON Parse Error: %s", e)
                self._log.error("Raw response (first 500): %s", raw[:500])
                result = {
                    "analyse": {
                        "zusammenfassung": "JSON-Parse Fehler — bitte erneut versuchen",
                        "staerken":   [],
                        "schwaechen": ["Antwort war unvollständig"],
                        "muster":     [],
                    },
                    "signal_bewertung":    {},
                    "neue_gewichtungen":   {},
                    "kontext_anpassungen": {},
                    "handlungsempfehlungen": ["Lernanalyse erneut starten"],
                    "naechste_analyse_in":   "Sofort erneut versuchen",
                }

            # Apply strategy parameter updates
            param_updates = result.get("parameter_updates", {})
            if param_updates and self._strategy_config:
                update_report = self._strategy_config.update_from_gemini(param_updates)
                result["update_report"] = update_report
                accepted = len(update_report.get("accepted", []))
                rejected = len(update_report.get("rejected", []))
                self._log.info(
                    "Parameter angepasst: %d übernommen, %d begrenzt/abgelehnt",
                    accepted, rejected,
                )

            analysis_record = {
                "timestamp":        datetime.utcnow().isoformat(),
                "stats_snapshot":   stats,
                "result":           result,
                "trades_analyzed":  stats["total"],
            }

            history: list = []
            try:
                history = json.loads(
                    self._analysis_file.read_text(encoding="utf-8")
                )
            except Exception:
                pass
            history.append(analysis_record)
            self._analysis_file.write_text(
                json.dumps(history[-10:], indent=2),
                encoding="utf-8",
            )

            new_weights     = result.get("neue_gewichtungen", {})
            context_weights = result.get("kontext_anpassungen", {})
            new_weights.update(context_weights)
            new_weights["_total_trades_analyzed"] = stats["total"]
            simulator.apply_weights(new_weights)

            self._log.info(
                "Learning Analysis abgeschlossen: %d Trades, Win-Rate %.1f%%, "
                "%d Gewichtungen aktualisiert",
                stats["total"], stats["win_rate"], len(new_weights),
            )

            return {"status": "success", "result": result, "stats": stats}

        except Exception as e:
            self._log.error("Learning Analysis Fehler: %s", e)
            return {"status": "error", "message": str(e)}

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
  }}
}}"""
