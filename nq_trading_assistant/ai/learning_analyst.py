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

    def __init__(self, client, model: str = "claude-sonnet-4-6"):
        self._client        = client
        self._model         = model
        self._log           = logging.getLogger(__name__)
        self._analysis_file = Path(__file__).parent.parent / "logs" / "learning_analysis.json"

    def run_daily_analysis(self, simulator) -> dict:
        stats         = simulator.get_stats()
        closed_trades = simulator.get_closed_trades(limit=100)
        current_weights = simulator.get_weights()

        if stats["total"] < 5:
            return {
                "status":       "insufficient_data",
                "message":      f"Zu wenige Trades ({stats['total']}). Mindestens 5 benötigt.",
                "min_required": 5,
            }

        prompt = self._build_analysis_prompt(stats, closed_trades, current_weights)

        try:
            response = self._client.messages.create(
                model      = self._model,
                max_tokens = 2000,
                system     = self._SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": prompt}],
            )

            raw    = response.content[0].text.strip()
            result = json.loads(raw)

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

        except json.JSONDecodeError as e:
            self._log.error("Claude Response Parse Error: %s", e)
            return {"status": "error", "message": f"JSON Parse Fehler: {e}"}
        except Exception as e:
            self._log.error("Learning Analysis Fehler: %s", e)
            return {"status": "error", "message": str(e)}

    def _build_analysis_prompt(self, stats: dict, trades: list, weights: dict) -> str:
        recent = trades[:20]
        trade_summary = [
            {
                "id":           t["trade_id"],
                "direction":    t["direction"],
                "confidence":   f"{t['confidence']:.0%}",
                "signals":      t["active_signals"],
                "outcome":      t["outcome"],
                "pnl_pts":      t.get("pnl_points", 0),
                "duration_min": t.get("duration_minutes", 0),
                "vix_regime":   t.get("vix_regime"),
                "time_of_day":  t.get("time_of_day"),
                "exit":         t.get("exit_reason"),
            }
            for t in recent
        ]

        return f"""Analysiere folgende NQ Futures Simulations-Daten und optimiere die Signal-Gewichtungen:

GESAMTSTATISTIK:
- Trades gesamt: {stats['total']}
- Win-Rate: {stats['win_rate']}%
- Avg P&L: {stats['avg_pnl_points']} Punkte / ${stats['avg_pnl_usd']} pro Trade
- Gesamt P&L: ${stats['total_pnl_usd']}
- Wins: {stats['wins']} | Losses: {stats['losses']} | Timeouts: {stats['timeouts']}

WIN-RATE NACH SIGNAL-TYP:
{json.dumps(stats.get('best_signal_types', {}), indent=2)}

WIN-RATE NACH VIX-REGIME:
{json.dumps(stats.get('win_rate_by_regime', {}), indent=2)}

WIN-RATE NACH TAGESZEIT:
{json.dumps(stats.get('win_rate_by_time', {}), indent=2)}

AKTUELLE GEWICHTUNGEN:
{json.dumps({k: v for k, v in weights.items() if not k.startswith('_')}, indent=2)}

LETZTE 20 TRADES (neueste zuerst):
{json.dumps(trade_summary, indent=2)}

AUFGABE:
1. Identifiziere welche Signal-Kombinationen zu Wins/Losses führen
2. Erkenne Muster (z.B. "FAIR_VALUE_GAP funktioniert nur bei normalem VIX")
3. Empfehle neue Gewichtungen die zukünftige Win-Rate verbessern
4. Berücksichtige: NQ hat typische Win-Rate von 60-65% bei guten Setups
5. Sei konservativ — max ±0.3 Änderung pro Signal"""
