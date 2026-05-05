import json
import logging
import re
from pathlib import Path
from datetime import datetime

CHANGELOG_FILE = Path(__file__).parent.parent / "logs" / "system_changelog.json"


def _load_system_changelog() -> dict:
    try:
        return json.loads(CHANGELOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _clean_diagnose(text: str) -> str:
    """Bereinige Gemini-Diagnosetext — entferne JSON-Artefakte wenn nötig."""
    clean = text.strip().replace("```json", "").replace("```", "").strip()

    if clean.startswith(("{", "[")):
        try:
            parsed = json.loads(clean)

            def _extract(obj, depth=0) -> str:
                if depth > 3:
                    return ""
                if isinstance(obj, str):
                    return obj
                if isinstance(obj, list):
                    return " | ".join(_extract(i, depth + 1) for i in obj if i)
                if isinstance(obj, dict):
                    return " | ".join(
                        _extract(v, depth + 1) for v in obj.values() if v
                    )
                return str(obj)

            clean = _extract(parsed)
        except Exception:
            clean = re.sub(r'[{}\[\]":]', " ", clean)
            clean = re.sub(r"\s+", " ", clean).strip()

    return clean


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

    def _evaluate_previous_recommendations(self, current_stats: dict) -> str:
        """Vergleiche aktuelle Win-Rate mit der letzten Analyse und gib Feedback."""
        try:
            history = json.loads(self._analysis_file.read_text(encoding="utf-8"))
        except Exception:
            return ""
        if not history:
            return ""

        last = history[-1]
        prev_wr = last.get("stats_snapshot", {}).get("win_rate", None)
        curr_wr = current_stats.get("win_rate", None)
        if prev_wr is None or curr_wr is None:
            return ""

        prev_trades = last.get("trades_analyzed", 0)
        curr_trades = current_stats.get("total", 0)
        delta       = curr_wr - prev_wr
        new_trades  = curr_trades - prev_trades

        prev_recs = (
            last.get("result", {}).get("handlungsempfehlungen", [])
        )
        recs_text = "\n".join(f"  - {r}" for r in prev_recs[:3]) if prev_recs else "  (keine)"

        direction = "verbessert" if delta > 0 else "verschlechtert" if delta < 0 else "unveraendert"
        return (
            f"VORHERIGE ANALYSE (vor {new_trades} neuen Trades):\n"
            f"  Win-Rate damals: {prev_wr}% → jetzt: {curr_wr}% ({delta:+.1f}%, {direction})\n"
            f"  Damalige Empfehlungen:\n{recs_text}\n"
            f"  → Beurteile ob die Empfehlungen geholfen haben und passe deine neue Analyse entsprechend an.\n"
        )

    def run_daily_analysis(self, simulator, strategy_config=None) -> dict:
        stats           = simulator.get_stats()
        closed_trades   = simulator.get_closed_trades(limit=20)
        current_weights = simulator.get_weights()
        self._strategy_config = strategy_config

        if stats["total"] < 5:
            return {
                "status":  "insufficient_data",
                "message": f"Zu wenige Trades ({stats['total']}). Mindestens 5 benötigt.",
            }

        # ── System-Kontext laden ──────────────────────────────────────────
        changelog = _load_system_changelog()
        active_fixes = changelog.get("active_fixes", [])
        baseline_wr  = changelog.get("baseline_win_rate", None)
        baseline_n   = changelog.get("baseline_trades", None)

        system_context_lines = ["AKTIVE SYSTEM-FIXES (Version 2.0):"]
        for fix in active_fixes:
            system_context_lines.append(
                f"  [{fix['id']}] {fix['beschreibung']}"
                f" — Erwartet: {fix.get('erwartet', '?')}"
            )
        if baseline_wr is not None:
            system_context_lines.append(
                f"\nBASELINE (vor Fixes): {baseline_wr}% Win-Rate über {baseline_n} Trades"
            )
        system_context = "\n".join(system_context_lines)

        # ── Fix-Validierung aufbereiten ───────────────────────────────────
        fix_validation = stats.get("fix_validation", {})
        if fix_validation:
            fv_lines = ["FIX-VALIDIERUNG (automatisch geprüft):"]
            for fix_id, val in fix_validation.items():
                status = val.get("status", "?")
                detail = val.get("detail", "")
                icon   = "✓" if status == "OK" else ("?" if status == "NICHT_PRUEFBAR" else "✗")
                fv_lines.append(f"  {icon} {fix_id}: {status} — {detail}")
            fix_val_text = "\n".join(fv_lines)
        else:
            fix_val_text = ""

        # ── Vorherige Empfehlungen auswerten ──────────────────────────────
        prev_rec_text = self._evaluate_previous_recommendations(stats)

        # ── Daten kompakt aufbereiten ─────────────────────────────────────
        trade_lines = []
        for t in closed_trades[:15]:
            kz  = t.get("killzone") or "-"
            ms  = (t.get("market_structure") or "-")[:15]
            ict = f"{t.get('ict_score', 0) or 0:.0%}"
            ob  = t.get("order_blocks_active", 0) or 0
            sigs = ",".join((t.get("active_signals") or [])[:3])
            trade_lines.append(
                f"{t['direction']} {t.get('confidence', 0):.0%}"
                f"→{t.get('outcome', '?')} | "
                f"KZ:{kz} MS:{ms} ICT:{ict} OB:{ob} | {sigs}"
            )
        trades_compact = "\n".join(trade_lines)

        stats_compact = (
            f"Trades:{stats['total']} "
            f"WR:{stats['win_rate']}% "
            f"W:{stats['wins']} L:{stats['losses']} T:{stats['timeouts']}\n"
            f"AvgPnL:{stats['avg_pnl_points']:+.1f}Pts "
            f"TotalPnL:${stats['total_pnl_usd']:+.0f}"
        )

        sig_perf = " | ".join(
            f"{k}:{v}%" for k, v in stats.get("best_signal_types", {}).items()
        )

        kz_stats   = stats.get("win_rate_by_killzone", {})
        kz_compact = (
            f"InKZ:{kz_stats.get('IN_KILLZONE', 0)}% "
            f"OutKZ:{kz_stats.get('OUTSIDE_KILLZONE', 0)}%"
        )
        ms_stats = " | ".join(
            f"{k}:{v}%" for k, v in stats.get("win_rate_by_market_structure", {}).items()
        )
        ict_score_stats = " | ".join(
            f"{k}:{v}%" for k, v in stats.get("win_rate_by_ict_score", {}).items()
        )
        ob_stats   = stats.get("win_rate_by_order_blocks", {})
        ob_compact = (
            f"MitOB:{ob_stats.get('MIT_OB', 0)}% "
            f"OhneOB:{ob_stats.get('OHNE_OB', 0)}%"
        )
        weights_compact = " | ".join(
            f"{k}:{v:.2f}"
            for k, v in current_weights.items()
            if not k.startswith("_") and isinstance(v, (int, float))
        )

        # ── CALL 1: Diagnose mit System-Kontext (~350 Token Output) ───────
        prompt_1 = (
            f"NQ Futures Trading System v2.0 — Diagnose auf Deutsch.\n\n"
            f"{system_context}\n\n"
            + (f"{fix_val_text}\n\n" if fix_val_text else "")
            + (f"{prev_rec_text}\n" if prev_rec_text else "")
            + f"PERFORMANCE:\n{stats_compact}\n"
            f"Signale: {sig_perf}\n"
            f"Zeit: {' | '.join(f'{k}:{v}%' for k, v in stats.get('win_rate_by_time', {}).items())}\n"
            f"VIX: {' | '.join(f'{k}:{v}%' for k, v in stats.get('win_rate_by_regime', {}).items())}\n\n"
            f"ICT ANALYSE:\n"
            f"Killzones: {kz_compact}\n"
            f"Market Structure: {ms_stats}\n"
            f"ICT Score: {ict_score_stats}\n"
            f"Order Blocks: {ob_compact}\n\n"
            f"LETZTE 15 TRADES:\n{trades_compact}\n\n"
            f"Beantworte genau diese 4 Fragen (je max 15 Wörter):\n"
            f"1. Grösste Stärke des Systems NACH den v2.0-Fixes?\n"
            f"2. Welcher Fix hat noch NICHT die erwartete Wirkung gezeigt und warum?\n"
            f"3. Gibt es Zeitfenster oder VIX-Regime mit auffällig schlechter Performance trotz Fixes?\n"
            f"4. Wichtigste nächste Massnahme die ÜBER die aktiven Fixes hinausgeht?"
        )
        diagnose = self._call_analyst(prompt_1, max_tokens=350)
        self._log.info("Diagnose erhalten: %d Zeichen", len(diagnose))

        # Bereinige Diagnose — Gemini gibt manchmal JSON statt Text zurück
        diagnose = _clean_diagnose(diagnose)

        # ── CALL 2: Signal-Gewichtungen (~200 Token Output) ───────────────
        prompt_2 = (
            f"NQ System Gewichtungen optimieren.\n\n"
            f"Diagnose: {diagnose[:200]}\n"
            f"Win-Rate: {stats['win_rate']}%\n"
            f"Signale (Win-Rate%): {sig_perf}\n"
            f"ICT Score Korrelation: {ict_score_stats}\n"
            f"OB Einfluss: {ob_compact}\n\n"
            f"Aktuelle Gewichtungen:\n{weights_compact}\n\n"
            f"Antworte NUR mit JSON (kein Text davor/danach):\n"
            f"Regeln: Werte 0.5-1.5, max ±0.2 Änderung, max 4 Änderungen.\n"
            f'Format exakt:\n'
            f'{{"MULTI_TF_BIAS":1.0,"FAIR_VALUE_GAP":1.0,"VIX_REGIME":1.0,'
            f'"OVERNIGHT_GAP":1.0,"EMA_TREND":1.0,"VWAP_POSITION":1.0,'
            f'"RSI_EXTREME":1.0,"MEAN_REVERSION":1.0,"SESSION_LEVELS":1.0,'
            f'"ICT_CONFLUENCE":1.0}}'
        )
        weights_raw = self._call_analyst(prompt_2, max_tokens=250)
        self._log.info("Gewichtungen erhalten: %d Zeichen", len(weights_raw))

        # ── CALL 3: ICT Parameter + Empfehlungen (~300 Token) ─────────────
        prompt_3 = (
            f"NQ ICT Parameter + Handlungsempfehlungen.\n\n"
            f"Diagnose: {diagnose[:150]}\n"
            f"Killzone: {kz_compact}\n"
            f"OB: {ob_compact}\n"
            f"MS: {ms_stats}\n\n"
            f"Antworte NUR mit JSON:\n"
            f'{{"ict_empfehlungen":{{"killzone_filter_staerken":true,'
            f'"beste_market_structure":"CHOCH_BULLISH",'
            f'"ict_score_schwelle":0.3,"order_block_pflicht":false}},'
            f'"kontext_anpassungen":{{"_vix_high_penalty":0.8,'
            f'"_time_open_bonus":1.2,"_time_close_penalty":0.9,'
            f'"outside_killzone_confidence_penalty":0.85}},'
            f'"handlungsempfehlungen":["Empfehlung 1 max 12 Wörter",'
            f'"Empfehlung 2 max 12 Wörter","Empfehlung 3 max 12 Wörter"],'
            f'"naechste_analyse_in":"Nach X Trades"}}'
        )
        params_raw = self._call_analyst(prompt_3, max_tokens=350)
        self._log.info("Parameter erhalten: %d Zeichen", len(params_raw))

        # ── Parse Gewichtungen ────────────────────────────────────────────
        neue_gewichtungen: dict = {}
        try:
            w = weights_raw.strip().replace("```json", "").replace("```", "").strip()
            start = w.find("{")
            end   = w.rfind("}") + 1
            if start >= 0 and end > start:
                w = w[start:end]
            parsed_w = json.loads(w)
            neue_gewichtungen = {
                k: max(0.5, min(1.5, float(v)))
                for k, v in parsed_w.items()
                if isinstance(v, (int, float))
            }
        except Exception as e:
            self._log.error("Gewichtungen Parse Error: %s | raw: %s", e, weights_raw[:200])

        # ── Parse ICT Parameter + Empfehlungen ───────────────────────────
        ict_empfehlungen:     dict = {}
        kontext_anpassungen:  dict = {}
        handlungsempfehlungen: list = []
        naechste_analyse = "Nach 20 weiteren Trades"

        try:
            p = params_raw.strip().replace("```json", "").replace("```", "").strip()
            start = p.find("{")
            end   = p.rfind("}") + 1
            if start >= 0 and end > start:
                p = p[start:end]
            parsed_p = json.loads(p)
            ict_empfehlungen      = parsed_p.get("ict_empfehlungen", {})
            kontext_anpassungen   = parsed_p.get("kontext_anpassungen", {})
            handlungsempfehlungen = parsed_p.get("handlungsempfehlungen", [])
            naechste_analyse      = parsed_p.get("naechste_analyse_in", naechste_analyse)
        except Exception as e:
            self._log.error("Parameter Parse Error: %s | raw: %s", e, params_raw[:200])

        # ── Signal-Bewertung aus Stats ────────────────────────────────────
        signal_bewertung: dict = {}
        for sig, wr in stats.get("best_signal_types", {}).items():
            signal_bewertung[sig] = {
                "win_rate":    wr,
                "empfehlung":  ("STAERKEN" if wr >= 60
                                else "REDUZIEREN" if wr <= 40
                                else "BEIBEHALTEN"),
                "begruendung": f"Win-Rate {wr}% aus {stats['total']} Trades",
            }

        # ── Gewichtungen + Kontext anwenden ──────────────────────────────
        all_weights = {**neue_gewichtungen, **kontext_anpassungen}
        if all_weights:
            simulator.apply_weights(all_weights)
            self._log.info(
                "Angepasst: %d Signale, %d Kontext-Parameter",
                len(neue_gewichtungen), len(kontext_anpassungen),
            )

        # ── ICT Empfehlungen in Strategy Config schreiben ────────────────
        if strategy_config and ict_empfehlungen:
            try:
                ict_params: dict = {}
                if "ict_score_schwelle" in ict_empfehlungen:
                    ict_params["ict_score_min_threshold"] = \
                        ict_empfehlungen["ict_score_schwelle"]
                if "outside_killzone_confidence_penalty" in kontext_anpassungen:
                    ict_params["outside_killzone_confidence_penalty"] = \
                        kontext_anpassungen["outside_killzone_confidence_penalty"]
                if ict_params:
                    strategy_config.update_from_gemini({"ICT_PARAMETER": ict_params})
            except Exception as e:
                self._log.error("ICT Config Update Fehler: %s", e)

        # ── Finales Result ────────────────────────────────────────────────
        result = {
            "analyse": {
                "zusammenfassung": diagnose[:600],
                "staerken":        [],
                "schwaechen":      [],
                "muster":          [],
            },
            "signal_bewertung":      signal_bewertung,
            "neue_gewichtungen":     neue_gewichtungen,
            "kontext_anpassungen":   kontext_anpassungen,
            "ict_empfehlungen":      ict_empfehlungen,
            "handlungsempfehlungen": handlungsempfehlungen[:3],
            "naechste_analyse_in":   naechste_analyse,
            "calls_used":            3,
            "system_version":        changelog.get("version", "?"),
            "fix_validation":        fix_validation,
            "baseline_win_rate":     baseline_wr,
        }

        # ── Speichern ─────────────────────────────────────────────────────
        weights_simple = {
            k: v for k, v in current_weights.items()
            if not k.startswith("_") and isinstance(v, (int, float))
        }
        analysis_record = {
            "timestamp":        datetime.utcnow().isoformat(),
            "stats_snapshot":   stats,
            "result":           result,
            "trades_analyzed":  stats["total"],
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

        last_result_path = Path(__file__).parent.parent / "logs" / "last_learning_result.json"
        last_result_path.write_text(
            json.dumps(analysis_record, indent=2, default=str),
            encoding="utf-8",
        )

        self._log.info(
            "Learning Analysis fertig: %d Trades, Win-Rate %s%%, "
            "%d Gewichtungen, %d ICT-Parameter",
            stats["total"], stats["win_rate"],
            len(neue_gewichtungen), len(ict_empfehlungen),
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
