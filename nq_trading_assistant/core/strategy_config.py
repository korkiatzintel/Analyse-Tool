import json
import logging
from datetime import datetime
from pathlib import Path

_BASE = Path(__file__).parent.parent
PARAMS_FILE  = _BASE / "config" / "strategy_params.json"
HISTORY_FILE = _BASE / "logs"   / "params_history.json"


class StrategyConfig:
    """
    Central configuration for all strategy parameters.
    Used by all modules instead of hard-coded values.
    Validates all changes against defined ranges.
    """

    def __init__(self):
        self._log    = logging.getLogger(__name__)
        self._params = self._load()

    def _load(self) -> dict:
        try:
            return json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            self._log.error("strategy_params.json nicht gefunden: %s", e)
            return {}

    def get(self, section: str, key: str, default=None):
        sec = self._params.get(section, {})
        if not isinstance(sec, dict):
            return default
        return sec.get(key, default)

    def get_section(self, section: str) -> dict:
        data = self._params.get(section, {})
        return {k: v for k, v in data.items() if not k.startswith("_")}

    def update_from_gemini(self, gemini_updates: dict) -> dict:
        """
        Apply parameter updates from Gemini.
        Validates against ranges and max_change.
        Returns a report of accepted/rejected changes.
        """
        report = {
            "accepted":  [],
            "rejected":  [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._save_history()

        for section_key, updates in gemini_updates.items():
            if section_key == "KERNREGELN":
                report["rejected"].append({
                    "section": section_key,
                    "reason":  "Kernregeln sind unveränderlich",
                })
                continue

            if section_key not in self._params:
                continue

            section   = self._params[section_key]
            val_range = section.get("_range", [0, 999])
            max_chg   = section.get("_max_change_per_update", 0.5)

            if not isinstance(updates, dict):
                continue

            for param, new_value in updates.items():
                if param.startswith("_"):
                    continue

                if param not in section:
                    report["rejected"].append({
                        "param":  f"{section_key}.{param}",
                        "reason": "Parameter nicht bekannt",
                    })
                    continue

                old_value = section[param]

                if not isinstance(new_value, (int, float)):
                    report["rejected"].append({
                        "param":  f"{section_key}.{param}",
                        "reason": "Kein numerischer Wert",
                    })
                    continue

                # Clamp to range
                clamped = False
                if isinstance(val_range, list) and len(val_range) == 2:
                    lo, hi = val_range[0], val_range[1]
                    if not (lo <= new_value <= hi):
                        new_value = max(lo, min(hi, new_value))
                        clamped   = True

                # Clamp to max change
                if isinstance(old_value, (int, float)):
                    actual_change = abs(new_value - old_value)
                    if actual_change > max_chg:
                        direction = 1 if new_value > old_value else -1
                        new_value = round(old_value + direction * max_chg, 4)
                        report["rejected"].append({
                            "param":      f"{section_key}.{param}",
                            "reason":     f"Änderung zu groß ({actual_change:.3f} > {max_chg}), "
                                          f"begrenzt auf {new_value:.3f}",
                            "clamped_to": new_value,
                        })
                        section[param] = round(new_value, 4)
                        report["accepted"].append({
                            "param":  f"{section_key}.{param}",
                            "old":    old_value,
                            "new":    round(new_value, 4),
                            "change": round(new_value - old_value, 4),
                        })
                        continue

                if clamped:
                    report["rejected"].append({
                        "param":      f"{section_key}.{param}",
                        "reason":     f"Außerhalb Range {val_range}, auf {new_value:.3f} begrenzt",
                        "clamped_to": new_value,
                    })

                section[param] = round(new_value, 4)
                report["accepted"].append({
                    "param":  f"{section_key}.{param}",
                    "old":    old_value,
                    "new":    round(new_value, 4),
                    "change": round(new_value - old_value, 4),
                })

        self._params["_version"]      = self._params.get("_version", 1) + 1
        self._params["_last_updated"] = datetime.utcnow().isoformat()
        self._params["_update_count"] = self._params.get("_update_count", 0) + 1

        self._save()
        self._log.info(
            "Parameter Update: %d angenommen, %d abgelehnt/begrenzt",
            len(report["accepted"]), len(report["rejected"]),
        )
        return report

    def _save(self):
        PARAMS_FILE.write_text(
            json.dumps(self._params, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _save_history(self):
        history: list = []
        try:
            history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "version":   self._params.get("_version", 1),
            "params": {
                k: {pk: pv for pk, pv in v.items() if not pk.startswith("_")}
                for k, v in self._params.items()
                if isinstance(v, dict) and not k.startswith("_")
            },
        }
        history.append(snapshot)
        HISTORY_FILE.parent.mkdir(exist_ok=True)
        HISTORY_FILE.write_text(
            json.dumps(history[-30:], indent=2),
            encoding="utf-8",
        )

    def get_all_for_gemini(self) -> dict:
        """Return all tunable parameters for the Gemini prompt."""
        result = {}
        for section, content in self._params.items():
            if section.startswith("_") or section == "KERNREGELN":
                continue
            if isinstance(content, dict):
                result[section] = {
                    "beschreibung":  content.get("_beschreibung", ""),
                    "range":         content.get("_range", "beliebig"),
                    "max_aenderung": content.get("_max_change_per_update", 0.5),
                    "aktuelle_werte": {
                        k: v for k, v in content.items()
                        if not k.startswith("_")
                    },
                }
        return result
