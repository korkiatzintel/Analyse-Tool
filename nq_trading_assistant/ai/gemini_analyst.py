import logging
import time
from datetime import datetime


class GeminiAnalyst:
    """
    Kostenloser KI-Analyst via Google Gemini API (google-genai SDK).
    Ersetzt Claude für Live-Analyse.
    Limit: 1 500 Requests/Tag mit Gemini 2.0 Flash — mehr als genug.
    """

    _SYSTEM = (
        "Du bist ein erfahrener NQ Futures Day Trader und quantitativer Analyst. "
        "Analysiere Marktdaten präzise und handlungsorientiert. "
        "Antworte immer auf Deutsch. Sei knapp und direkt."
    )

    def __init__(self, api_key: str):
        from google import genai
        self._client       = genai.Client(api_key=api_key)
        self._log          = logging.getLogger(__name__)
        self._last_call    = 0.0
        self._min_interval = 30   # Sekunden zwischen Calls
        self.total_calls   = 0
        self.estimated_cost_usd = 0.0

    def get_cost_stats(self) -> dict:
        return {
            "total_calls":        self.total_calls,
            "cached_calls":       0,
            "cache_hit_rate":     0.0,
            "estimated_cost_usd": 0.0,
            "provider":           "Google Gemini (Free)",
        }

    def get_memory(self) -> list:
        return []

    async def analyze(self, signal_data: dict) -> dict:
        """Live-Analyse eines Trade-Signals."""
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            return {"skipped": True, "reason": "Rate limit"}

        rec  = signal_data.get("recommendation") or signal_data
        snap = signal_data.get("free_snapshot")   or signal_data

        flat = {
            "direction":   rec.get("direction", "NEUTRAL"),
            "confidence":  rec.get("confidence", 0.0),
            "signals":     rec.get("signals", []),
            "trade_setup": rec.get("trade_setup") or {},
            "price":       snap.get("last_price") or snap.get("price", 0),
            "vwap":        snap.get("session_vwap") or snap.get("vwap", 0),
            "vix":         snap.get("vix", 0),
            "vix_regime":  snap.get("vix_regime", "normal"),
            "bias":        snap.get("bias") or rec.get("bias", {}),
        }

        if flat["confidence"] < 0.65:
            return {"skipped": True, "reason": "Zu niedrige Konfidenz"}

        prompt = self._build_signal_prompt(flat)

        try:
            from google.genai import types
            response = self._client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self._SYSTEM,
                    max_output_tokens=500,
                ),
            )
            self._last_call  = time.time()
            self.total_calls += 1
            return self._parse_response(response.text.strip())
        except Exception as e:
            self._log.error("Gemini API Fehler: %s", e)
            return {"error": str(e), "skipped": True}

    def _build_signal_prompt(self, data: dict) -> str:
        bias = data.get("bias") or {}
        ts   = data.get("trade_setup") or {}
        sigs = data.get("signals", [])
        sig_lines = "\n".join(
            f"• {s.get('type', '?')}: {s.get('description', '')}"
            for s in sigs
        )
        return (
            f"NQ Futures Signal-Analyse — {datetime.utcnow().strftime('%H:%M UTC')}\n\n"
            f"MARKT-BIAS: {bias.get('direction', '?')} ({bias.get('probability', 0):.0f}%)\n"
            f"SIGNAL: {data.get('direction', '?')} | Konfidenz: {data.get('confidence', 0):.0%}\n"
            f"PREIS: {data.get('price', 0):.2f} | VWAP: {data.get('vwap', 0):.2f}\n"
            f"VIX: {data.get('vix', 0):.1f} ({data.get('vix_regime', '?')})\n\n"
            f"AKTIVE SIGNALE:\n{sig_lines}\n\n"
            f"TRADE SETUP:\n"
            f"Entry: {ts.get('entry_price', 0):.2f}\n"
            f"SL: {ts.get('stop_loss_price', 0):.2f}\n"
            f"TP1: {ts.get('take_profit_1_price', 0):.2f}\n"
            f"TP2: {ts.get('take_profit_2_price', 0):.2f}\n\n"
            "Antworte in exakt diesem Format:\n"
            "URTEIL: [BESTÄTIGT / ABGELEHNT / WARTE]\n"
            "BEGRÜNDUNG: [max 2 Sätze]\n"
            "BEACHTUNG: [ein kritischer Punkt]"
        )

    def _parse_response(self, text: str) -> dict:
        result = {
            "verdict":      "WARTE",
            "begruendung":  "",
            "beachtung":    "",
            "raw_response": text,
            "skipped":      False,
            "timestamp":    datetime.utcnow().isoformat(),
            "provider":     "Gemini",
        }
        for line in text.splitlines():
            if line.startswith("URTEIL:"):
                raw = line.replace("URTEIL:", "").strip()
                if "BESTÄTIGT" in raw:
                    result["verdict"] = "BESTÄTIGT"
                elif "ABGELEHNT" in raw:
                    result["verdict"] = "ABGELEHNT"
            elif line.startswith("BEGRÜNDUNG:"):
                result["begruendung"] = line.replace("BEGRÜNDUNG:", "").strip()
            elif line.startswith("BEACHTUNG:"):
                result["beachtung"] = line.replace("BEACHTUNG:", "").strip()
        return result
