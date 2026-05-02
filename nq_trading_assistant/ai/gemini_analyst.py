import logging
import time
from datetime import datetime


class GeminiAnalyst:
    """
    Kostenloser KI-Analyst via Google Gemini API (google-genai SDK).
    Nutzt gemini-2.5-flash primär, fällt auf gemini-2.0-flash-lite zurück
    wenn Quota (429) erreicht wird.
    """

    MODELS = [
        "gemini-2.5-flash",      # Primär — klüger, 20 Req/Tag Free
        "gemini-2.0-flash-lite", # Fallback — 1500 Req/Tag Free
    ]

    _SYSTEM_LIVE = (
        "Du bist ein erfahrener NQ Futures Day Trader. "
        "Analysiere Trade-Signale präzise und handlungsorientiert. "
        "Antworte auf Deutsch. Sei knapp und direkt.\n"
        "Antworte NUR in diesem Format:\n"
        "URTEIL: [BESTÄTIGT / ABGELEHNT / WARTE]\n"
        "BEGRÜNDUNG: [max 2 Sätze]\n"
        "BEACHTUNG: [ein kritischer Punkt]"
    )

    _SYSTEM_LEARNING = (
        "Du bist ein quantitativer Trading-Analyst "
        "spezialisiert auf NQ Futures (E-Mini Nasdaq-100). "
        "Antworte AUSSCHLIESSLICH in validem JSON ohne Markdown-Backticks."
    )

    def __init__(self, api_key: str):
        from google import genai
        self._client          = genai.Client(api_key=api_key)
        self._log             = logging.getLogger(__name__)
        self._last_call       = 0.0
        self._min_interval    = 60   # Sekunden zwischen Live-Calls
        self.total_calls      = 0
        self.estimated_cost_usd = 0.0

        self._current_model_idx = 0
        self._model_failures    = {m: 0 for m in self.MODELS}
        self._model_last_429    = {m: 0.0 for m in self.MODELS}

    # ── Model selection ────────────────────────────────────────────────────

    def _get_model(self) -> str:
        now = time.time()
        for i, model in enumerate(self.MODELS):
            if now - self._model_last_429.get(model, 0.0) > 3600:
                self._current_model_idx = i
                return model
        return self.MODELS[-1]

    # ── Core API call with automatic fallback ──────────────────────────────

    def _call_gemini(
        self,
        prompt:      str,
        system:      str,
        max_tokens:  int = 500,
        force_model: str = None,
    ) -> str:
        from google.genai import types

        models_to_try = [force_model] if force_model else self.MODELS

        for model in models_to_try:
            try:
                self._log.info("Gemini: Versuche %s...", model)
                response = self._client.models.generate_content(
                    model   = model,
                    contents = prompt,
                    config  = types.GenerateContentConfig(
                        system_instruction = system,
                        max_output_tokens  = max_tokens,
                    ),
                )
                self._log.info("Gemini: %s erfolgreich", model)
                self.total_calls += 1
                return response.text.strip()

            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    self._model_last_429[model] = time.time()
                    self._model_failures[model] += 1
                    self._log.warning(
                        "Gemini: %s Quota erreicht — versuche nächstes Modell...", model
                    )
                    continue
                self._log.error("Gemini API Fehler (%s): %s", model, e)
                raise

        raise Exception(
            "Alle Gemini Modelle haben Quota-Limit erreicht. "
            "Versuche es in einer Stunde wieder oder upgrade auf Paid Tier."
        )

    # ── Live signal analysis ───────────────────────────────────────────────

    async def analyze(self, signal_data: dict) -> dict:
        now = time.time()
        if now - self._last_call < self._min_interval:
            return {"skipped": True, "reason": f"Rate limit ({self._min_interval}s)"}

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
            text = self._call_gemini(prompt, self._SYSTEM_LIVE, max_tokens=300)
            self._last_call = time.time()
            return self._parse_response(text)
        except Exception as e:
            self._log.error("Gemini analyze Fehler: %s", e)
            return {"error": str(e), "skipped": True}

    # ── Daily learning analysis ────────────────────────────────────────────

    def run_daily_analysis(self, prompt: str) -> str:
        """
        Lernanalyse — nutzt primär gemini-2.5-flash (klüger).
        Fällt auf gemini-2.0-flash-lite zurück wenn Quota erreicht.
        """
        return self._call_gemini(
            prompt, self._SYSTEM_LEARNING, max_tokens=2000
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def get_cost_stats(self) -> dict:
        return {
            "total_calls":        self.total_calls,
            "cached_calls":       0,
            "cache_hit_rate":     0.0,
            "estimated_cost_usd": 0.0,
            "provider":           "Google Gemini (Free)",
            "current_model":      self._get_model(),
            "model_failures":     self._model_failures,
            "models_available": [
                m for m in self.MODELS
                if time.time() - self._model_last_429.get(m, 0.0) > 3600
            ],
        }

    def get_memory(self) -> list:
        return []

    def _build_signal_prompt(self, data: dict) -> str:
        bias      = data.get("bias") or {}
        ts        = data.get("trade_setup") or {}
        sig_lines = "\n".join(
            f"• {s.get('type', '?')}: {s.get('description', '')}"
            for s in data.get("signals", [])
        )
        direction  = data.get("direction", "?")
        confidence = data.get("confidence", 0.0)
        price      = data.get("price", 0)
        vwap       = data.get("vwap", 0)
        vix        = data.get("vix", 0)
        vix_regime = data.get("vix_regime", "?")
        bias_dir   = bias.get("direction", "?")
        bias_prob  = bias.get("probability", 0)
        entry      = ts.get("entry_price", 0)
        sl         = ts.get("stop_loss_price", 0)
        tp1        = ts.get("take_profit_1_price", 0)
        tp2        = ts.get("take_profit_2_price", 0)
        utc_time   = datetime.utcnow().strftime("%H:%M UTC")

        return (
            f"NQ Futures Signal-Analyse — {utc_time}\n\n"
            f"MARKT-BIAS: {bias_dir} ({bias_prob:.0f}%)\n"
            f"SIGNAL: {direction} | Konfidenz: {confidence:.0%}\n"
            f"PREIS: {price:.2f} | VWAP: {vwap:.2f}\n"
            f"VIX: {vix:.1f} ({vix_regime})\n\n"
            f"AKTIVE SIGNALE:\n{sig_lines}\n\n"
            f"TRADE SETUP:\n"
            f"Entry: {entry:.2f}\n"
            f"SL: {sl:.2f}\n"
            f"TP1: {tp1:.2f}\n"
            f"TP2: {tp2:.2f}"
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
            "model":        self._get_model(),
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
