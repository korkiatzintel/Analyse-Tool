"""
AI-powered trade signal analysis via the Anthropic API.

ClaudeAnalyst wraps the Signal Engine output and L2 market data into
a structured prompt, calls claude-sonnet-4-6, and returns a
parsed verdict dict.

Rate limiting: one API call per 30 seconds maximum.
Call guard: only fires when confidence > 0.65 AND >= 2 distinct signal types.

Memory: rolling deque of last 20 analyses (in-process, no persistence).
"""

import asyncio
import configparser
import logging
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

_CONFIG_PATH   = Path(__file__).parent.parent / "config" / "credentials.ini"
_MODEL         = "claude-sonnet-4-6"
_MIN_INTERVAL  = 30.0      # seconds between API calls
_MIN_CONFIDENCE = 0.65
_MIN_SIGNAL_TYPES = 2
_MEMORY_SIZE   = 20
_MAX_TOKENS    = 512

_SYSTEM_PROMPT = """\
Du bist ein erfahrener NQ Futures Day Trader und Order Flow Spezialist.
Du analysierst Echtzeit-Marktdaten des E-Mini Nasdaq 100 (NQ).
Deine Aufgabe: Gib präzise, handlungsorientierte Einschätzungen basierend auf Order Flow und technischer Analyse.
Antworte auf Deutsch. Sei knapp und präzise. Keine allgemeinen Ratschläge.\
"""

_USER_TEMPLATE = """\
Aktuelle Marktlage NQ/CME — {timestamp}

Preis: {current_price} | Spread: {spread} Punkte
VWAP: {vwap} ({price_vs_vwap:+.1f} Punkte)
Session: High {session_high} / Low {session_low}

ORDER BOOK L2:
- Top Bid Levels: {top_bids}
- Top Ask Levels: {top_asks}
- Imbalance Ratio: {imbalance_ratio:.2f} ({imbalance_direction})
- Große Orders (>50 Kontrakte): {large_orders}

SIGNAL ENGINE OUTPUT:
- Richtung: {direction} (Konfidenz: {confidence:.0%})
- Aktive Signale: {active_signals}
- Vorgeschlagene Entry-Zone: {entry_low}–{entry_high}
- Stop-Loss: {stop_loss} | Ziel 1: {target_1} | Ziel 2: {target_2}

Bewerte die Situation: Bestätigst du das Signal? Gibt es Warnzeichen im Order Book?
Antworte im Format:
URTEIL: [BESTÄTIGT / ABGELEHNT / WARTE]
BEGRÜNDUNG: [max 3 Sätze]
BEACHTUNG: [ein kritischer Punkt aus dem L2 Order Book]\
"""


def _load_api_key() -> str:
    cfg = configparser.ConfigParser()
    if _CONFIG_PATH.exists():
        cfg.read(_CONFIG_PATH)
        key = cfg.get("anthropic", "api_key", fallback="")
        if key and not key.startswith("YOUR_"):
            return key
    # Fallback to environment variable
    import os
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError(
            "Anthropic API key not found. Set it in config/credentials.ini "
            "[anthropic] api_key = ... or via ANTHROPIC_API_KEY env var."
        )
    return key


class ClaudeAnalyst:
    """
    Calls Claude for a structured trade verdict whenever the signal engine
    produces a high-confidence, multi-signal recommendation.

    Usage:
        analyst = ClaudeAnalyst()
        result = await analyst.analyze(signal_data)

    signal_data is the combined dict passed from the main loop:
        {
            "recommendation": engine.to_dict(rec),   # signal engine output
            "book_snapshot":  book.get_snapshot(),    # order book state
            "data_snapshot":  buf.get_analysis_snapshot(),  # buffer state
        }
    """

    def __init__(self) -> None:
        self._client: Optional[anthropic.AsyncAnthropic] = None
        self._last_call_ts: float = 0.0
        self._memory: deque = deque(maxlen=_MEMORY_SIZE)
        self._lock = asyncio.Lock()

        # Cost & cache tracking (reset on process restart, exposed via get_cost_stats)
        self.total_calls:        int   = 0
        self.cached_calls:       int   = 0
        self.estimated_cost_usd: float = 0.0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def analyze(self, signal_data: dict) -> dict:
        """
        Evaluate whether to call Claude, build the prompt, parse the response.

        Returns a result dict always — either a real Claude verdict or a
        skip dict explaining why the call was not made.
        """
        rec       = signal_data.get("recommendation", {})
        # Auto-detect data source: free mode if free_snapshot key present
        free_snap = signal_data.get("free_snapshot")
        book      = signal_data.get("book_snapshot", {})
        data      = signal_data.get("data_snapshot", {})

        skip_reason = self._should_skip(rec)
        if skip_reason:
            logger.debug("Claude call skipped: %s", skip_reason)
            return _skip_result(skip_reason)

        async with self._lock:
            # Double-check rate limit after acquiring lock
            elapsed = time.monotonic() - self._last_call_ts
            if elapsed < _MIN_INTERVAL:
                wait = _MIN_INTERVAL - elapsed
                logger.debug("Rate limit: waiting %.1fs before Claude call.", wait)
                await asyncio.sleep(wait)

            try:
                result = await self._call_api(rec, book, data, free_snap=free_snap)
            except Exception as exc:
                logger.error("Claude API error: %s", exc)
                return _error_result(str(exc))

            self._last_call_ts = time.monotonic()

        self._memory.append(result)
        logger.info(
            "Claude verdict: %s (conf=%.0f%%) — %s",
            result.get("verdict"),
            rec.get("confidence", 0) * 100,
            result.get("begruendung", "")[:80],
        )
        return result

    # ------------------------------------------------------------------
    # Memory access
    # ------------------------------------------------------------------

    def get_memory(self) -> list:
        """Return last ≤20 analysis results, newest first."""
        return list(reversed(self._memory))

    def clear_memory(self) -> None:
        self._memory.clear()

    # ------------------------------------------------------------------
    # Guard conditions
    # ------------------------------------------------------------------

    def _should_skip(self, rec: dict) -> Optional[str]:
        if not rec or rec.get("direction") == "NEUTRAL":
            return "no recommendation"

        confidence = rec.get("confidence", 0.0)
        if confidence < _MIN_CONFIDENCE:
            return f"confidence {confidence:.2f} < {_MIN_CONFIDENCE}"

        signals = rec.get("signals", [])
        unique_types = {s.get("type") for s in signals if s.get("type")}
        if len(unique_types) < _MIN_SIGNAL_TYPES:
            return f"only {len(unique_types)} distinct signal type(s), need {_MIN_SIGNAL_TYPES}"

        # Rate limit pre-check (non-blocking, accurate check happens inside lock)
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < _MIN_INTERVAL:
            return f"rate limit ({_MIN_INTERVAL - elapsed:.0f}s remaining)"

        return None

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    async def _call_api(
        self,
        rec: dict,
        book: dict,
        data: dict,
        free_snap: Optional[dict] = None,
    ) -> dict:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=_load_api_key())

        prompt = (
            _build_free_prompt(rec, free_snap)
            if free_snap
            else _build_prompt(rec, book, data)
        )
        ts_str = _utc_now()

        logger.info("[%s] Calling Claude (%s)…", ts_str, _MODEL)

        message = await self._client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        # Token counts — cache fields may be absent on older SDK versions
        input_tokens   = message.usage.input_tokens
        output_tokens  = message.usage.output_tokens
        cache_read     = getattr(message.usage, "cache_read_input_tokens",    0) or 0
        cache_creation = getattr(message.usage, "cache_creation_input_tokens", 0) or 0

        # Update running stats
        self.total_calls += 1
        if cache_read > 0:
            self.cached_calls += 1

        # Pricing: Input $3/MTok · Cache Read $0.30/MTok · Output $15/MTok
        call_cost = (
            (input_tokens   * 3.00) / 1_000_000
            + (cache_read   * 0.30) / 1_000_000
            + (output_tokens * 15.0) / 1_000_000
        )
        self.estimated_cost_usd += call_cost

        logger.info(
            "Cache — read: %d tok, created: %d tok | "
            "input: %d, output: %d | call cost: $%.5f | "
            "hits: %d/%d (%.0f%%)",
            cache_read, cache_creation,
            input_tokens, output_tokens,
            call_cost,
            self.cached_calls, self.total_calls,
            (self.cached_calls / self.total_calls * 100) if self.total_calls else 0,
        )

        raw_text = message.content[0].text if message.content else ""
        parsed   = _parse_response(raw_text)

        return {
            "timestamp":             ts_str,
            "verdict":               parsed["verdict"],
            "begruendung":           parsed["begruendung"],
            "beachtung":             parsed["beachtung"],
            "raw_response":          raw_text,
            "direction":             rec.get("direction"),
            "confidence":            rec.get("confidence"),
            "entry_zone":            rec.get("entry_zone"),
            "stop_loss":             rec.get("stop_loss"),
            "target_1":              rec.get("target_1"),
            "target_2":              rec.get("target_2"),
            "input_tokens":          input_tokens,
            "output_tokens":         output_tokens,
            "cache_read_tokens":     cache_read,
            "cache_creation_tokens": cache_creation,
            "skipped":               False,
        }

    # ------------------------------------------------------------------
    # Seconds until next allowed call (for UI display)
    # ------------------------------------------------------------------

    def seconds_until_next_call(self) -> float:
        elapsed = time.monotonic() - self._last_call_ts
        return max(0.0, _MIN_INTERVAL - elapsed)

    # ------------------------------------------------------------------
    # Cost & cache statistics (read by dashboard sidebar)
    # ------------------------------------------------------------------

    def get_cost_stats(self) -> dict:
        """Return cache efficiency and accumulated cost since process start."""
        hit_rate = (
            self.cached_calls / self.total_calls
            if self.total_calls else 0.0
        )
        return {
            "total_calls":        self.total_calls,
            "cached_calls":       self.cached_calls,
            "cache_hit_rate":     round(hit_rate, 3),
            "estimated_cost_usd": round(self.estimated_cost_usd, 5),
        }


# ------------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------------

def _build_prompt(rec: dict, book: dict, data: dict) -> str:
    current_price = data.get("last_price", 0.0)
    vwap          = data.get("vwap") or 0.0
    session_high  = data.get("session_high") or "–"
    session_low   = data.get("session_low")  or "–"
    spread        = book.get("spread")       or 0.0

    price_vs_vwap = current_price - vwap if vwap else 0.0

    # Format L2 ladder (top 5 each side)
    bid_ladder = book.get("bid_ladder", [])[:5]
    ask_ladder = book.get("ask_ladder", [])[:5]
    top_bids = "  ".join(f"{lv['price']:.2f}×{lv['size']}" for lv in bid_ladder) or "–"
    top_asks = "  ".join(f"{lv['price']:.2f}×{lv['size']}" for lv in ask_ladder) or "–"

    imbalance = book.get("imbalance_ratio", 0.5)
    imbalance_direction = (
        "Bid-lastig (bullish)"  if imbalance > 0.60 else
        "Ask-lastig (bearish)"  if imbalance < 0.40 else
        "Ausgeglichen"
    )

    large_orders = book.get("large_orders", [])
    if large_orders:
        lo_parts = [
            f"{o['side'].upper()} {o['size']}×{o['price']:.2f}"
            for o in large_orders[-5:]   # last 5 to keep prompt short
        ]
        large_orders_str = "  ".join(lo_parts)
    else:
        large_orders_str = "Keine"

    signals     = rec.get("signals", [])
    active_sigs = ", ".join(
        f"{s.get('type','?')}({s.get('confidence',0):.0%})" for s in signals
    ) or "–"

    entry_zone  = rec.get("entry_zone", {})
    entry_low   = entry_zone.get("low",  "–")
    entry_high  = entry_zone.get("high", "–")
    stop_loss   = rec.get("stop_loss") or "–"
    target_1    = rec.get("target_1")  or "–"
    target_2    = rec.get("target_2")  or "–"

    return _USER_TEMPLATE.format(
        timestamp           = _utc_now(),
        current_price       = f"{current_price:.2f}",
        spread              = f"{spread:.2f}",
        vwap                = f"{vwap:.2f}",
        price_vs_vwap       = price_vs_vwap,
        session_high        = session_high,
        session_low         = session_low,
        top_bids            = top_bids,
        top_asks            = top_asks,
        imbalance_ratio     = imbalance,
        imbalance_direction = imbalance_direction,
        large_orders        = large_orders_str,
        direction           = rec.get("direction", "–"),
        confidence          = rec.get("confidence", 0.0),
        active_signals      = active_sigs,
        entry_low           = entry_low,
        entry_high          = entry_high,
        stop_loss           = stop_loss,
        target_1            = target_1,
        target_2            = target_2,
    )


# ------------------------------------------------------------------
# Response parser
# ------------------------------------------------------------------

_VERDICT_RE     = re.compile(r"URTEIL\s*:\s*(BESTÄTIGT|ABGELEHNT|WARTE)", re.IGNORECASE)
_BEGRUENDUNG_RE = re.compile(r"BEGRÜNDUNG\s*:\s*(.+?)(?=BEACHTUNG\s*:|$)", re.DOTALL | re.IGNORECASE)
_BEACHTUNG_RE   = re.compile(r"BEACHTUNG\s*:\s*(.+?)$", re.DOTALL | re.IGNORECASE)


def _parse_response(text: str) -> dict:
    verdict_match     = _VERDICT_RE.search(text)
    begruendung_match = _BEGRUENDUNG_RE.search(text)
    beachtung_match   = _BEACHTUNG_RE.search(text)

    verdict     = verdict_match.group(1).upper() if verdict_match else "WARTE"
    begruendung = begruendung_match.group(1).strip() if begruendung_match else text.strip()
    beachtung   = beachtung_match.group(1).strip()   if beachtung_match   else ""

    # Normalise accented variant that some models produce
    if "BESTATIGT" in verdict or "BESTÄTIGT" in verdict:
        verdict = "BESTÄTIGT"

    return {
        "verdict":     verdict,
        "begruendung": begruendung,
        "beachtung":   beachtung,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


_FREE_USER_TEMPLATE = """\
Aktuelle Marktlage NQ Futures — {timestamp}

PREIS & SESSION:
Preis: {current_price} | Session: {session_change:+.2f}% | vs VWAP: {vs_vwap:+.1f} Pkt
Session High: {session_high} / Low: {session_low} | VWAP: {vwap}
Overnight Gap: {gap_str}

MULTI-TIMEFRAME ANALYSE:
5min:  EMA9 {ema9_5m} {cmp_5m} EMA21 {ema21_5m} → {bias_5m}
15min: EMA9 {ema9_15m} {cmp_15m} EMA21 {ema21_15m} → {bias_15m}

MARKTREGIME:
VIX: {vix:.1f} ({vix_regime}) | 10j Yield: {yield_str}
Wirtschaftskalender: {calendar_str}

SIGNAL ENGINE OUTPUT:
Richtung: {direction} (Konfidenz: {confidence:.0%})
Aktive Signale: {active_signals}
Entry-Zone: {entry_low}–{entry_high}
Stop-Loss: {stop_loss} | Ziel 1: {target_1} | Ziel 2: {target_2}

Bewerte die Situation: Bestätigst du das Signal?
Antworte im Format:
URTEIL: [BESTÄTIGT / ABGELEHNT / WARTE]
BEGRÜNDUNG: [max 3 Sätze]
BEACHTUNG: [ein kritischer Punkt — z.B. VIX-Niveau, Event-Risiko oder TF-Konflikt]\
"""


def _build_free_prompt(rec: dict, free_snap: dict) -> str:
    if not free_snap:
        free_snap = {}

    price          = free_snap.get("last_price", 0.0)
    vwap           = free_snap.get("session_vwap") or 0.0
    session_high   = free_snap.get("session_high") or "–"
    session_low    = free_snap.get("session_low")  or "–"
    session_chg    = free_snap.get("session_change_pct", 0.0)
    vs_vwap        = price - vwap if vwap else 0.0
    vix            = free_snap.get("vix", 0.0)
    vix_regime     = free_snap.get("vix_regime", "unknown")
    yield_10y      = free_snap.get("yield_10y", 0.0)

    # Gap description
    gap_pct = free_snap.get("overnight_gap_pct", 0.0)
    gap_dir = free_snap.get("overnight_gap_dir", "none")
    if gap_dir != "none":
        gap_str = f"{gap_dir.upper()} {abs(gap_pct):.2f}% (Fill-Ziel: {free_snap.get('yesterday_close', 0):.2f})"
    else:
        gap_str = "Kein signifikanter Gap"

    # EMAs from signal metadata
    ema9_5m = ema21_5m = ema9_15m = ema21_15m = "?"
    bias_5m = bias_15m = "–"
    cmp_5m = cmp_15m = "≈"
    for sig in rec.get("signals", []):
        if sig.get("type") == "MULTI_TF_BIAS":
            md = sig.get("metadata", {})
            if "ema9_5m" in md:
                ema9_5m   = f"{md['ema9_5m']:.2f}"
                ema21_5m  = f"{md['ema21_5m']:.2f}"
                ema9_15m  = f"{md['ema9_15m']:.2f}"
                ema21_15m = f"{md['ema21_15m']:.2f}"
                _d5  = md["ema9_5m"]  - md["ema21_5m"]
                _d15 = md["ema9_15m"] - md["ema21_15m"]
                cmp_5m   = ">" if _d5  > 0 else "<"
                cmp_15m  = ">" if _d15 > 0 else "<"
                bias_5m  = "BULLISH" if _d5  > 0 else "BEARISH"
                bias_15m = "BULLISH" if _d15 > 0 else "BEARISH"
            break

    # Calendar
    upcoming = [e for e in free_snap.get("upcoming_events", []) if e.get("impact") == "high"]
    if upcoming:
        calendar_str = " | ".join(
            f"{e.get('name','')} in {e.get('minutes_away', '?'):.0f} min"
            for e in upcoming[:3]
        )
    elif free_snap.get("event_window_active"):
        calendar_str = "Post-Event Wartezeit aktiv"
    else:
        calendar_str = "Keine High-Impact Events heute"

    yield_str = f"{yield_10y:.2f}%" if yield_10y else "–"

    # Signal engine fields
    signals    = rec.get("signals", [])
    active_str = ", ".join(
        f"{s.get('type','?')}({s.get('confidence',0):.0%})" for s in signals
    ) or "–"
    entry_zone = rec.get("entry_zone", {})
    entry_low  = entry_zone.get("low",  "–")
    entry_high = entry_zone.get("high", "–")
    sl         = rec.get("stop_loss") or "–"
    t1         = rec.get("target_1")  or "–"
    t2         = rec.get("target_2")  or "–"

    return _FREE_USER_TEMPLATE.format(
        timestamp      = _utc_now(),
        current_price  = f"{price:.2f}",
        session_change = session_chg,
        vs_vwap        = vs_vwap,
        vwap           = f"{vwap:.2f}" if vwap else "–",
        session_high   = f"{session_high:.2f}" if isinstance(session_high, float) else session_high,
        session_low    = f"{session_low:.2f}"  if isinstance(session_low,  float) else session_low,
        gap_str        = gap_str,
        ema9_5m        = ema9_5m,   cmp_5m  = cmp_5m,  ema21_5m  = ema21_5m,  bias_5m  = bias_5m,
        ema9_15m       = ema9_15m,  cmp_15m = cmp_15m, ema21_15m = ema21_15m, bias_15m = bias_15m,
        vix            = vix,
        vix_regime     = vix_regime.upper(),
        yield_str      = yield_str,
        calendar_str   = calendar_str,
        direction      = rec.get("direction", "–"),
        confidence     = rec.get("confidence", 0.0),
        active_signals = active_str,
        entry_low      = f"{entry_low:.2f}" if isinstance(entry_low, float) else entry_low,
        entry_high     = f"{entry_high:.2f}" if isinstance(entry_high, float) else entry_high,
        stop_loss      = f"{sl:.2f}" if isinstance(sl, float) else sl,
        target_1       = f"{t1:.2f}" if isinstance(t1, float) else t1,
        target_2       = f"{t2:.2f}" if isinstance(t2, float) else t2,
    )


def _skip_result(reason: str) -> dict:
    return {
        "timestamp":    _utc_now(),
        "verdict":      "ÜBERSPRUNGEN",
        "begruendung":  reason,
        "beachtung":    "",
        "raw_response": "",
        "skipped":      True,
        "skip_reason":  reason,
    }


def _error_result(error: str) -> dict:
    return {
        "timestamp":    _utc_now(),
        "verdict":      "FEHLER",
        "begruendung":  error,
        "beachtung":    "",
        "raw_response": "",
        "skipped":      False,
        "error":        error,
    }
