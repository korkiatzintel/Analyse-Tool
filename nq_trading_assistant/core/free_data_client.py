"""
Free market data client for NQ Futures.

Data sources (all free, no subscription required):
  1. yfinance  — NQ=F bars at 1m / 5m / 15m plus ^VIX, ^TNX, QQQ, SPY
  2. FRED API  — VIXCLS, DGS10 (optional: add [fred] api_key to credentials.ini)
  3. Investing.com — economic calendar via POST endpoint (daily cache)

Interface mirrors RithmicConnectionManager so main.py can switch between them:
    client.on_tick(tick_dict)
    client.on_time_bar(bar_dict)   — emitted for every new 1m bar
    client.on_market_context(ctx)  — emitted after every poll cycle

Polling cadence: yfinance every 60 s, calendar once per day, FRED every 6 h.
"""

import asyncio
import configparser
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "credentials.ini"
_POLL_INTERVAL    = 60       # seconds between yfinance polls
_CALENDAR_TTL     = 86_400   # seconds — refresh calendar once per day
_FRED_TTL         = 21_600   # seconds — refresh FRED every 6 hours
_EVENT_WARN_MINS  = 30       # minutes before event to block signals
_POST_EVENT_WAIT  = 10       # minutes to wait after high-impact event

# yfinance ticker map
_TICKERS = {
    "nq":     "NQ=F",
    "vix":    "^VIX",
    "tnx":    "^TNX",  # 10-year yield
    "qqq":    "QQQ",
    "spy":    "SPY",
}


# ---------------------------------------------------------------------------
# Lazy imports — so the module loads even if packages are missing
# ---------------------------------------------------------------------------

def _import_yfinance():
    try:
        import yfinance as yf
        return yf
    except ImportError:
        raise RuntimeError(
            "yfinance not installed. Run: pip install yfinance"
        )


def _import_fredapi():
    try:
        import fredapi
        return fredapi
    except ImportError:
        return None


def _import_bs4():
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# FreeDataClient
# ---------------------------------------------------------------------------

class FreeDataClient:
    """
    Polls yfinance + FRED + economic calendar and exposes a get_snapshot()
    dict that the signal engine can consume directly.

    Also supports the same callback interface as RithmicConnectionManager
    so main.py can wire it into the existing pipeline:
        client.on_tick       = my_handler
        client.on_time_bar   = my_handler
        client.on_market_context = my_handler
    """

    def __init__(self) -> None:
        # Callbacks (same contract as RithmicConnectionManager)
        self.on_tick:            Optional[Callable] = None
        self.on_time_bar:        Optional[Callable] = None
        self.on_market_context:  Optional[Callable] = None

        # Internal bar storage (keyed by timeframe string)
        self._bars: Dict[str, List[dict]] = {"1m": [], "5m": [], "15m": [], "1h": []}
        self._last_bar_ts: Dict[str, int] = {"1m": 0, "5m": 0, "15m": 0, "1h": 0}

        # Realtime price override (Barchart fallback, updated every 10s)
        self._last_price: float = 0.0
        self._realtime_last_update: str = ""

        # Tradovate real-time feed state
        self._tradovate             = None
        self._tradovate_price: float = 0.0
        self._tradovate_bid:   float = 0.0
        self._tradovate_ask:   float = 0.0
        self._tradovate_connected:   bool = False

        # Config reader (for Tradovate credentials)
        self._config = configparser.ConfigParser()
        if _CONFIG_PATH.exists():
            self._config.read(_CONFIG_PATH)

        # Session accumulators
        self._session_date:        Optional[date] = None
        self._session_vwap_sum_pv: float = 0.0
        self._session_vwap_sum_v:  float = 0.0
        self._session_high:        float = float("-inf")
        self._session_low:         float = float("inf")
        self._yesterday_close:     float = 0.0
        self._today_open:          float = 0.0

        # Macro / context
        self._vix:        float = 0.0
        self._yield_10y:  float = 0.0
        self._events:     List[dict] = []

        # FRED
        self._fred_client = None
        self._last_fred_fetch: float = 0.0

        # Calendar cache
        self._calendar_date:  Optional[date] = None
        self._last_event_end: Optional[datetime] = None

        self._running = False

    # ------------------------------------------------------------------
    # Public: streaming loop (mirrors RithmicConnectionManager)
    # ------------------------------------------------------------------

    async def start_streaming(self) -> None:
        """Poll all data sources until stop() is called."""
        self._running = True
        logger.info("FreeDataClient: starting polling loop (interval=%ds)", _POLL_INTERVAL)

        # Try Tradovate real-time feed first; fall back to Barchart scraper
        tradovate_ok = await self._init_tradovate()
        if tradovate_ok:
            asyncio.create_task(self._tradovate.listen())
        else:
            # Barchart fallback (every 10 s, ~5 s delay)
            asyncio.create_task(self._realtime_price_loop())

        # Warm-up: initial fetch with history
        await self._initial_fetch()

        while self._running:
            try:
                await self._poll_cycle()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("FreeDataClient poll error — retrying in %ds", _POLL_INTERVAL)

            await asyncio.sleep(_POLL_INTERVAL)

        logger.info("FreeDataClient: stopped.")

    async def stop(self) -> None:
        self._running = False
        if self._tradovate:
            await self._tradovate.disconnect()

    # ------------------------------------------------------------------
    # Tradovate integration
    # ------------------------------------------------------------------

    async def _init_tradovate(self) -> bool:
        """
        Attempt to connect Tradovate Demo real-time feed.
        Returns True on success, False when credentials are absent or auth fails.
        """
        import os
        from dotenv import load_dotenv
        load_dotenv(override=True)

        username = os.getenv("TRADOVATE_USERNAME")
        password = os.getenv("TRADOVATE_PASSWORD", "")
        password = password.strip('"').strip("'").strip()

        if not username:
            username = self._config.get("tradovate", "username", fallback=None)
        if not password:
            raw = self._config.get("tradovate", "password", fallback=None)
            if raw:
                password = raw.strip('"').strip("'").strip()

        logger.info(
            f"Tradovate: PW length={len(password)}, ends with={password[-1] if password else 'None'}"
        )

        if not username or not password or username.startswith("deine@"):
            logger.warning(
                "Tradovate: Keine Credentials — nutze yfinance/Barchart als Preisfeed. "
                "Setze TRADOVATE_USERNAME und TRADOVATE_PASSWORD in .env"
            )
            return False

        try:
            from core.tradovate_client import TradovateClient

            app_id  = (
                os.getenv("TRADOVATE_APP_ID")
                or self._config.get("tradovate", "app_id", fallback="Sample App")
            )
            app_ver = (
                os.getenv("TRADOVATE_APP_VERSION")
                or self._config.get("tradovate", "app_version", fallback="1.0")
            )

            self._tradovate = TradovateClient(username, password, app_id, app_ver)
            await self._tradovate.authenticate()
            await self._tradovate.get_front_month_symbol()
            await self._tradovate.connect_marketdata()

            async def on_quote(price: float, bid: float, ask: float) -> None:
                self._tradovate_price      = price
                self._tradovate_bid        = bid
                self._tradovate_ask        = ask
                self._tradovate_connected  = True
                self._realtime_last_update = _utc_now()

            self._tradovate.on_quote = on_quote
            logger.info("Tradovate: Echtzeit-Feed aktiv ✓")
            return True

        except Exception as exc:
            logger.error("Tradovate Init fehlgeschlagen: %s", exc)
            logger.info("Fallback: nutze Barchart/yfinance Preisfeed")
            self._tradovate = None
            return False

    def _get_current_price(self) -> float:
        """Return best available last price (Tradovate > Barchart > yfinance bar)."""
        if self._tradovate_connected and self._tradovate_price > 0:
            return self._tradovate_price
        if self._last_price > 0:
            return self._last_price
        return self._bars["1m"][-1]["close"] if self._bars["1m"] else 0.0

    # ------------------------------------------------------------------
    # Realtime price feed (Barchart, ~5 s delay, every 10 s)
    # ------------------------------------------------------------------

    async def _realtime_price_loop(self) -> None:
        """Update self._last_price every 10 s from Barchart public quotes."""
        while self._running:
            price = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_realtime_price
            )
            if price > 0:
                yf_price = self._bars["1m"][-1]["close"] if self._bars["1m"] else 0.0
                if yf_price > 0 and abs(price - yf_price) > 5:
                    logger.info(
                        "Realtime price %s differs from yfinance %s by %.1f pts",
                        price, yf_price, abs(price - yf_price),
                    )
                self._last_price = price
                self._realtime_last_update = _utc_now()
            await asyncio.sleep(10)

    def _fetch_realtime_price(self) -> float:
        """
        Fetch current NQ price from Barchart public page (~5 s delay).
        Returns 0.0 on any failure so the caller can fall back gracefully.
        """
        import re
        import requests as req
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36"
                )
            }
            r = req.get(
                "https://www.barchart.com/futures/quotes/NQ*0/futures-prices",
                headers=headers,
                timeout=5,
            )
            match = re.search(r'"lastPrice"\s*:\s*"?([\d,]+\.?\d*)"?', r.text)
            if match:
                return float(match.group(1).replace(",", ""))
        except Exception as exc:
            logger.debug("Realtime price fetch failed: %s", exc)
        return 0.0

    # ------------------------------------------------------------------
    # Public: snapshot for signal engine
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """Return all current market data as a single dict."""
        last_price = self._get_current_price()
        yesterday_close = self._yesterday_close or last_price

        # Overnight gap
        gap_pct = 0.0
        if yesterday_close and self._today_open:
            gap_pct = (self._today_open - yesterday_close) / yesterday_close * 100

        vix_regime = _vix_to_regime(self._vix)
        upcoming   = self._upcoming_events()
        event_active, mins_to_next = self._event_status()

        session_change_pct = 0.0
        if yesterday_close and last_price:
            session_change_pct = (last_price - yesterday_close) / yesterday_close * 100

        return {
            "timestamp":           _utc_now(),
            "last_price":          last_price,
            "session_change_pct":  round(session_change_pct, 3),
            "session_high":        self._session_high if self._session_high != float("-inf") else None,
            "session_low":         self._session_low  if self._session_low  != float("inf")  else None,
            "yesterday_close":     yesterday_close,
            "today_open":          self._today_open,
            "session_vwap":        self._vwap(),
            "overnight_gap_pct":   round(gap_pct, 3),
            "overnight_gap_dir":   "up" if gap_pct > 0.3 else "down" if gap_pct < -0.3 else "none",
            "bars_1m":             list(self._bars["1m"][-60:]),
            "bars_5m":             list(self._bars["5m"][-30:]),
            "bars_15m":            list(self._bars["15m"][-20:]),
            "bars_1h":             list(self._bars["1h"][-200:]),
            "vix":                 self._vix,
            "vix_regime":          vix_regime,
            "yield_10y":           self._yield_10y,
            "upcoming_events":     upcoming,
            "event_window_active": event_active,
            "minutes_to_next_event": mins_to_next,
            # last_price duplicated for DataBuffer compatibility
            "last_volume":         self._bars["1m"][-1].get("volume", 0) if self._bars["1m"] else 0,
            # Tradovate real-time fields (zero/False when not connected)
            "bid":                 self._tradovate_bid,
            "ask":                 self._tradovate_ask,
            "spread":              (
                round(self._tradovate_ask - self._tradovate_bid, 2)
                if self._tradovate_bid > 0 and self._tradovate_ask > 0 else None
            ),
            "tradovate_connected":    self._tradovate_connected,
            "realtime_price":         self._tradovate_price or self._last_price,
            "realtime_last_update":   self._realtime_last_update,
        }

    # ------------------------------------------------------------------
    # Fetch lifecycle
    # ------------------------------------------------------------------

    async def _initial_fetch(self) -> None:
        """Fetch several days of history on startup."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._fetch_yfinance, "5d")
        await loop.run_in_executor(None, self._fetch_calendar)
        await loop.run_in_executor(None, self._fetch_fred)
        logger.info(
            "FreeDataClient warm-up: 1m=%d bars, 5m=%d bars, 15m=%d bars, 1h=%d bars, VIX=%.1f",
            len(self._bars["1m"]), len(self._bars["5m"]),
            len(self._bars["15m"]), len(self._bars["1h"]),
            self._vix,
        )
        # Emit context immediately after warm-up so ui_state.json is written
        # without waiting for the first 60-second poll cycle.
        if self.on_market_context:
            try:
                await self.on_market_context(self.get_snapshot())
            except Exception:
                logger.exception("on_market_context (warm-up) callback error")

    async def _poll_cycle(self) -> None:
        loop = asyncio.get_event_loop()
        new_bars = await loop.run_in_executor(None, self._fetch_yfinance, "1d")

        # Calendar: refresh once per day
        today = date.today()
        if self._calendar_date != today:
            await loop.run_in_executor(None, self._fetch_calendar)
            self._calendar_date = today

        # FRED: refresh every 6 hours
        if time.monotonic() - self._last_fred_fetch > _FRED_TTL:
            await loop.run_in_executor(None, self._fetch_fred)

        # Emit callbacks for new bars
        await self._emit_new_bars(new_bars)

        # Emit market context
        ctx = self.get_snapshot()
        if self.on_market_context:
            try:
                await self.on_market_context(ctx)
            except Exception:
                logger.exception("on_market_context callback error")

    # ------------------------------------------------------------------
    # yfinance fetcher
    # ------------------------------------------------------------------

    def _fetch_yfinance(self, period: str) -> Dict[str, List[dict]]:
        yf = _import_yfinance()
        new_bars: Dict[str, List[dict]] = {"1m": [], "5m": [], "15m": [], "1h": []}

        # 1h uses a fixed 60-day window to capture ~200 hourly bars
        tf_configs = [
            ("1m",  "1m",  period),
            ("5m",  "5m",  period),
            ("15m", "15m", period),
            ("1h",  "1h",  "60d"),
        ]

        for tf, interval, tf_period in tf_configs:
            try:
                df = yf.download(
                    _TICKERS["nq"],
                    period=tf_period,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    prepost=False,
                )
                if df.empty:
                    continue

                df = _flatten_df(df)
                if df.empty:
                    continue

                df = df.dropna(subset=["Close"])
                bars  = _df_to_bars(df, tf)
                last  = self._last_bar_ts[tf]
                fresh = [b for b in bars if b["timestamp"] > last]

                self._bars[tf].extend(fresh)
                # Rolling window: 200 bars for 1h, 500 for others
                limit = 200 if tf == "1h" else 500
                self._bars[tf] = self._bars[tf][-limit:]

                if fresh:
                    self._last_bar_ts[tf] = fresh[-1]["timestamp"]
                    new_bars[tf] = fresh

                    if tf == "1m":
                        self._update_session(df, bars)

            except Exception:
                logger.exception("yfinance fetch error (interval=%s)", interval)

        # Fetch VIX and 10y yield
        self._fetch_vix_from_yf(yf)
        self._fetch_yield_from_yf(yf)

        return new_bars

    def _fetch_vix_from_yf(self, yf) -> None:
        try:
            df = yf.download("^VIX", period="2d", interval="1m",
                             progress=False, auto_adjust=True)
            df = _flatten_df(df)
            if not df.empty:
                self._vix = float(df["Close"].dropna().iloc[-1])
        except Exception:
            logger.debug("VIX fetch error: %s", exc_info=True)

    def _fetch_yield_from_yf(self, yf) -> None:
        try:
            df = yf.download("^TNX", period="2d", interval="1d",
                             progress=False, auto_adjust=True)
            df = _flatten_df(df)
            if not df.empty:
                self._yield_10y = float(df["Close"].dropna().iloc[-1])
        except Exception:
            logger.debug("TNX fetch error: %s", exc_info=True)

    def _update_session(self, df: pd.DataFrame, bars: List[dict]) -> None:
        """Recompute session VWAP, high/low, yesterday close, today open."""
        today = date.today()

        if self._session_date != today:
            # New session — reset accumulators
            self._session_date    = today
            self._session_vwap_sum_pv = 0.0
            self._session_vwap_sum_v  = 0.0
            self._session_high    = float("-inf")
            self._session_low     = float("inf")
            self._today_open      = 0.0

        # Find yesterday's close and today's open from the DataFrame
        try:
            if hasattr(df.index, "tz_localize"):
                idx = df.index.tz_convert("UTC") if df.index.tzinfo else df.index.tz_localize("UTC")
            else:
                idx = df.index

            today_utc    = pd.Timestamp(today).tz_localize("UTC")
            today_bars_df = df[idx >= today_utc]
            yest_bars_df  = df[idx <  today_utc]

            if not yest_bars_df.empty:
                self._yesterday_close = float(yest_bars_df["Close"].dropna().iloc[-1])

            if not today_bars_df.empty and self._today_open == 0.0:
                self._today_open = float(today_bars_df["Open"].dropna().iloc[0])
        except Exception:
            logger.debug("Session date split error", exc_info=True)

        # Accumulate VWAP from today's bars
        for bar in bars:
            ts = datetime.fromtimestamp(bar["timestamp"], tz=timezone.utc).date()
            if ts == today:
                tp  = (bar["open"] + bar["high"] + bar["low"] + bar["close"]) / 4
                vol = bar.get("volume", 0)
                self._session_vwap_sum_pv += tp * vol
                self._session_vwap_sum_v  += vol
                self._session_high = max(self._session_high, bar["high"])
                self._session_low  = min(self._session_low,  bar["low"])

    # ------------------------------------------------------------------
    # Economic calendar (investing.com)
    # ------------------------------------------------------------------

    def _fetch_calendar(self) -> None:
        try:
            self._events = _scrape_investing_calendar()
            logger.info(
                "Economic calendar: %d high-impact USD events today.",
                sum(1 for e in self._events if e.get("impact") == "high")
            )
        except Exception:
            logger.warning("Calendar fetch failed — running without event filter.", exc_info=True)
            self._events = []

    # ------------------------------------------------------------------
    # FRED API (optional)
    # ------------------------------------------------------------------

    def _fetch_fred(self) -> None:
        fredapi_mod = _import_fredapi()
        if fredapi_mod is None:
            return

        key = _load_fred_key()
        if not key:
            return

        try:
            fred = fredapi_mod.Fred(api_key=key)
            self._last_fred_fetch = time.monotonic()

            # VIXCLS — end-of-day VIX, more reliable than intraday
            vix_s = fred.get_series("VIXCLS")
            if vix_s is not None and not vix_s.empty:
                self._vix = float(vix_s.dropna().iloc[-1])

            # DGS10 — 10-year constant maturity yield
            y10_s = fred.get_series("DGS10")
            if y10_s is not None and not y10_s.empty:
                self._yield_10y = float(y10_s.dropna().iloc[-1])

            logger.info("FRED: VIX=%.1f, 10y=%.2f%%", self._vix, self._yield_10y)
        except Exception:
            logger.warning("FRED fetch error", exc_info=True)

    # ------------------------------------------------------------------
    # Emit callbacks for new bars
    # ------------------------------------------------------------------

    async def _emit_new_bars(self, new_bars: Dict[str, List[dict]]) -> None:
        for bar in new_bars.get("1m", []):
            # Synthesize a tick from each new 1m bar close
            if self.on_tick:
                tick = {
                    "type":      "tick",
                    "symbol":    "NQ=F",
                    "price":     bar["close"],
                    "volume":    bar.get("volume", 0),
                    "side":      "",
                    "timestamp": bar["timestamp"],
                }
                try:
                    await self.on_tick(tick)
                except Exception:
                    logger.exception("on_tick callback error")

            if self.on_time_bar:
                try:
                    await self.on_time_bar(bar)
                except Exception:
                    logger.exception("on_time_bar callback error")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vwap(self) -> Optional[float]:
        if self._session_vwap_sum_v == 0:
            return None
        return round(self._session_vwap_sum_pv / self._session_vwap_sum_v, 2)

    def _upcoming_events(self) -> List[dict]:
        """Return today's events with computed minutesToStart field."""
        now = datetime.now(timezone.utc)
        result = []
        for ev in self._events:
            try:
                ev_time = ev.get("time_utc")
                if ev_time:
                    diff = (ev_time - now).total_seconds() / 60
                    result.append({**ev, "minutes_away": round(diff, 1)})
            except Exception:
                result.append(ev)
        return result

    def _event_status(self):
        """Returns (event_window_active: bool, minutes_to_next: float|None)."""
        now = datetime.now(timezone.utc)

        # Check post-event wait
        if self._last_event_end:
            elapsed = (now - self._last_event_end).total_seconds() / 60
            if elapsed < _POST_EVENT_WAIT:
                return True, None

        # Look for upcoming high-impact events within the warning window
        min_away = None
        for ev in self._events:
            if ev.get("impact") != "high":
                continue
            ev_time = ev.get("time_utc")
            if not ev_time:
                continue
            diff_min = (ev_time - now).total_seconds() / 60
            if -2 <= diff_min <= _EVENT_WARN_MINS:   # -2 = just passed
                if diff_min < 0:
                    # Event just occurred — start post-event wait
                    self._last_event_end = now
                    return True, None
                if min_away is None or diff_min < min_away:
                    min_away = diff_min

        if min_away is not None and min_away <= _EVENT_WARN_MINS:
            return True, round(min_away, 1)

        return False, min_away


# ---------------------------------------------------------------------------
# Economic calendar scraper
# ---------------------------------------------------------------------------

def _scrape_investing_calendar() -> List[dict]:
    """
    Fetches today's economic events from investing.com.
    Returns list of dicts with keys: time, time_utc, name, impact, country.
    Falls back to empty list on any error.
    """
    import requests as req

    today_str = date.today().strftime("%Y-%m-%d")
    url = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest",
        "Referer":          "https://www.investing.com/economic-calendar/",
        "Content-Type":     "application/x-www-form-urlencoded",
        "Accept":           "*/*",
    }
    payload = {
        "country[]":    "5",           # United States
        "dateFrom":     today_str,
        "dateTo":       today_str,
        "timeZone":     "8",           # Eastern Time
        "timeFilter":   "timeRemain",
        "currentTab":   "today",
        "submitFilters": "1",
        "limit_from":   "0",
    }

    try:
        resp = req.post(url, headers=headers, data=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        html = data.get("data", "")
    except Exception:
        logger.debug("investing.com POST failed", exc_info=True)
        return []

    return _parse_calendar_html(html)


def _parse_calendar_html(html: str) -> List[dict]:
    BS = _import_bs4()
    if not BS or not html:
        return []

    try:
        soup = BS(html, "html.parser")
        events = []
        rows = soup.find_all("tr", class_=lambda c: c and "js-event-item" in c)

        for row in rows:
            # Impact: bull icons — 3 = high, 2 = medium, 1 = low
            bulls = row.find_all("i", class_="grayFullBullishIcon")
            impact = "high" if len(bulls) >= 3 else "medium" if len(bulls) == 2 else "low"
            if impact != "high":
                continue

            # Time cell
            time_td = row.find("td", class_="time")
            time_str = time_td.get_text(strip=True) if time_td else ""

            # Event name
            name_td = row.find("td", class_="event")
            name = name_td.get_text(strip=True) if name_td else ""

            if not name:
                continue

            # Convert to UTC (assuming Eastern Time)
            time_utc = _parse_event_time(time_str)

            events.append({
                "time":     time_str,
                "time_utc": time_utc,
                "name":     name,
                "impact":   impact,
                "country":  "USD",
            })

        return events
    except Exception:
        logger.debug("Calendar HTML parse error", exc_info=True)
        return []


def _parse_event_time(time_str: str) -> Optional[datetime]:
    """Parse 'HH:MM' Eastern Time string to UTC datetime."""
    if not time_str or ":" not in time_str:
        return None
    try:
        from zoneinfo import ZoneInfo
        today = date.today()
        h, m = int(time_str[:2]), int(time_str[3:5])
        eastern = datetime(today.year, today.month, today.day, h, m,
                           tzinfo=ZoneInfo("America/New_York"))
        return eastern.astimezone(timezone.utc)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns returned by yfinance for futures symbols
    (e.g. ('Close', 'NQ=F') → 'Close') and normalise to Title Case."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        col_map = {c.lower(): c for c in df.columns}
        if "close" not in col_map:
            logger.warning("_flatten_df: no Close column — columns: %s", list(df.columns))
            return pd.DataFrame()
        rename = {
            col_map[std.lower()]: std
            for std in ("Close", "Open", "High", "Low", "Volume")
            if std.lower() in col_map and col_map[std.lower()] != std
        }
        if rename:
            df = df.rename(columns=rename)
    return df


def _df_to_bars(df: pd.DataFrame, timeframe: str) -> List[dict]:
    bars = []
    for ts, row in df.iterrows():
        # Handle both tz-aware and tz-naive timestamps
        if hasattr(ts, "timestamp"):
            unix_ts = int(ts.timestamp())
        else:
            unix_ts = int(pd.Timestamp(ts).timestamp())

        # Flatten MultiIndex columns if present (yfinance >= 0.2.38)
        def _col(name: str):
            for c in df.columns:
                col_str = c[0] if isinstance(c, tuple) else c
                if col_str.lower() == name.lower():
                    return float(row[c]) if pd.notna(row[c]) else 0.0
            return 0.0

        bars.append({
            "type":      "time_bar",
            "symbol":    "NQ=F",
            "timeframe": timeframe,
            "open":      _col("Open"),
            "high":      _col("High"),
            "low":       _col("Low"),
            "close":     _col("Close"),
            "volume":    _col("Volume"),
            "timestamp": unix_ts,
        })
    return bars


def _vix_to_regime(vix: float) -> str:
    if vix <= 0:
        return "unknown"
    if vix < 15:
        return "low"
    if vix < 25:
        return "normal"
    if vix <= 30:
        return "high"
    return "extreme"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _load_fred_key() -> str:
    cfg = configparser.ConfigParser()
    if _CONFIG_PATH.exists():
        cfg.read(_CONFIG_PATH)
        key = cfg.get("fred", "api_key", fallback="")
        if key and not key.startswith("YOUR_"):
            return key
    import os
    return os.getenv("FRED_API_KEY", "")
