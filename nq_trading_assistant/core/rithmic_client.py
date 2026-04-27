"""
Rithmic connection manager for NQ Futures streaming.

Handles authentication, front-month contract discovery, L2 order book
and tick streaming, plus auto-reconnect with exponential backoff.
"""

import asyncio
import configparser
import logging
from pathlib import Path
from typing import Callable, Optional

from async_rithmic import RithmicClient, DataType, Exchange

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "credentials.ini"

# Reconnect parameters
_BACKOFF_BASE = 2.0       # seconds
_BACKOFF_MAX  = 120.0     # seconds
_MAX_RETRIES  = 10        # 0 = unlimited handled separately


def _load_credentials() -> dict:
    cfg = configparser.ConfigParser()
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"credentials.ini not found at {_CONFIG_PATH}. "
            "Copy credentials.ini.example and fill in your details."
        )
    cfg.read(_CONFIG_PATH)
    section = cfg["rithmic"]
    return {
        "user":         section["user"],
        "password":     section["password"],
        "system_name":  section["system_name"],
        "app_name":     section["app_name"],
        "app_version":  section["app_version"],
        "url":          section["url"],
    }


class RithmicConnectionManager:
    """
    Manages a persistent, auto-reconnecting Rithmic streaming session.

    Callbacks are plain async callables:
        on_tick(tick_data: dict)
        on_order_book(book_data: dict)
        on_time_bar(bar_data: dict)

    Usage:
        mgr = RithmicConnectionManager()
        mgr.on_tick      = my_tick_handler
        mgr.on_order_book = my_book_handler
        mgr.on_time_bar  = my_bar_handler
        await mgr.start_streaming()
    """

    def __init__(self) -> None:
        self._creds: dict = _load_credentials()
        self._client: Optional[RithmicClient] = None
        self._symbol: Optional[str] = None          # e.g. "NQM5"
        self._exchange: str = "CME"

        # Public callback hooks — replace before calling start_streaming()
        self.on_tick:       Optional[Callable] = None
        self.on_order_book: Optional[Callable] = None
        self.on_time_bar:   Optional[Callable] = None

        self._running = False
        self._attempt = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def start_streaming(self) -> None:
        """Connect, discover front-month contract, subscribe, and stream."""
        self._running = True
        self._attempt = 0

        while self._running:
            try:
                await self._connect_and_stream()
                # Clean disconnect — stop retrying
                break
            except asyncio.CancelledError:
                logger.info("Streaming cancelled — shutting down.")
                self._running = False
                break
            except Exception as exc:
                if not self._running:
                    break
                self._attempt += 1
                wait = min(_BACKOFF_BASE ** self._attempt, _BACKOFF_MAX)
                logger.error(
                    "[attempt %d] Connection lost: %s — retrying in %.0fs",
                    self._attempt, exc, wait,
                )
                await asyncio.sleep(wait)

    async def stop(self) -> None:
        self._running = False
        if self._client:
            try:
                await self._client.disconnect()
                logger.info("Disconnected from Rithmic.")
            except Exception as exc:
                logger.warning("Error during disconnect: %s", exc)

    # ------------------------------------------------------------------
    # Internal connect / stream lifecycle
    # ------------------------------------------------------------------

    async def _connect_and_stream(self) -> None:
        logger.info(
            "[%s] Connecting to %s as '%s' (system=%s)…",
            _ts(), self._creds["url"], self._creds["user"],
            self._creds["system_name"],
        )

        self._client = RithmicClient(
            user=self._creds["user"],
            password=self._creds["password"],
            system_name=self._creds["system_name"],
            app_name=self._creds["app_name"],
            app_version=self._creds["app_version"],
            url=self._creds["url"],
        )

        await self._client.connect()
        logger.info("[%s] Connected. Authenticating…", _ts())

        await self._client.login()
        logger.info("[%s] Authenticated successfully.", _ts())

        # Reset backoff counter on successful auth
        self._attempt = 0

        self._symbol = await self._resolve_front_month()
        logger.info("[%s] Front-month contract: %s", _ts(), self._symbol)

        await self._subscribe()
        logger.info(
            "[%s] Subscriptions active — streaming %s on %s.",
            _ts(), self._symbol, self._exchange,
        )

        # Dispatch incoming messages until disconnected
        async for message in self._client.listen():
            if not self._running:
                break
            await self._dispatch(message)

    # ------------------------------------------------------------------
    # Front-month discovery
    # ------------------------------------------------------------------

    async def _resolve_front_month(self) -> str:
        """
        Request the front-month NQ contract from Rithmic.

        async_rithmic exposes instrument search; we pick the nearest
        expiry in the CME NQ product family.
        """
        logger.info("[%s] Resolving front-month NQ contract…", _ts())

        instruments = await self._client.get_instrument_by_underlying(
            symbol="NQ",
            exchange=self._exchange,
        )

        if not instruments:
            raise RuntimeError("No NQ instruments returned from Rithmic.")

        # Sort by expiry ascending, pick first (nearest)
        instruments.sort(key=lambda i: getattr(i, "expiry_date", ""))
        front = instruments[0]
        symbol = getattr(front, "symbol", None) or getattr(front, "ticker", None)

        if not symbol:
            raise RuntimeError(f"Could not extract symbol from instrument: {front}")

        return symbol

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    async def _subscribe(self) -> None:
        # Level 1 ticks
        await self._client.subscribe_to_market_data(
            symbol=self._symbol,
            exchange=self._exchange,
            data_type=DataType.LAST_TRADE,
        )
        logger.info("[%s] Subscribed: last-trade ticks.", _ts())

        # Level 2 order book
        await self._client.subscribe_to_market_data(
            symbol=self._symbol,
            exchange=self._exchange,
            data_type=DataType.ORDER_BOOK,
        )
        logger.info("[%s] Subscribed: L2 order book.", _ts())

        # 1-minute time bars
        await self._client.subscribe_to_time_bar_data(
            symbol=self._symbol,
            exchange=self._exchange,
            bar_type="minute",
            bar_type_specifier=1,
        )
        logger.info("[%s] Subscribed: 1-minute time bars.", _ts())

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, message) -> None:
        msg_type = getattr(message, "msg_type", None) or type(message).__name__

        try:
            if "trade" in str(msg_type).lower() or "tick" in str(msg_type).lower():
                if self.on_tick:
                    await self.on_tick(_normalise_tick(message))

            elif "book" in str(msg_type).lower() or "depth" in str(msg_type).lower():
                if self.on_order_book:
                    await self.on_order_book(_normalise_book(message))

            elif "bar" in str(msg_type).lower():
                if self.on_time_bar:
                    await self.on_time_bar(_normalise_bar(message))

        except Exception as exc:
            logger.warning("Callback error for msg_type=%s: %s", msg_type, exc)


# ------------------------------------------------------------------
# Normalisation helpers — map Rithmic proto objects → plain dicts
# ------------------------------------------------------------------

def _normalise_tick(msg) -> dict:
    return {
        "type":      "tick",
        "symbol":    _attr(msg, "symbol", ""),
        "price":     _attr(msg, "trade_price", _attr(msg, "price", 0.0)),
        "volume":    _attr(msg, "trade_size",  _attr(msg, "volume", 0)),
        "side":      _attr(msg, "aggressor_side", ""),
        "timestamp": _attr(msg, "ssboe", 0),
    }


def _normalise_book(msg) -> dict:
    bids = getattr(msg, "bids", []) or []
    asks = getattr(msg, "asks", []) or []
    return {
        "type":   "order_book",
        "symbol": _attr(msg, "symbol", ""),
        "bids":   [{"price": b.price, "size": b.size} for b in bids],
        "asks":   [{"price": a.price, "size": a.size} for a in asks],
        "timestamp": _attr(msg, "ssboe", 0),
    }


def _normalise_bar(msg) -> dict:
    return {
        "type":      "time_bar",
        "symbol":    _attr(msg, "symbol", ""),
        "open":      _attr(msg, "open_price",  0.0),
        "high":      _attr(msg, "high_price",  0.0),
        "low":       _attr(msg, "low_price",   0.0),
        "close":     _attr(msg, "close_price", 0.0),
        "volume":    _attr(msg, "volume",      0),
        "timestamp": _attr(msg, "bar_end_ssboe", _attr(msg, "ssboe", 0)),
    }


def _attr(obj, name: str, default):
    return getattr(obj, name, default)


def _ts() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
