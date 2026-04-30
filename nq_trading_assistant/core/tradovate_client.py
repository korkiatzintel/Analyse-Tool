"""
Tradovate Demo API client — REST auth + WebSocket market data.

Provides real-time NQ quotes (price, bid, ask) via the Tradovate Demo
environment. Authentication is done once at startup; the WebSocket
stream is kept alive via heartbeat pong responses.

Usage:
    client = TradovateClient(username, password)
    await client.authenticate()
    await client.get_front_month_symbol()
    await client.connect_marketdata()
    client.on_quote = my_async_handler   # fn(price, bid, ask)
    await client.listen()                # runs until disconnected
"""

import asyncio
import json
import logging
from typing import Callable, Optional

import aiohttp

DEMO_REST_URL = "https://demo.tradovateapi.com/v1"
DEMO_WS_URL   = "wss://md-demo.tradovateapi.com/v1/websocket"

logger = logging.getLogger(__name__)


class TradovateClient:
    def __init__(
        self,
        username: str,
        password: str,
        app_id: str = "Sample App",
        app_version: str = "1.0",
    ) -> None:
        self.username    = username
        self.password    = password
        self.app_id      = app_id
        self.app_version = app_version

        self._access_token: Optional[str]             = None
        self._ws:           Optional[aiohttp.ClientWebSocketResponse] = None
        self._session:      Optional[aiohttp.ClientSession]           = None

        # Callbacks
        self.on_quote: Optional[Callable] = None  # async fn(price, bid, ask)
        self.on_bar:   Optional[Callable] = None  # async fn(bar_dict)

        self._subscribed_symbol = "NQM6"  # updated by get_front_month_symbol()

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def authenticate(self) -> str:
        """Fetch access token from Tradovate Demo REST API."""
        self._session = aiohttp.ClientSession()
        payload = {
            "name":       self.username,
            "password":   self.password,
            "appId":      self.app_id,
            "appVersion": self.app_version,
            "cid":        0,
            "sec":        "",
        }
        async with self._session.post(
            f"{DEMO_REST_URL}/auth/accesstokenrequest",
            json=payload,
        ) as resp:
            data = await resp.json()

        if "accessToken" not in data:
            raise ConnectionError(f"Tradovate Auth fehlgeschlagen: {data}")

        self._access_token = data["accessToken"]
        logger.info("Tradovate: Authentifizierung erfolgreich")
        return self._access_token

    # ------------------------------------------------------------------
    # Contract lookup
    # ------------------------------------------------------------------

    async def get_front_month_symbol(self) -> str:
        """Resolve the active NQ front-month contract name (e.g. NQM6)."""
        try:
            async with self._session.get(
                f"{DEMO_REST_URL}/contract/find?name=NQ",
                headers={"Authorization": f"Bearer {self._access_token}"},
            ) as resp:
                contracts = await resp.json()

            if contracts and isinstance(contracts, list):
                self._subscribed_symbol = contracts[0].get("name", "NQM6")
                logger.info("Tradovate: Front-Month = %s", self._subscribed_symbol)
        except Exception:
            logger.warning(
                "Tradovate: Kontrakt-Lookup fehlgeschlagen — nutze %s",
                self._subscribed_symbol,
            )
        return self._subscribed_symbol

    # ------------------------------------------------------------------
    # WebSocket market data
    # ------------------------------------------------------------------

    async def connect_marketdata(self) -> None:
        """Open WebSocket connection and subscribe to NQ quotes."""
        self._ws = await self._session.ws_connect(DEMO_WS_URL)

        # Authenticate over WebSocket
        await self._ws.send_str(
            f'authorize\n1\n\n{{"token":"{self._access_token}"}}'
        )
        resp = await self._ws.receive()
        logger.info("Tradovate WS Auth: %s", (resp.data or "")[:100])

        # Subscribe to real-time quotes
        await self._ws.send_str(
            f'md/subscribeQuote\n2\n\n'
            f'{{"symbol":"{self._subscribed_symbol}"}}'
        )
        logger.info("Tradovate: Subscribed auf %s", self._subscribed_symbol)

    async def listen(self) -> None:
        """Receive and process WebSocket frames until connection closes."""
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                raw = msg.data
                if raw.startswith("a["):
                    try:
                        frames = json.loads(raw[1:])
                        for frame in frames:
                            await self._process_frame(frame)
                    except Exception as exc:
                        logger.debug("Frame parse error: %s", exc)
                elif raw == "h":
                    await self._ws.send_str("[]")   # heartbeat pong
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                logger.warning("Tradovate WS geschlossen")
                break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error("Tradovate WS Fehler: %s", msg.data)
                break

    async def _process_frame(self, frame: dict) -> None:
        """Dispatch a single decoded Tradovate frame to registered callbacks."""
        if not isinstance(frame, dict):
            return

        event_type = frame.get("e", "")
        data       = frame.get("d", {})

        if event_type == "md" and "quotes" in data:
            for quote in data["quotes"]:
                price = (quote.get("trade") or {}).get("price", 0)
                bid   = (quote.get("bid")   or {}).get("price", 0)
                ask   = (quote.get("ask")   or {}).get("price", 0)
                if price > 0 and self.on_quote:
                    await self.on_quote(price, bid, ask)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def disconnect(self) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("Tradovate: Verbindung getrennt")
