"""
Level 2 order book state for NQ Futures.

Processes all Rithmic update_type values and maintains a live, sorted
bid/ask book with real-time derived metrics.

Input format (from rithmic_client._normalise_book):
    {
        "type": "order_book",
        "update_type": str,          # CLEAR_ORDER_BOOK | BEGIN | MIDDLE | END | SOLO | SNAPSHOT_IMAGE | NO_BOOK
        "symbol": str,
        "bids": [{"price": float, "size": int}, ...],
        "asks": [{"price": float, "size": int}, ...],
        "timestamp": int,            # ssboe (seconds since beginning of epoch)
    }
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

LARGE_ORDER_THRESHOLD = 50  # contracts
DEFAULT_DEPTH = 10


# Update-type constants as sent by Rithmic
class UpdateType:
    CLEAR_ORDER_BOOK  = "CLEAR_ORDER_BOOK"
    BEGIN             = "BEGIN"
    MIDDLE            = "MIDDLE"
    END               = "END"
    SOLO              = "SOLO"       # single-message complete update
    SNAPSHOT_IMAGE    = "SNAPSHOT_IMAGE"
    NO_BOOK           = "NO_BOOK"


@dataclass
class LargeOrder:
    side:      str    # "bid" | "ask"
    price:     float
    size:      int
    timestamp: int


class OrderBook:
    """
    Full L2 order book with live metrics.

    Thread-safe via an internal Lock so the Streamlit UI thread
    can call get_snapshot() concurrently with the async feed.

    Typical call sequence:
        book = OrderBook("NQM5")
        book.apply_update(normalised_book_dict)   # called from on_order_book callback
        snapshot = book.get_snapshot()            # called from UI / signal engine
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._lock = Lock()

        # Sorted dicts — bids descending, asks ascending maintained via
        # sorted() on access; stored as plain {price: size} for O(1) writes.
        self._bids: Dict[float, int] = {}
        self._asks: Dict[float, int] = {}

        # Accumulate BEGIN / MIDDLE frames here, commit on END
        self._pending_bids: Dict[float, int] = {}
        self._pending_asks: Dict[float, int] = {}
        self._in_multi_frame = False

        self.last_timestamp: int = 0
        self.update_count:   int = 0

        # Recent large orders — rolling window of last 200
        self._large_orders: deque = deque(maxlen=200)

    # ------------------------------------------------------------------
    # Public update entry point
    # ------------------------------------------------------------------

    def apply_update(self, msg: dict) -> None:
        update_type = msg.get("update_type", UpdateType.SOLO).upper()
        timestamp   = msg.get("timestamp", 0)
        bids        = msg.get("bids", [])
        asks        = msg.get("asks", [])

        with self._lock:
            self.last_timestamp = timestamp

            if update_type == UpdateType.CLEAR_ORDER_BOOK:
                self._clear()

            elif update_type == UpdateType.NO_BOOK:
                # Exchange signalling no book available — clear and wait
                self._clear()
                logger.debug("[OrderBook] NO_BOOK received — book cleared.")

            elif update_type == UpdateType.SNAPSHOT_IMAGE:
                # Full replacement snapshot
                self._clear()
                self._apply_levels(bids, asks, timestamp)
                self._in_multi_frame = False

            elif update_type == UpdateType.SOLO:
                # Single-frame full update (most common during live trading)
                self._apply_levels(bids, asks, timestamp)

            elif update_type == UpdateType.BEGIN:
                # Start of a multi-frame update — buffer into pending
                self._pending_bids.clear()
                self._pending_asks.clear()
                self._buffer_levels(bids, asks)
                self._in_multi_frame = True

            elif update_type == UpdateType.MIDDLE:
                if self._in_multi_frame:
                    self._buffer_levels(bids, asks)

            elif update_type == UpdateType.END:
                if self._in_multi_frame:
                    self._buffer_levels(bids, asks)
                    # Commit accumulated pending levels
                    self._apply_levels(
                        [{"price": p, "size": s} for p, s in self._pending_bids.items()],
                        [{"price": p, "size": s} for p, s in self._pending_asks.items()],
                        timestamp,
                    )
                    self._pending_bids.clear()
                    self._pending_asks.clear()
                    self._in_multi_frame = False

            self.update_count += 1

    # ------------------------------------------------------------------
    # Real-time metrics (lock-free — always called from within _lock)
    # ------------------------------------------------------------------

    @property
    def best_bid(self) -> Optional[float]:
        return max(self._bids.keys(), default=None)

    @property
    def best_ask(self) -> Optional[float]:
        return min(self._asks.keys(), default=None)

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return round(ba - bb, 2)

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return round((bb + ba) / 2, 2)

    def bid_depth(self, n: int = DEFAULT_DEPTH) -> int:
        """Total size in top-N bid price levels."""
        top = sorted(self._bids.keys(), reverse=True)[:n]
        return sum(self._bids[p] for p in top)

    def ask_depth(self, n: int = DEFAULT_DEPTH) -> int:
        """Total size in top-N ask price levels."""
        top = sorted(self._asks.keys())[:n]
        return sum(self._asks[p] for p in top)

    @property
    def total_bid_volume(self) -> int:
        return sum(self._bids.values())

    @property
    def total_ask_volume(self) -> int:
        return sum(self._asks.values())

    @property
    def imbalance_ratio(self) -> float:
        """
        bid_vol / (bid_vol + ask_vol) across full book.
        Returns 0.5 when book is empty.
        0.0 = fully ask-heavy, 1.0 = fully bid-heavy.
        """
        bv = self.total_bid_volume
        av = self.total_ask_volume
        total = bv + av
        return round(bv / total, 4) if total else 0.5

    @property
    def top_n_imbalance(self) -> float:
        """imbalance_ratio restricted to top-10 levels on each side."""
        bv = self.bid_depth(DEFAULT_DEPTH)
        av = self.ask_depth(DEFAULT_DEPTH)
        total = bv + av
        return round(bv / total, 4) if total else 0.5

    @property
    def large_orders(self) -> List[LargeOrder]:
        return list(self._large_orders)

    # ------------------------------------------------------------------
    # Snapshot for UI / signal engine
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        with self._lock:
            bid_levels = sorted(self._bids.items(), reverse=True)[:DEFAULT_DEPTH]
            ask_levels = sorted(self._asks.items())[:DEFAULT_DEPTH]

            return {
                "symbol":           self.symbol,
                "timestamp":        self.last_timestamp,
                "update_count":     self.update_count,
                # Best prices
                "best_bid":         self.best_bid,
                "best_ask":         self.best_ask,
                "spread":           self.spread,
                "mid_price":        self.mid_price,
                # Depth metrics (top-10)
                "bid_depth_10":     self.bid_depth(10),
                "ask_depth_10":     self.ask_depth(10),
                # Full book totals
                "total_bid_volume": self.total_bid_volume,
                "total_ask_volume": self.total_ask_volume,
                # Imbalance
                "imbalance_ratio":  self.imbalance_ratio,
                "top10_imbalance":  self.top_n_imbalance,
                # Full top-N ladders for rendering
                "bid_ladder":       [{"price": p, "size": s} for p, s in bid_levels],
                "ask_ladder":       [{"price": p, "size": s} for p, s in ask_levels],
                # Large orders
                "large_orders": [
                    {"side": o.side, "price": o.price, "size": o.size,
                     "timestamp": o.timestamp}
                    for o in self._large_orders
                ],
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clear(self) -> None:
        self._bids.clear()
        self._asks.clear()

    def _apply_levels(
        self,
        bids: List[dict],
        asks: List[dict],
        timestamp: int,
    ) -> None:
        for level in bids:
            price = float(level["price"])
            size  = int(level["size"])
            if size == 0:
                self._bids.pop(price, None)
            else:
                self._bids[price] = size
                if size >= LARGE_ORDER_THRESHOLD:
                    self._large_orders.append(
                        LargeOrder("bid", price, size, timestamp)
                    )

        for level in asks:
            price = float(level["price"])
            size  = int(level["size"])
            if size == 0:
                self._asks.pop(price, None)
            else:
                self._asks[price] = size
                if size >= LARGE_ORDER_THRESHOLD:
                    self._large_orders.append(
                        LargeOrder("ask", price, size, timestamp)
                    )

    def _buffer_levels(self, bids: List[dict], asks: List[dict]) -> None:
        """Merge incoming levels into pending dicts during multi-frame updates."""
        for level in bids:
            price = float(level["price"])
            size  = int(level["size"])
            if size == 0:
                self._pending_bids.pop(price, None)
            else:
                self._pending_bids[price] = size

        for level in asks:
            price = float(level["price"])
            size  = int(level["size"])
            if size == 0:
                self._pending_asks.pop(price, None)
            else:
                self._pending_asks[price] = size
