"""
NQ Futures Day Trading Assistant — entry point.

Usage:
    python main.py [--no-ui]

    --no-ui   Start the streaming engine without the Streamlit dashboard
              (useful for headless / testing scenarios).

The process owns two concurrent activities:
  1. asyncio event loop  — Rithmic streaming + signal pipeline
  2. subprocess          — Streamlit dashboard (unless --no-ui)

CTRL-C triggers a graceful shutdown: Rithmic is disconnected cleanly,
the Streamlit subprocess is terminated, and the asyncio loop exits.
"""

import argparse
import asyncio
import collections
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — file + console, set up before any import that might log
# ---------------------------------------------------------------------------

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE     = _LOG_DIR / "app.log"
_STATE_FILE   = _LOG_DIR / "ui_state.json"    # read by dashboard.py
_COMMAND_FILE = _LOG_DIR / "ui_command.json"  # written by dashboard.py "force" button

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
# Suppress overly verbose third-party loggers
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Project imports (after logging is configured)
# ---------------------------------------------------------------------------

try:
    from core.rithmic_client import RithmicConnectionManager
    _RITHMIC_AVAILABLE = True
except ImportError as _e:
    logger.warning("async_rithmic not installed (%s) — L2 mode unavailable.", _e)
    _RITHMIC_AVAILABLE = False

from core.order_book import OrderBook
from core.data_buffer import DataBuffer
from core.free_data_client import FreeDataClient
from signals.order_flow import OrderFlowAnalyzer
from signals.signal_engine import SignalEngine
from ai.claude_analyst import ClaudeAnalyst

# ---------------------------------------------------------------------------
# Application state — shared between async loop and UI subprocess
# ---------------------------------------------------------------------------

class AppComponents:
    """All live components, initialised once and reused across reconnects."""

    def __init__(self) -> None:
        self.order_book   = OrderBook("NQ")
        self.data_buffer  = DataBuffer()
        self.of_analyzer  = OrderFlowAnalyzer()
        self.signal_engine = SignalEngine()
        self.claude       = ClaudeAnalyst()
        self.rithmic: "RithmicConnectionManager | None" = None

        # Most recent outputs — shared with the UI process via _STATE_FILE.
        self.last_rec:    dict = {}
        self.last_claude: dict = {}
        self.connected:   bool = False
        self.contract:    str  = "–"
        self.mode:        str  = "demo"

        # Free-data extras (populated by _stream_free on_market_context)
        self.last_free_snap:    dict                = {}
        self.last_free_poll_ts: float               = 0.0
        self.delta_history:     collections.deque   = collections.deque(maxlen=120)


_APP = AppComponents()

# ---------------------------------------------------------------------------
# Callback pipeline
# ---------------------------------------------------------------------------

async def _on_tick(tick: dict) -> None:
    """L1 tick → DataBuffer only."""
    try:
        _APP.data_buffer.on_tick(tick)
    except Exception:
        logger.exception("on_tick error")


async def _on_order_book(msg: dict) -> None:
    """L2 update → OrderBook state machine → DataBuffer (future: footprint)."""
    try:
        _APP.order_book.apply_update(msg)
    except Exception:
        logger.exception("on_order_book error")

    # NO_BOOK from exchange — warn once per session
    if msg.get("update_type", "").upper() == "NO_BOOK":
        logger.warning(
            "Exchange signalled NO_BOOK — L2 unavailable. "
            "Signals will fall back to tick-only mode."
        )


async def _on_time_bar(bar: dict) -> None:
    """
    1-minute bar → DataBuffer → SignalEngine → (optionally) ClaudeAnalyst.

    This is the main signal-evaluation trigger.  We call it on every
    completed bar so the pipeline runs at ~1-minute cadence, well within
    the 30-second Claude rate limit.
    """
    try:
        _APP.data_buffer.on_time_bar(bar)
    except Exception:
        logger.exception("on_time_bar DataBuffer error")
        return

    await _evaluate_and_analyze()


async def _evaluate_and_analyze() -> None:
    """Run the full signal → AI pipeline and cache results on _APP."""
    try:
        book_snap = _APP.order_book.get_snapshot()
        data_snap = _APP.data_buffer.get_analysis_snapshot()

        rec = _APP.signal_engine.evaluate(book_snap, data_snap)
        _APP.last_rec = _APP.signal_engine.to_dict(rec)

        if rec is not None:
            signal_data = {
                "recommendation": _APP.last_rec,
                "book_snapshot":  book_snap,
                "data_snapshot":  data_snap,
            }
            result = await _APP.claude.analyze(signal_data)
            if not result.get("skipped"):
                _APP.last_claude = result
                _log_verdict(result)

        _write_ui_state()

    except Exception:
        logger.exception("Signal/AI pipeline error — continuing.")


def _log_verdict(result: dict) -> None:
    verdict = result.get("verdict", "?")
    direction = result.get("direction", "?")
    conf = result.get("confidence", 0.0)
    logger.info(
        "Claude: %-12s | %-5s | conf=%.0f%% | %s",
        verdict, direction, conf * 100,
        result.get("begruendung", "")[:120],
    )


# ---------------------------------------------------------------------------
# Shared UI state writer
# ---------------------------------------------------------------------------

def _write_ui_state() -> None:
    """
    Atomically serialize current app state to _STATE_FILE so the Streamlit
    dashboard can read it on every 5-second rerun cycle.

    Uses a .tmp → rename pattern so the dashboard never reads a partial file.
    """
    try:
        book_snap = _APP.order_book.get_snapshot()
        data_snap = _APP.data_buffer.get_analysis_snapshot()
        free      = _APP.last_free_snap   # {} when not in free mode

        # Rolling delta history (cumulative delta of last completed 1s bar)
        _APP.delta_history.append(data_snap.get("cumulative_delta_last1", 0))

        # Time until next yfinance poll (free mode only)
        next_update_in = None
        if _APP.mode == "free" and _APP.last_free_poll_ts:
            elapsed = time.monotonic() - _APP.last_free_poll_ts
            next_update_in = round(max(0.0, 60.0 - elapsed), 1)

        # Prefer free-data session metrics when available (more accurate)
        last_price = free.get("last_price") or data_snap.get("last_price", 0.0)
        vwap       = free.get("session_vwap") or data_snap.get("vwap")
        s_high     = free.get("session_high") or data_snap.get("session_high")
        s_low      = free.get("session_low")  or data_snap.get("session_low")

        state = {
            "mode":             _APP.mode,
            "contract":         _APP.contract,
            "connected":        _APP.connected,
            "last_update":      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "next_update_in":   next_update_in,
            "market": {
                "last_price":      last_price,
                "vwap":            vwap,
                "session_high":    s_high,
                "session_low":     s_low,
                "spread":          book_snap.get("spread"),
                "imbalance_ratio": book_snap.get("imbalance_ratio", 0.5),
                "bid_ladder":      book_snap.get("bid_ladder", [])[:5],
                "ask_ladder":      book_snap.get("ask_ladder", [])[:5],
            },
            "signals":                   _APP.last_rec,
            "claude":                    _APP.last_claude,
            "claude_memory":             _APP.claude.get_memory()[:5],
            "claude_cost_stats":         _APP.claude.get_cost_stats(),
            "vix":                       free.get("vix", 0.0),
            "vix_regime":                free.get("vix_regime", "unknown"),
            "yield_10y":                 free.get("yield_10y", 0.0),
            "upcoming_events":           free.get("upcoming_events", []),
            "event_window_active":       free.get("event_window_active", False),
            "minutes_to_next_event":     free.get("minutes_to_next_event"),
            "delta_history":             list(_APP.delta_history),
        }

        tmp = _STATE_FILE.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, default=str)
        tmp.replace(_STATE_FILE)

    except Exception:
        logger.debug("_write_ui_state failed", exc_info=True)


def _read_command() -> dict:
    """
    Read and consume the UI command file written by dashboard.py's
    'Jetzt analysieren' button. Returns {} if no command is pending.
    """
    try:
        if not _COMMAND_FILE.exists():
            return {}
        with _COMMAND_FILE.open("r", encoding="utf-8") as f:
            cmd = json.load(f)
        _COMMAND_FILE.unlink(missing_ok=True)
        return cmd
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Rithmic streaming coroutine
# ---------------------------------------------------------------------------

async def _stream_live() -> None:
    """Connect to Rithmic, wire callbacks, stream until shutdown."""
    mgr = RithmicConnectionManager()
    mgr.on_tick       = _on_tick
    mgr.on_order_book = _on_order_book
    mgr.on_time_bar   = _on_time_bar
    _APP.rithmic      = mgr

    try:
        _APP.connected = True
        await mgr.start_streaming()
    finally:
        _APP.connected = False
        _APP.rithmic   = None


# ---------------------------------------------------------------------------
# Free-data mode — yfinance + FRED + economic calendar
# ---------------------------------------------------------------------------

async def _stream_free() -> None:
    """
    Use FreeDataClient (yfinance / FRED / calendar) as the data source.
    Evaluates signals via SignalEngine.evaluate_multi_tf() and passes
    free_snapshot to ClaudeAnalyst for the adapted prompt.
    """
    client = FreeDataClient()

    async def on_tick(tick: dict) -> None:
        await _on_tick(tick)

    async def on_time_bar(bar: dict) -> None:
        try:
            _APP.data_buffer.on_time_bar(bar)
        except Exception:
            logger.exception("FreeDataClient on_time_bar DataBuffer error")

    async def on_market_context(ctx: dict) -> None:
        """Full evaluation cycle triggered on every poll (60s)."""
        try:
            free_snap = client.get_snapshot()
            rec = _APP.signal_engine.evaluate_multi_tf(free_snap)
            _APP.last_rec = _APP.signal_engine.to_dict(rec)
            _APP.contract = "NQ=F (yfinance)"
            _APP.connected = True
            _APP.last_free_snap    = free_snap
            _APP.last_free_poll_ts = time.monotonic()

            if rec is not None:
                result = await _APP.claude.analyze({
                    "recommendation": _APP.last_rec,
                    "free_snapshot":  free_snap,
                })
                if not result.get("skipped"):
                    _APP.last_claude = result
                    _log_verdict(result)

            _write_ui_state()
        except Exception:
            logger.exception("Free-data signal/AI pipeline error — continuing.")

    client.on_tick           = on_tick
    client.on_time_bar       = on_time_bar
    client.on_market_context = on_market_context

    try:
        await client.start_streaming()
    finally:
        _APP.connected = False


# ---------------------------------------------------------------------------
# Synthetic demo mode (no network required)
# ---------------------------------------------------------------------------

async def _stream_demo() -> None:
    """
    Generate synthetic NQ data for UI testing without any network access.
    Falls back automatically if yfinance is unavailable.
    """
    import random

    logger.info("Synthetic demo mode: generating fake NQ data.")
    _APP.connected = True
    _APP.contract  = "NQM5 (synthetic)"

    price    = 19_000.0
    ts       = int(__import__("time").time())
    bar_open = price

    while True:
        try:
            price  = round(max(18_000.0, price + random.gauss(0, 0.5)), 2)
            ts    += 1
            vol    = random.randint(1, 15)
            side   = "B" if random.random() > 0.48 else "S"

            await _on_tick({"type": "tick", "symbol": "NQM5",
                            "price": price, "volume": vol,
                            "side": side, "timestamp": ts})

            bids, asks = _gen_demo_book(price)
            await _on_order_book({"type": "order_book", "update_type": "SOLO",
                                  "symbol": "NQM5", "timestamp": ts,
                                  "bids": bids, "asks": asks})

            if ts % 10 == 0:
                await _on_time_bar({
                    "type": "time_bar", "symbol": "NQM5",
                    "open":  bar_open,  "high": price + random.uniform(0, 2),
                    "low":   price - random.uniform(0, 2), "close": price,
                    "volume": vol * 10, "timestamp": ts,
                })
                bar_open = price
                _write_ui_state()

            await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Synthetic demo error")
            await asyncio.sleep(1)

    _APP.connected = False


def _gen_demo_book(price: float) -> tuple:
    import random
    bids, asks = [], []
    # Occasionally inject a heavy bid side to trigger signals
    bias = random.random()
    for i in range(10):
        bid_size = random.randint(5, 150) + (100 if bias > 0.75 else 0)
        ask_size = random.randint(5, 150) + (100 if bias < 0.25 else 0)
        bids.append({"price": round(price - (i + 1) * 0.25, 2), "size": bid_size})
        asks.append({"price": round(price + (i + 1) * 0.25, 2), "size": ask_size})
    return bids, asks


# ---------------------------------------------------------------------------
# Streamlit subprocess
# ---------------------------------------------------------------------------

_ui_proc: "subprocess.Popen | None" = None


def _start_dashboard() -> "subprocess.Popen":
    """Launch the Streamlit dashboard as a child process."""
    ui_path = Path(__file__).parent / "ui" / "dashboard.py"
    if not ui_path.exists():
        logger.warning("ui/dashboard.py not found — skipping UI launch.")
        return None

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ui_path),
        "--server.headless", "true",
        "--server.port", "8501",
        "--theme.base", "dark",
        "--logger.level", "error",
    ]
    logger.info("Starting Streamlit dashboard: http://localhost:8501")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def _stop_dashboard(proc: "subprocess.Popen | None") -> None:
    if proc is None or proc.poll() is not None:
        return
    logger.info("Stopping Streamlit dashboard (PID %d)…", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_event = asyncio.Event()


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Register SIGINT / SIGTERM handlers that trigger clean shutdown."""

    def _request_shutdown(signame: str) -> None:
        logger.info("Received %s — initiating graceful shutdown…", signame)
        loop.call_soon_threadsafe(_shutdown_event.set)

    # Windows does not support add_signal_handler; use signal.signal instead
    if sys.platform == "win32":
        signal.signal(signal.SIGINT,  lambda *_: _request_shutdown("SIGINT"))
        signal.signal(signal.SIGTERM, lambda *_: _request_shutdown("SIGTERM"))
    else:
        loop.add_signal_handler(signal.SIGINT,  lambda: _request_shutdown("SIGINT"))
        loop.add_signal_handler(signal.SIGTERM, lambda: _request_shutdown("SIGTERM"))


async def _shutdown_hook() -> None:
    """
    Wait for shutdown signal, then cleanly stop the Rithmic stream.
    Runs concurrently with _stream_* via asyncio.gather().
    """
    await _shutdown_event.wait()
    logger.info("Shutdown hook running…")
    if _APP.rithmic is not None:
        try:
            await _APP.rithmic.stop()
            logger.info("Rithmic disconnected cleanly.")
        except Exception:
            logger.exception("Error during Rithmic disconnect.")


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

async def _async_main(mode: str) -> None:
    _APP.mode = mode
    if mode == "live":
        stream_coro = _stream_live()
    elif mode == "free":
        stream_coro = _stream_free()
    else:
        stream_coro = _stream_demo()

    try:
        # Run streaming + shutdown watcher concurrently.
        # When shutdown_hook completes (signal received), we cancel streaming.
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(stream_coro,    name="stream"),
                asyncio.create_task(_shutdown_hook(), name="shutdown"),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    except asyncio.CancelledError:
        pass

    logger.info("Async loop finished.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NQ Futures Day Trading Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py              # yfinance free-data + Streamlit UI\n"
            "  python main.py --live       # live Rithmic L2 data\n"
            "  python main.py --demo       # synthetic data, no network needed\n"
            "  python main.py --no-ui      # headless, engine only\n"
        ),
    )
    parser.add_argument("--no-ui", action="store_true", help="Skip Streamlit dashboard")
    parser.add_argument("--live",  action="store_true", help="Use Rithmic L2 (requires credentials)")
    parser.add_argument("--demo",  action="store_true", help="Synthetic data mode, no network")
    args = parser.parse_args()

    # Determine run mode: free (default) → live → demo (fallback)
    if args.demo:
        mode = "demo"
    elif args.live:
        if not _RITHMIC_AVAILABLE:
            logger.warning("--live requested but async_rithmic not installed — using free mode.")
            mode = "free"
        else:
            mode = "live"
    else:
        mode = "free"   # default: yfinance + FRED + calendar

    logger.info("=" * 60)
    logger.info("NQ Trading Assistant starting up")
    logger.info("  Mode    : %s", mode.upper())
    logger.info("  UI      : %s", "disabled" if args.no_ui else "http://localhost:8501")
    logger.info("  Log     : %s", _LOG_FILE)
    logger.info("=" * 60)

    ui_proc = None
    if not args.no_ui:
        ui_proc = _start_dashboard()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_signal_handlers(loop)

    try:
        loop.run_until_complete(_async_main(mode=mode))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in main thread.")
    finally:
        # Clean up remaining tasks
        pending = asyncio.all_tasks(loop)
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

        loop.close()
        _stop_dashboard(ui_proc)
        logger.info("Shutdown complete. Goodbye.")


if __name__ == "__main__":
    main()
