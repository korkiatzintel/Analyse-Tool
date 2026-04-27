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
import logging
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — file + console, set up before any import that might log
# ---------------------------------------------------------------------------

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "app.log"

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
    logger.warning("async_rithmic not installed (%s) — running in demo mode.", _e)
    _RITHMIC_AVAILABLE = False

from core.order_book import OrderBook
from core.data_buffer import DataBuffer
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

        # Most recent outputs — read by the UI process via shared-memory or
        # simply via its own independently initialised components.
        self.last_rec:    dict = {}
        self.last_claude: dict = {}
        self.connected:   bool = False
        self.contract:    str  = "–"


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
# Demo mode — synthetic NQ data when Rithmic is unavailable
# ---------------------------------------------------------------------------

async def _stream_demo() -> None:
    """
    Generate synthetic NQ tick / book / bar data so the pipeline and UI
    stay exercised without a live Rithmic connection.
    """
    import random
    from datetime import datetime, timezone

    logger.info("Demo mode: generating synthetic NQ data.")
    _APP.connected = True
    _APP.contract  = "NQM5 (demo)"

    price   = 19_000.0
    ts      = int(datetime.now(timezone.utc).timestamp())
    bar_vol = 0
    bar_buy = 0
    bar_sell = 0
    bar_open = price

    while True:
        try:
            # --- Tick ---
            move  = random.gauss(0, 0.5)
            price = round(max(18_000.0, price + move), 2)
            ts   += 1
            side  = "B" if random.random() > 0.48 else "S"
            vol   = random.randint(1, 15)

            tick = {
                "type": "tick", "symbol": "NQM5",
                "price": price, "volume": vol,
                "side": side, "timestamp": ts,
            }
            await _on_tick(tick)

            bar_vol  += vol
            bar_open  = bar_open or price
            if side == "B":
                bar_buy += vol
            else:
                bar_sell += vol

            # --- Order book update (every tick) ---
            bids, asks = _gen_demo_book(price)
            await _on_order_book({
                "type": "order_book", "update_type": "SOLO",
                "symbol": "NQM5", "timestamp": ts,
                "bids": bids, "asks": asks,
            })

            # --- 1-second bar (every ~10 ticks ≈ 1s at 10Hz) ---
            if ts % 10 == 0:
                bar = {
                    "type": "time_bar", "symbol": "NQM5",
                    "open":  bar_open,
                    "high":  price + random.uniform(0, 2),
                    "low":   price - random.uniform(0, 2),
                    "close": price,
                    "volume": bar_vol,
                    "timestamp": ts,
                }
                await _on_time_bar(bar)
                bar_vol = bar_buy = bar_sell = 0
                bar_open = price

            await asyncio.sleep(0.1)   # ~10 ticks/sec

        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Demo stream error")
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

async def _async_main(demo: bool) -> None:
    stream_coro = _stream_demo() if demo else _stream_live()

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
            "  python main.py              # live Rithmic + Streamlit UI\n"
            "  python main.py --no-ui      # headless, engine only\n"
            "  python main.py --demo       # synthetic data, no Rithmic needed\n"
        ),
    )
    parser.add_argument("--no-ui",  action="store_true", help="Skip Streamlit dashboard")
    parser.add_argument("--demo",   action="store_true", help="Force demo / synthetic data mode")
    args = parser.parse_args()

    # Decide whether to use demo mode
    demo = args.demo or not _RITHMIC_AVAILABLE
    if demo and not args.demo:
        logger.info("async_rithmic unavailable — switching to demo mode automatically.")

    logger.info("=" * 60)
    logger.info("NQ Trading Assistant starting up")
    logger.info("  Mode    : %s", "DEMO" if demo else "LIVE (Rithmic)")
    logger.info("  UI      : %s", "disabled" if args.no_ui else "http://localhost:8501")
    logger.info("  Log     : %s", _LOG_FILE)
    logger.info("=" * 60)

    # Start Streamlit in a child process (non-blocking)
    ui_proc = None
    if not args.no_ui:
        ui_proc = _start_dashboard()

    # Build and run the asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_signal_handlers(loop)

    try:
        loop.run_until_complete(_async_main(demo=demo))
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
