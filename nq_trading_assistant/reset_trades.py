"""
Reset Trade-History und Signal-Gewichtungen.
Sichert alte Daten vor dem Löschen.

Usage: python reset_trades.py
"""
import json
from pathlib import Path

LOGS = Path(__file__).parent / "logs"
LOGS.mkdir(exist_ok=True)

# ── Trades sichern und zurücksetzen ──────────────────────────────────────────
old_trades = LOGS / "simulated_trades.json"
backup     = LOGS / "simulated_trades_backup.json"

if old_trades.exists():
    backup.write_text(old_trades.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Backup erstellt: {backup}")

old_trades.write_text("[]", encoding="utf-8")
print("Trade-History zurückgesetzt")

# ── Gewichtungen auf optimierte Defaults zurücksetzen ────────────────────────
default_weights = {
    "MULTI_TF_BIAS":                      1.0,
    "FAIR_VALUE_GAP":                     1.2,
    "VIX_REGIME":                         0.8,
    "OVERNIGHT_GAP":                      0.9,
    "EMA_TREND":                          0.9,
    "VWAP_POSITION":                      1.1,
    "RSI_EXTREME":                        0.7,
    "MEAN_REVERSION":                     0.8,
    "SESSION_LEVELS":                     1.0,
    "ICT_CONFLUENCE":                     1.3,
    "_vix_high_penalty":                  0.75,
    "_vix_extreme_penalty":               0.40,
    "_time_open_bonus":                   1.30,
    "_time_close_penalty":                0.85,
    "outside_killzone_confidence_penalty": 0.80,
    "_last_updated":                      None,
    "_update_count":                      0,
    "_total_trades_analyzed":             0,
}

weights_file = LOGS / "signal_weights.json"
weights_file.write_text(
    json.dumps(default_weights, indent=2), encoding="utf-8"
)
print("Signal-Gewichtungen zurückgesetzt")
print()
print("Neu starten mit: python main.py")
