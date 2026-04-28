# NQ Futures Day Trading Assistant

KI-gestützte Echtzeit-Analyse für E-Mini Nasdaq 100 (NQ) Futures mit
Claude AI — drei Betriebsmodi, kein Rithmic-Account für den Start nötig.

> ⚠️ **Wichtiger Hinweis: Dieses Tool führt KEINE Orders aus.**
> Es ist ausschließlich für Paper Trading und Analyse-Zwecke gedacht.
> Alle Signale sind informativ — keine automatische Order-Ausführung.

---

## Betriebsmodi

| Modus | Befehl | Benötigt | Datenquelle |
|---|---|---|---|
| **Free** (Standard) | `python main.py` | `ANTHROPIC_API_KEY` | yfinance (1m/5m/15m) + FRED + Kalender |
| **Demo** | `python main.py --demo` | – | Synthetische NQ-Daten, kein Netzwerk |
| **Live** | `python main.py --live` | Rithmic-Account + `ANTHROPIC_API_KEY` | Rithmic L2 Order Book (Echtzeit) |

---

## Quick Start (Free Mode)

```bash
git clone <repo-url>
cd nq_trading_assistant

# Abhängigkeiten
pip install -r requirements.txt

# API Key setzen
cp ../.env.example ../.env
# .env öffnen und ANTHROPIC_API_KEY eintragen

# Starten — Dashboard öffnet sich auf http://localhost:8501
python main.py
```

Für einen ersten Test **ohne** API Key:

```bash
python main.py --demo   # synthetische Daten, kein Account nötig
```

---

## Voraussetzungen

| Komponente | Mindestversion | Hinweis |
|---|---|---|
| Python | 3.11+ | |
| Anthropic API Key | – | Für Free- und Live-Modus · [console.anthropic.com](https://console.anthropic.com) |
| FRED API Key | – | Optional · kostenlos · Makrodaten (VIX, 10j Yield) |
| Rithmic Account | – | Nur für `--live` · CME L2 Market Depth Entitlement erforderlich |

---

## Installation

```bash
# 1. Repository klonen
git clone <repo-url>
cd nq_trading_assistant

# 2. Virtuelle Umgebung (empfohlen)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Abhängigkeiten installieren
pip install -r requirements.txt
```

> **Hinweis zu pandas-ta:** Falls die Installation fehlschlägt, enthält
> `signals/technical.py` native Fallback-Implementierungen für EMA, RSI
> und ATR auf Basis von `pandas` + `numpy`. Das Tool läuft ohne pandas-ta.

---

## Setup

### Umgebungsvariablen (empfohlen)

```bash
cp .env.example .env
# .env mit Texteditor öffnen und ANTHROPIC_API_KEY eintragen
```

`python-dotenv` lädt `.env` automatisch — oder Variablen direkt exportieren:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Rithmic Credentials (nur `--live`)

```bash
cp nq_trading_assistant/config/credentials.ini.example \
   nq_trading_assistant/config/credentials.ini
```

```ini
[rithmic]
user        = IHR_RITHMIC_USERNAME
password    = IHR_RITHMIC_PASSWORT
system_name = Rithmic Test          # oder: Rithmic 01
app_name    = nq_trading_assistant
app_version = 1.0
url         = rituz00100.rithmic.com:443

[anthropic]
api_key     = sk-ant-...            # alternativ: ANTHROPIC_API_KEY Env-Var
```

| System | URL |
|---|---|
| Rithmic Test / Paper | `rituz00100.rithmic.com:443` |
| Rithmic 01 (Live) | `rithmic01.rithmic.com:443` |

> `.env` und `credentials.ini` sind in `.gitignore` — werden **nie** committed.

---

## Start

### Free Mode — yfinance + FRED + Kalender (Standard)

```bash
cd nq_trading_assistant
python main.py           # gleichbedeutend mit: python main.py --free
```

Keine Rithmic-Verbindung nötig. Daten kommen von yfinance (1m/5m/15m Bars),
FRED (VIX, 10j Yield) und dem Investing.com Wirtschaftskalender.
Dashboard: **http://localhost:8501**

### Demo Mode — synthetische Daten, kein Netzwerk

```bash
python main.py --demo
```

Generiert zufällige NQ-Preisbewegungen und ein synthetisches L2 Order Book.
Ideal zum Testen der UI und Signal-Pipeline ohne Account oder Internet.

### Live Mode — Rithmic L2 Echtzeit

```bash
python main.py --live
```

Rithmic-Account mit CME L2 Market Depth Entitlement erforderlich.
Liefert echte Tick-Daten, L2 Order Book und 1-Minuten-Bars.

### Nur Engine, kein Streamlit

```bash
python main.py --no-ui
```

### Alle Optionen

```
python main.py --help

  --free    yfinance Free-Modus (Standard)
  --demo    Synthetische Daten, kein Netzwerk, kein API Key
  --live    Rithmic L2 Echtzeit (Credentials erforderlich)
  --no-ui   Streamlit Dashboard nicht starten (headless / Server-Betrieb)
```

---

## Projektstruktur

```
nq_trading_assistant/
├── config/
│   ├── credentials.ini.example   # Vorlage — kopieren und befüllen
│   └── credentials.ini           # Echte Zugangsdaten (nie committed)
├── core/
│   ├── rithmic_client.py         # Verbindungsmanager, Auto-Reconnect
│   ├── order_book.py             # L2 Order Book State Machine
│   └── data_buffer.py            # Tick/Bar Rolling Buffer, VWAP, Footprint
├── signals/
│   ├── order_flow.py             # Imbalance, Delta-Divergenz, Absorption
│   ├── technical.py              # VWAP, FVG, EMA, RSI, Session Levels
│   └── signal_engine.py          # Signal-Aggregator, Entry/SL/Target
├── ai/
│   └── claude_analyst.py         # Claude AI Integration (30s Rate Limit)
├── ui/
│   └── dashboard.py              # Streamlit Dashboard
├── logs/
│   └── app.log                   # Automatisch erstellt beim ersten Start
├── main.py                       # Einstiegspunkt
└── requirements.txt
```

---

## Signal-Pipeline

```
Rithmic WebSocket
      │
      ├─ on_tick ──────────────► DataBuffer (10 000 Ticks)
      │
      ├─ on_order_book ─────────► OrderBook (L2 State Machine)
      │                                │
      │                                └─ Imbalance, Large Orders
      │
      └─ on_time_bar ───────────► DataBuffer (500 Bars, VWAP, Footprint)
                                        │
                                        ▼
                                  SignalEngine
                                  ├─ OrderFlowAnalyzer
                                  │   ├─ BID_ASK_IMBALANCE
                                  │   ├─ DELTA_DIVERGENCE
                                  │   ├─ STACKED_IMBALANCES
                                  │   └─ LARGE_ORDER_ABSORPTION
                                  ├─ TechnicalAnalyzer
                                  │   ├─ VWAP_POSITION
                                  │   ├─ FAIR_VALUE_GAP
                                  │   ├─ EMA_TREND
                                  │   ├─ RSI_EXTREME
                                  │   └─ SESSION_LEVELS
                                  │
                                  └─ TradeRecommendation
                                        │ (confidence > 0.65)
                                        ▼
                                  ClaudeAnalyst
                                  └─ claude-sonnet-4-6
                                     URTEIL: BESTÄTIGT / ABGELEHNT / WARTE
```

---

## Fehlerbehandlung

| Szenario | Verhalten |
|---|---|
| Rithmic nicht erreichbar | Auto-Reconnect mit exponential Backoff (2s → 120s), UI zeigt "Offline" |
| L2 nicht verfügbar (NO_BOOK) | Warnung im Log, Signale nur auf Tick-Basis |
| Anthropic API Fehler | Letzte Analyse bleibt sichtbar, kein Crash, Fehler in `logs/app.log` |
| `credentials.ini` fehlt | `--live` fällt auf Free-Modus zurück, `--demo` läuft ohne Credentials |
| Netzwerkunterbrechung | Rithmic-Client reconnectet automatisch |

Alle Exceptions werden in `logs/app.log` geschrieben.

---

## Konfiguration der Signal-Schwellwerte

Die wichtigsten Parameter sind direkt in den Modulen als Konstanten definiert:

| Datei | Konstante | Standard | Bedeutung |
|---|---|---|---|
| `signals/signal_engine.py` | `_MIN_CONFIDENCE` | `0.65` | Mindest-Konfidenz für Empfehlung |
| `signals/signal_engine.py` | `_RISK_REWARD_T1` | `1.5` | Risk/Reward Ziel 1 |
| `signals/signal_engine.py` | `_RISK_REWARD_T2` | `2.5` | Risk/Reward Ziel 2 |
| `signals/order_flow.py` | `_IMBALANCE_BULL` | `0.70` | Bullish Imbalance Schwelle |
| `signals/order_flow.py` | `_LARGE_ORDER_CUTOFF` | `50` | Min. Kontrakte = "Large Order" |
| `ai/claude_analyst.py` | `_MIN_INTERVAL` | `30` | Sekunden zwischen API-Calls |

---

## Logs

```bash
# Live-Log verfolgen
tail -f logs/app.log

# Nur Signale und Claude-Urteile
grep -E "Signal|Claude|BESTÄTIGT|ABGELEHNT|WARTE" logs/app.log
```

---

## Häufige Probleme

**`ModuleNotFoundError: No module named 'async_rithmic'`**
→ `pip install async_rithmic` oder mit `--demo` starten.

**`FileNotFoundError: credentials.ini not found`**
→ `cp config/credentials.ini.example config/credentials.ini` und ausfüllen.
   Oder `python main.py --demo` für Demo-Modus ohne Credentials.

**Streamlit öffnet sich nicht**
→ Browser manuell auf http://localhost:8501 öffnen.
   Port 8501 belegt? In `main.py` die `--server.port` Zeile anpassen.

**`RuntimeError: Anthropic API key not found`**
→ In `credentials.ini` unter `[anthropic] api_key = sk-ant-...` eintragen,
   oder `ANTHROPIC_API_KEY` als Umgebungsvariable setzen.

---

## Disclaimer

Dieses Tool dient ausschließlich zu Bildungs- und Analysezwecken. Es stellt
keine Anlageberatung dar. Futures-Handel birgt erhebliche Verlustrisiken.
Testen Sie immer zuerst im Paper Trading, bevor Sie echtes Kapital einsetzen.
