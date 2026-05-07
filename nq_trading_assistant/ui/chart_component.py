"""
TradingView Lightweight Charts HTML component for Streamlit.

Usage:
    from ui.chart_component import build_chart_html
    import streamlit.components.v1 as components
    components.html(build_chart_html(bars, signals, limit_orders), height=500)
"""

import json
from datetime import datetime

import pandas as pd


def build_chart_html(
    bars: list,
    signals: dict,
    limit_orders: list,
    height: int = 480,
) -> str:
    """
    Build a TradingView Lightweight Charts HTML string embedded via CDN.
    Renders candlesticks + EMA9/EMA21 + VWAP + Entry/SL/TP price lines.
    """
    if not bars:
        return (
            "<div style='background:#1a1a2e;color:#888;"
            "padding:24px;font-family:sans-serif;height:480px;"
            "display:flex;align-items:center;justify-content:center;'>"
            "⏳ Warte auf Bars…</div>"
        )

    try:
        df = pd.DataFrame(bars[-200:])
    except Exception:
        return "<div style='color:#888;padding:24px'>Fehler beim Laden der Bars</div>"

    if "timestamp" not in df.columns or df.empty:
        return "<div style='color:#888;padding:24px'>Keine Timestamp-Daten</div>"

    # Parse + sort timestamps
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])

    if df.empty:
        return "<div style='color:#888;padding:24px'>Keine gültigen OHLC-Daten</div>"

    # EMAs
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    # VWAP (session-cumulative when volume available)
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        vol_sum = df["volume"].sum()
        if vol_sum > 0:
            df["_vp"]       = df["close"] * df["volume"]
            df["_cum_vp"]   = df["_vp"].cumsum()
            df["_cum_vol"]  = df["volume"].cumsum()
            df["vwap_calc"] = df["_cum_vp"] / df["_cum_vol"]

    # Build JSON series
    candle_data: list = []
    ema9_data:   list = []
    ema21_data:  list = []
    vwap_data:   list = []

    for _, row in df.iterrows():
        ts_unix = int(row["ts"].timestamp())
        candle_data.append({
            "time":  ts_unix,
            "open":  round(float(row["open"]),  2),
            "high":  round(float(row["high"]),  2),
            "low":   round(float(row["low"]),   2),
            "close": round(float(row["close"]), 2),
        })
        ema9_data.append({"time": ts_unix, "value": round(float(row["ema9"]),  2)})
        ema21_data.append({"time": ts_unix, "value": round(float(row["ema21"]), 2)})
        if "vwap_calc" in df.columns and not pd.isna(row.get("vwap_calc")):
            vwap_data.append({"time": ts_unix,
                               "value": round(float(row["vwap_calc"]), 2)})

    # Price lines for Entry / SL / TP
    price_lines: list = []
    markers:     list = []

    trade_setup = (signals or {}).get("trade_setup") or {}
    direction   = (signals or {}).get("direction", "NEUTRAL")
    confidence  = (signals or {}).get("confidence", 0.0)

    if trade_setup and direction != "NEUTRAL":
        entry = trade_setup.get("entry_price", 0) or 0
        sl    = trade_setup.get("stop_loss_price", 0) or 0
        tp1   = trade_setup.get("take_profit_1_price", 0) or 0
        tp2   = trade_setup.get("take_profit_2_price", 0) or 0
        entry_color = "#00ff88" if direction == "LONG" else "#ff4444"

        if entry:
            price_lines.append({
                "price": entry, "color": entry_color,
                "lineWidth": 2, "lineStyle": 0,
                "title": f"{'▲' if direction == 'LONG' else '▼'} ENTRY {entry:.2f}",
            })
        if sl:
            price_lines.append({
                "price": sl, "color": "#ff3333",
                "lineWidth": 1, "lineStyle": 1,
                "title": f"SL {sl:.2f}",
            })
        if tp1:
            price_lines.append({
                "price": tp1, "color": "#00cc44",
                "lineWidth": 1, "lineStyle": 2,
                "title": f"TP1 {tp1:.2f}",
            })
        if tp2:
            price_lines.append({
                "price": tp2, "color": "#00993a",
                "lineWidth": 1, "lineStyle": 2,
                "title": f"TP2 {tp2:.2f}",
            })

        if candle_data and entry:
            markers.append({
                "time":     candle_data[-1]["time"],
                "position": "belowBar" if direction == "LONG" else "aboveBar",
                "color":    entry_color,
                "shape":    "arrowUp" if direction == "LONG" else "arrowDown",
                "text":     f"{direction} {confidence:.0%}",
            })

    # Triggered limit order markers
    for order in (limit_orders or []):
        if order.get("status") == "TRIGGERED" and candle_data:
            markers.append({
                "time":     candle_data[-1]["time"],
                "position": "aboveBar",
                "color":    "#ffaa00",
                "shape":    "circle",
                "text":     f"⚡ {order.get('type', '')}",
            })

    # JS snippets for price lines and VWAP series
    price_line_js = "\n".join(
        f"candleSeries.createPriceLine({{"
        f"price:{pl['price']},color:'{pl['color']}',"
        f"lineWidth:{pl['lineWidth']},lineStyle:{pl['lineStyle']},"
        f"axisLabelVisible:true,title:'{pl['title']}'}}); "
        for pl in price_lines
    )

    vwap_js = ""
    if vwap_data:
        vwap_js = (
            f"const vwapSeries=chart.addLineSeries({{"
            f"color:'#ffcc00',lineWidth:2,lineStyle:1,"
            f"priceLineVisible:false,lastValueVisible:true,title:'VWAP'}}); "
            f"vwapSeries.setData({json.dumps(vwap_data)});"
        )

    markers_js = (
        f"candleSeries.setMarkers({json.dumps(markers)});"
        if markers else ""
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#1a1a2e;overflow:hidden}}
#chart{{width:100%;height:{height}px}}
.legend{{position:absolute;top:8px;left:12px;display:flex;gap:14px;
         z-index:10;font-size:11px;color:#aaa;pointer-events:none}}
.li{{display:flex;align-items:center;gap:4px}}
.dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
</style>
</head>
<body>
<div style="position:relative">
  <div class="legend">
    <div class="li"><div class="dot" style="background:#00aaff"></div>EMA9</div>
    <div class="li"><div class="dot" style="background:#ff6600"></div>EMA21</div>
    <div class="li"><div class="dot" style="background:#ffcc00"></div>VWAP</div>
  </div>
  <div id="chart"></div>
</div>
<script>
const chart=LightweightCharts.createChart(document.getElementById('chart'),{{
  width:document.getElementById('chart').offsetWidth||900,
  height:{height},
  layout:{{background:{{color:'#1a1a2e'}},textColor:'#d0d0d0'}},
  grid:{{vertLines:{{color:'#252540'}},horzLines:{{color:'#252540'}}}},
  crosshair:{{mode:LightweightCharts.CrosshairMode.Normal}},
  rightPriceScale:{{borderColor:'#3a3a5e',scaleMargins:{{top:0.08,bottom:0.08}}}},
  timeScale:{{borderColor:'#3a3a5e',timeVisible:true,secondsVisible:false}},
}});

const candleSeries=chart.addCandlestickSeries({{
  upColor:'#00ff88',downColor:'#ff4444',
  borderUpColor:'#00ff88',borderDownColor:'#ff4444',
  wickUpColor:'#00ff88',wickDownColor:'#ff4444',
}});
candleSeries.setData({json.dumps(candle_data)});

const ema9Series=chart.addLineSeries({{
  color:'#00aaff',lineWidth:1,priceLineVisible:false,lastValueVisible:false}});
ema9Series.setData({json.dumps(ema9_data)});

const ema21Series=chart.addLineSeries({{
  color:'#ff6600',lineWidth:1,priceLineVisible:false,lastValueVisible:false}});
ema21Series.setData({json.dumps(ema21_data)});

{vwap_js}
{price_line_js}
{markers_js}

chart.timeScale().fitContent();
window.addEventListener('resize',()=>{{
  chart.applyOptions({{width:document.getElementById('chart').offsetWidth}});
}});
</script>
</body>
</html>"""
    return html
