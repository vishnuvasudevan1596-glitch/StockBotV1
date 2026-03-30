#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║          🇮🇳  NSE INTRADAY STOCK SCANNER                        ║
║   Universe  : Nifty 50 + Nifty 100 + F&O Stocks                 ║
║   Timeframe : 15-minute candles                                  ║
║   Signals   : Breakout · Trend (EMA/MACD) · RSI Reversal        ║
║               VWAP Bounce · Volume Spike                         ║
║   Alerts    : Telegram with full trade plan + indicators         ║
╚══════════════════════════════════════════════════════════════════╝

SETUP (one-time):
  pip install yfinance pandas numpy requests pytz

HOW TO RUN:
  python nse_intraday_scanner.py            ← single scan
  python nse_intraday_scanner.py --loop     ← auto-scan every 15 min during market hours

TELEGRAM SETUP:
  1. Open Telegram → search @BotFather → /newbot → copy your token
  2. Message your bot once, then visit:
     https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
  3. Find "chat" → "id" in the JSON response
  4. Paste both below in the CONFIG section
"""

import sys
import time
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pytz

# ════════════════════════════════════════════════════════════════════
#  ⚙️  CONFIG  —  Edit this section before running
# ════════════════════════════════════════════════════════════════════

TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"   # e.g. "7123456789:AAFxxxx"
TELEGRAM_CHAT_ID   = "YOUR_CHAT_ID_HERE"     # e.g. "-100123456789" or "123456789"

# ── Scan settings ──────────────────────────────────────────────────
SCAN_INTERVAL_MIN  = 15        # Minutes between auto-scans (--loop mode)
TIMEFRAME          = "15m"     # Candle interval
DATA_PERIOD_DAYS   = 5         # How many days of history to pull

# ── Price & Liquidity filters ───────────────────────────────────────
MIN_PRICE          = 50        # ₹ minimum LTP
MAX_PRICE          = 5000      # ₹ maximum LTP
MIN_AVG_VOLUME     = 300_000   # Minimum 20-day average daily volume (3 lakh)

# ── Signal quality filters ──────────────────────────────────────────
MIN_CONFIDENCE     = 55        # Minimum score (0-100) to send an alert
MIN_RR_RATIO       = 1.5       # Minimum Risk:Reward ratio for TP1

# ── ATR multipliers for SL/TP ───────────────────────────────────────
SL_ATR_MULT        = 1.5       # Stop Loss  = LTP - (SL_ATR_MULT × ATR)
TP1_RR             = 1.5       # TP1 Risk:Reward
TP2_RR             = 2.5       # TP2 Risk:Reward
TP3_RR             = 3.5       # TP3 Risk:Reward


# ════════════════════════════════════════════════════════════════════
#  📋  STOCK UNIVERSE
# ════════════════════════════════════════════════════════════════════

NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
    "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "HCLTECH", "AXISBANK",
    "ASIANPAINT", "BAJFINANCE", "MARUTI", "TITAN", "SUNPHARMA", "ULTRACEMCO",
    "WIPRO", "NTPC", "POWERGRID", "M&M", "TATAMOTORS", "TECHM", "BAJAJFINSV",
    "NESTLEIND", "TATASTEEL", "JSWSTEEL", "ADANIENT", "ADANIPORTS", "COALINDIA",
    "ONGC", "BPCL", "HEROMOTOCO", "DRREDDY", "DIVISLAB", "CIPLA", "APOLLOHOSP",
    "EICHERMOT", "GRASIM", "INDUSINDBK", "SBILIFE", "HDFCLIFE", "BRITANNIA",
    "TATACONSUM", "HINDALCO", "UPL", "LTIM", "BAJAJ-AUTO",
]

NIFTY_100_EXTRA = [
    "ADANIGREEN", "AMBUJACEM", "AUROPHARMA", "BANDHANBNK", "BERGEPAINT",
    "BIOCON", "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL", "DABUR", "DLF",
    "GAIL", "GODREJCP", "GODREJPROP", "HAVELLS", "ICICIPRULI", "IDFCFIRSTB",
    "INDIGO", "INDUSTOWER", "IRCTC", "JINDALSTEL", "JUBLFOOD", "LICI",
    "LUPIN", "MARICO", "MUTHOOTFIN", "NAUKRI", "NMDC", "PAGEIND",
    "PIDILITIND", "PNB", "RECLTD", "SBICARD", "SHREECEM", "SIEMENS",
    "TATAPOWER", "TORNTPHARM", "TRENT", "UNIONBANK", "VEDL", "VOLTAS",
    "ZOMATO", "PAYTM", "NYKAA", "MCDOWELL-N",
]

FNO_EXTRA = [
    "AARTIIND", "ABCAPITAL", "ABFRL", "ACC", "ALKEM", "AMARAJABAT",
    "APLLTD", "ASHOKLEY", "ASTRAL", "ATUL", "AUBANK", "BALRAMCHIN",
    "BATAINDIA", "BEL", "BHEL", "CESC", "COFORGE", "CONCOR", "CUMMINSIND",
    "DEEPAKNTR", "DIXON", "ESCORTS", "EXIDEIND", "FEDERALBNK", "GMRINFRA",
    "GNFC", "GRANULES", "GUJGASLTD", "HAL", "HDFCAMC", "HINDCOPPER",
    "HINDPETRO", "IEX", "IGL", "INDIACEM", "IOC", "IPCALAB", "IRFC",
    "JKCEMENT", "JSWENERGY", "KAJARIACER", "KPITTECH", "LAURUSLABS",
    "LICHSGFIN", "LTTS", "MANAPPURAM", "MCX", "MFSL", "MOTHERSON",
    "MPHASIS", "MRF", "NATIONALUM", "NAVINFLUOR", "OBEROIRLTY", "PEL",
    "PERSISTENT", "PETRONET", "POLYCAB", "RAMCOCEM", "RBLBANK", "SRF",
    "SUZLON", "TATACHEM", "TATACOMM", "TVSMOTOR", "UBL", "ZEEL",
]

# Deduplicated combined universe
ALL_STOCKS   = list(dict.fromkeys(NIFTY_50 + NIFTY_100_EXTRA + FNO_EXTRA))
NSE_SYMBOLS  = [s + ".NS" for s in ALL_STOCKS]
IST          = pytz.timezone("Asia/Kolkata")


# ════════════════════════════════════════════════════════════════════
#  📝  LOGGING
# ════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scanner")


# ════════════════════════════════════════════════════════════════════
#  📨  TELEGRAM HELPERS
# ════════════════════════════════════════════════════════════════════

def send_telegram(text: str) -> bool:
    """Send a message via Telegram Bot API."""
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("\n" + "─" * 60)
        print("📨  TELEGRAM ALERT (bot not configured — printing here):")
        print(text)
        print("─" * 60)
        return True

    url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "HTML",
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            return True
        log.warning(f"Telegram API error {r.status_code}: {r.text[:120]}")
    except requests.RequestException as e:
        log.error(f"Telegram send failed: {e}")
    return False


# ════════════════════════════════════════════════════════════════════
#  📐  TECHNICAL INDICATORS  (pure pandas/numpy — no extra libs)
# ════════════════════════════════════════════════════════════════════

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_f   = series.ewm(span=fast,   adjust=False).mean()
    ema_s   = series.ewm(span=slow,   adjust=False).mean()
    line    = ema_f - ema_s
    sig     = line.ewm(span=signal, adjust=False).mean()
    hist    = line - sig
    return line, sig, hist


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    cum = (tp * df["Volume"]).cumsum()
    return cum / df["Volume"].cumsum()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    plus_dm    = hi.diff().clip(lower=0)
    minus_dm   = (-lo.diff()).clip(lower=0)
    plus_dm[plus_dm  < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr_   = atr(df, period=1)
    atr_  = tr_.rolling(period).mean()
    plus_di  = 100 * plus_dm.rolling(period).mean()  / atr_.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(period).mean() / atr_.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0):
    """Returns (direction Series): +1 = bullish, -1 = bearish."""
    atr_   = atr(df, period)
    hl2    = (df["High"] + df["Low"]) / 2
    upper  = hl2 + mult * atr_
    lower  = hl2 - mult * atr_
    close  = df["Close"]
    direction = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if close.iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
    return direction


# ════════════════════════════════════════════════════════════════════
#  🔎  SIGNAL DETECTION ENGINE
# ════════════════════════════════════════════════════════════════════

def detect_signals(df: pd.DataFrame, symbol: str) -> dict | None:
    """
    Analyse a 15-minute OHLCV DataFrame.
    Returns a result dict if signals are found, else None.
    """
    if df is None or len(df) < 55:
        return None

    # ── flatten multi-index columns if yfinance returns them ─────────
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close  = df["Close"]
    volume = df["Volume"]

    # ── Compute all indicators ────────────────────────────────────────
    rsi_s          = rsi(close)
    macd_l, sig_l, hist_s = macd(close)
    ema9_s         = ema(close, 9)
    ema21_s        = ema(close, 21)
    ema50_s        = ema(close, 50)
    vwap_s         = vwap(df)
    atr_s          = atr(df)
    bb_up, _, bb_lo = bollinger(close)
    adx_s          = adx(df)
    st_dir         = supertrend(df)

    # ── Last-bar snapshot ─────────────────────────────────────────────
    ltp       = round(float(close.iloc[-1]),   2)
    rsi_v     = round(float(rsi_s.iloc[-1]),   2)
    rsi_prev  = float(rsi_s.iloc[-2])
    macd_v    = round(float(macd_l.iloc[-1]),  4)
    sig_v     = round(float(sig_l.iloc[-1]),   4)
    hist_v    = round(float(hist_s.iloc[-1]),  4)
    hist_prev = float(hist_s.iloc[-2])
    ema9_v    = float(ema9_s.iloc[-1])
    ema21_v   = float(ema21_s.iloc[-1])
    ema50_v   = float(ema50_s.iloc[-1])
    vwap_v    = round(float(vwap_s.iloc[-1]),  2)
    vwap_prev = float(vwap_s.iloc[-2])
    atr_v     = float(atr_s.iloc[-1])
    adx_v     = round(float(adx_s.iloc[-1]),   2)
    st_now    = int(st_dir.iloc[-1])
    bb_up_v   = float(bb_up.iloc[-1])
    bb_lo_v   = float(bb_lo.iloc[-1])

    avg_vol_20  = float(volume.rolling(20).mean().iloc[-1])
    curr_vol    = float(volume.iloc[-1])
    vol_ratio   = curr_vol / avg_vol_20 if avg_vol_20 > 0 else 0
    vol_spike_p = round((vol_ratio - 1) * 100, 1)

    # ── Pre-filters ───────────────────────────────────────────────────
    if not (MIN_PRICE <= ltp <= MAX_PRICE):
        return None
    if avg_vol_20 < MIN_AVG_VOLUME:
        return None
    if np.isnan(atr_v) or atr_v <= 0:
        return None

    # ── Signal evaluation ─────────────────────────────────────────────
    signals_found = []
    score         = 0

    # 1. BREAKOUT SIGNAL  ──────────────────────────────────────────────
    #    Price crosses above 20-period high (resistance) with volume surge
    resistance = float(close.rolling(20).max().iloc[-2])
    if ltp > resistance and vol_ratio >= 1.5:
        signals_found.append("🚀 Breakout above resistance")
        score += 25
    # Bollinger Band upper breakout
    if close.iloc[-2] < bb_up.iloc[-2] and ltp > bb_up_v and vol_ratio >= 1.3:
        signals_found.append("🚀 Bollinger Upper Breakout")
        score += 15

    # 2. TREND-FOLLOWING  ──────────────────────────────────────────────
    ema_bull = ema9_v > ema21_v > ema50_v
    macd_cross_bull = hist_v > 0 and hist_prev <= 0   # fresh bullish cross
    if ema_bull and macd_cross_bull:
        signals_found.append("📈 Trend: EMA stack + fresh MACD cross")
        score += 28
    elif ema_bull and macd_v > sig_v:
        signals_found.append("📈 Trend: EMA bullish stack")
        score += 16

    # 3. RSI REVERSAL  ─────────────────────────────────────────────────
    #    RSI rising from oversold territory (30-55 band), turning up
    if 35 <= rsi_v <= 58 and rsi_v > rsi_prev and rsi_prev < 50:
        signals_found.append("🔄 RSI Reversal (rising from oversold)")
        score += 20

    # 4. VWAP BOUNCE  ──────────────────────────────────────────────────
    #    Price was below VWAP, crossed above with volume
    price_was_below = float(close.iloc[-2]) < vwap_prev
    price_now_above = ltp > vwap_v
    if price_was_below and price_now_above and vol_ratio >= 1.0:
        signals_found.append("💹 VWAP Bounce (price crossed above VWAP)")
        score += 22

    # 5. VOLUME SPIKE  ─────────────────────────────────────────────────
    if vol_ratio >= 2.0:
        signals_found.append(f"🔊 Volume Spike: {vol_spike_p}% above 20D avg")
        score += 15
    elif vol_ratio >= 1.5:
        signals_found.append(f"🔊 Volume Surge: {vol_spike_p}% above 20D avg")
        score += 8

    # ── Bonus modifiers ───────────────────────────────────────────────
    if adx_v > 25:        score += 8    # Strong trend
    if st_now == 1:       score += 10   # Supertrend bullish
    if ltp > vwap_v:      score += 5    # Price above VWAP
    if ema9_v > ema21_v:  score += 4    # Short-term EMA positive

    score = min(score, 100)

    # ── Quality gate ──────────────────────────────────────────────────
    if not signals_found:
        return None
    if score < MIN_CONFIDENCE:
        return None

    # ── Trade Plan (ATR-based) ────────────────────────────────────────
    sl   = round(ltp - SL_ATR_MULT * atr_v, 2)
    risk = ltp - sl
    if risk <= 0:
        return None

    tp1  = round(ltp + TP1_RR * risk, 2)
    tp2  = round(ltp + TP2_RR * risk, 2)
    tp3  = round(ltp + TP3_RR * risk, 2)

    # Verify R:R meets minimum
    if (tp1 - ltp) / risk < MIN_RR_RATIO:
        return None

    return {
        "symbol":       symbol.replace(".NS", ""),
        "ltp":          ltp,
        "signals":      signals_found,
        "score":        score,
        # Indicators
        "rsi":          rsi_v,
        "macd":         macd_v,
        "macd_hist":    hist_v,
        "ema9":         round(ema9_v,  2),
        "ema21":        round(ema21_v, 2),
        "ema50":        round(ema50_v, 2),
        "vwap":         vwap_v,
        "adx":          adx_v,
        "supertrend":   "🟢 BUY" if st_now == 1 else "🔴 SELL",
        "above_vwap":   ltp > vwap_v,
        "vol_spike_p":  vol_spike_p,
        "avg_vol_20":   int(avg_vol_20),
        "curr_vol":     int(curr_vol),
        "bb_upper":     round(bb_up_v, 2),
        "bb_lower":     round(bb_lo_v, 2),
        # Trade plan
        "entry":        ltp,
        "sl":           sl,
        "tp1":          tp1,
        "tp2":          tp2,
        "tp3":          tp3,
        "atr":          round(atr_v, 2),
        "risk_per_share": round(risk, 2),
    }


# ════════════════════════════════════════════════════════════════════
#  📨  TELEGRAM MESSAGE FORMATTER
# ════════════════════════════════════════════════════════════════════

def format_alert(r: dict) -> str:
    bar     = "━" * 30
    stars   = "⭐" * (r["score"] // 20)
    signals = "\n    ".join(r["signals"])
    vwap_lbl = "✅ Above VWAP" if r["above_vwap"] else "❌ Below VWAP"
    rsi_lbl  = "✅ Good" if r["rsi"] < 60 else "⚠️ Overbought"
    adx_lbl  = "✅ Strong trend" if r["adx"] > 25 else "〰️ Weak trend"
    now_str  = datetime.now(IST).strftime("%d %b %Y  %I:%M %p IST")

    sl_pct   = round(((r["ltp"] - r["sl"])  / r["ltp"]) * 100, 2)
    tp1_pct  = round(((r["tp1"] - r["ltp"]) / r["ltp"]) * 100, 2)
    tp2_pct  = round(((r["tp2"] - r["ltp"]) / r["ltp"]) * 100, 2)
    tp3_pct  = round(((r["tp3"] - r["ltp"]) / r["ltp"]) * 100, 2)

    return f"""
🇮🇳 <b>NSE INTRADAY SIGNAL</b> 🔔
{bar}
📌 <b>Stock :</b> <code>{r["symbol"]}</code>  |  NSE
💰 <b>LTP   :</b> ₹{r["ltp"]}
📅 <b>Time  :</b> {now_str}
{bar}
<b>📡 SIGNALS TRIGGERED:</b>
    {signals}
{bar}
<b>🎯 TRADE PLAN  (15-min chart)</b>

  ➤ <b>Entry :</b>  ₹{r["entry"]}
  🛡 <b>SL    :</b>  ₹{r["sl"]}   <i>(-{sl_pct}%  |  ATR × {SL_ATR_MULT})</i>
  🎯 <b>TP 1  :</b>  ₹{r["tp1"]}  <i>(+{tp1_pct}%  |  RR 1:{TP1_RR})</i>
  🎯 <b>TP 2  :</b>  ₹{r["tp2"]}  <i>(+{tp2_pct}%  |  RR 1:{TP2_RR})</i>
  🎯 <b>TP 3  :</b>  ₹{r["tp3"]}  <i>(+{tp3_pct}%  |  RR 1:{TP3_RR})</i>
  📏 <b>ATR(14):</b> ₹{r["atr"]}   |  Risk/share: ₹{r["risk_per_share"]}
{bar}
<b>📊 INDICATORS SNAPSHOT</b>

  RSI (14)      : {r["rsi"]}   {rsi_lbl}
  MACD          : {r["macd"]}  (Hist: {r["macd_hist"]})
  EMA 9 / 21 / 50: {r["ema9"]} / {r["ema21"]} / {r["ema50"]}
  VWAP          : ₹{r["vwap"]}   {vwap_lbl}
  ADX (14)      : {r["adx"]}   {adx_lbl}
  Supertrend    : {r["supertrend"]}
  BB Upper/Lower: {r["bb_upper"]} / {r["bb_lower"]}
  Volume (curr) : {r["curr_vol"]:,}
  Vol vs 20D avg: +{r["vol_spike_p"]}%
{bar}
<b>🧠 CONFIDENCE SCORE: {r["score"]} / 100</b>
{stars}
{bar}
⚠️ <i>Educational use only. Not SEBI-registered advice.
   Always apply your own risk management.</i>
""".strip()


def format_summary(results: list[dict], scan_time: str) -> str:
    lines = [
        f"📊 <b>NSE Scan Complete</b> — {scan_time}",
        f"Found <b>{len(results)}</b> signal(s)\n",
    ]
    for r in results:
        stars = "⭐" * (r["score"] // 20)
        lines.append(
            f"{stars} <code>{r['symbol']}</code>  ₹{r['ltp']}"
            f"  |  Score: <b>{r['score']}/100</b>"
            f"  |  SL ₹{r['sl']}  TP1 ₹{r['tp1']}"
        )
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
#  🔄  MAIN SCAN LOGIC
# ════════════════════════════════════════════════════════════════════

def run_scan() -> list[dict]:
    now_ist   = datetime.now(IST)
    scan_time = now_ist.strftime("%d %b %Y  %I:%M %p IST")
    log.info(f"🔍  Scan started at {scan_time}  |  Universe: {len(NSE_SYMBOLS)} stocks")

    results, errors = [], 0

    for sym in NSE_SYMBOLS:
        try:
            df = yf.download(
                sym,
                period=f"{DATA_PERIOD_DAYS}d",
                interval=TIMEFRAME,
                progress=False,
                auto_adjust=True,
            )
            result = detect_signals(df, sym)
            if result:
                results.append(result)
                sigs_short = " + ".join([s.split(" ")[1] for s in result["signals"]])
                log.info(f"  ✅  {result['symbol']:15s}  LTP ₹{result['ltp']:>8.2f}  Score {result['score']:>3}  [{sigs_short}]")
        except Exception as e:
            errors += 1
            log.debug(f"  ⚠️  {sym}: {e}")
        time.sleep(0.25)     # polite rate limiting for yfinance

    results.sort(key=lambda x: x["score"], reverse=True)
    log.info(f"✅  Scan done — {len(results)} signals found  ({errors} fetch errors)")

    # ── Send Telegram alerts ──────────────────────────────────────────
    if not results:
        send_telegram(f"📊 <b>NSE Scan</b> — {scan_time}\nNo qualifying signals found this cycle.")
        return results

    # Summary message
    send_telegram(format_summary(results, scan_time))
    time.sleep(1)

    # Individual detailed alerts
    for r in results:
        send_telegram(format_alert(r))
        time.sleep(0.6)

    return results


# ════════════════════════════════════════════════════════════════════
#  ▶️  ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║       🇮🇳  NSE Intraday Stock Scanner  —  v1.0              ║
║   Universe : Nifty 50 + Nifty 100 + F&O  ({} stocks)     ║
║   Timeframe: 15m   |   Data source: yfinance (free)         ║
╚══════════════════════════════════════════════════════════════╝
""".format(len(ALL_STOCKS)))


def check_telegram_config():
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("⚠️  Telegram NOT configured — alerts will print to console.\n")
        print("   To configure Telegram:")
        print("   1. Open Telegram → @BotFather → /newbot → copy token")
        print("   2. Message your bot once")
        print("   3. Visit: https://api.telegram.org/bot<TOKEN>/getUpdates")
        print("   4. Copy chat.id from the response JSON")
        print("   5. Paste both into TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID above\n")


def is_market_open() -> bool:
    now = datetime.now(IST)
    # Skip weekends
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE Intraday Scanner")
    parser.add_argument(
        "--loop", action="store_true",
        help="Auto-scan every 15 min during market hours (9:15–3:30 IST)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force a scan even outside market hours (for testing)"
    )
    args = parser.parse_args()

    print_banner()
    check_telegram_config()

    if not args.loop:
        # ── Single scan mode ──────────────────────────────────────────
        if not is_market_open() and not args.force:
            now = datetime.now(IST)
            print(f"⏰  Market is currently closed (IST: {now.strftime('%H:%M  %a')}).")
            print("   Use --force to scan anyway, or --loop to wait for market open.\n")
            sys.exit(0)
        run_scan()

    else:
        # ── Loop mode ─────────────────────────────────────────────────
        print(f"🔁  Loop mode active. Scanning every {SCAN_INTERVAL_MIN} min during market hours.\n"
              "    Press Ctrl+C to stop.\n")
        while True:
            try:
                if is_market_open() or args.force:
                    run_scan()
                    log.info(f"💤  Next scan in {SCAN_INTERVAL_MIN} min …")
                    time.sleep(SCAN_INTERVAL_MIN * 60)
                else:
                    now = datetime.now(IST)
                    log.info(f"⏰  Market closed ({now.strftime('%H:%M IST')}) — checking again in 5 min …")
                    time.sleep(5 * 60)
            except KeyboardInterrupt:
                log.info("👋  Scanner stopped.")
                break
            except Exception as e:
                log.error(f"Unexpected error: {e} — retrying in 60s …")
                time.sleep(60)
