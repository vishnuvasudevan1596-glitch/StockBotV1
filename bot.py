#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║          🇮🇳  NSE INTRADAY STOCK SCANNER  —  v2.0              ║
║   Universe  : Nifty 50 + Nifty 100 + F&O Stocks                 ║
║   Timeframe : 15-minute candles                                  ║
║   Signals   : Breakout · Trend (EMA/MACD) · RSI Reversal        ║
║               VWAP Bounce · Volume Spike                         ║
║   Alerts    : Telegram with full trade plan + indicators         ║
║   Hosting   : Optimised for Railway.app                          ║
╚══════════════════════════════════════════════════════════════════╝

RAILWAY START COMMAND:
    python nse_intraday_scanner.py --loop

LOCAL USAGE:
    python nse_intraday_scanner.py            <- single scan
    python nse_intraday_scanner.py --loop     <- auto-scan every N min
    python nse_intraday_scanner.py --force    <- scan outside market hours (testing)

RAILWAY ENV VARIABLES (set in Variables tab):
    TELEGRAM_BOT_TOKEN   -- required
    TELEGRAM_CHAT_ID     -- required
    MIN_PRICE            -- default 50
    MAX_PRICE            -- default 5000
    MIN_AVG_VOLUME       -- default 300000
    MIN_CONFIDENCE       -- default 55
    SCAN_INTERVAL_MIN    -- default 15
    SL_ATR_MULT          -- default 1.5
    TP1_RR / TP2_RR / TP3_RR -- default 1.5 / 2.5 / 3.5
"""

import os
import sys
import time
import logging
import argparse
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pytz


# ════════════════════════════════════════════════════════════════════
#  ENVIRONMENT-DRIVEN CONFIG
# ════════════════════════════════════════════════════════════════════

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID",   "YOUR_CHAT_ID_HERE")

SCAN_INTERVAL_MIN  = int(os.environ.get("SCAN_INTERVAL_MIN", "15"))
TIMEFRAME          = os.environ.get("TIMEFRAME",             "15m")
DATA_PERIOD_DAYS   = int(os.environ.get("DATA_PERIOD_DAYS",  "5"))

MIN_PRICE          = float(os.environ.get("MIN_PRICE",       "50"))
MAX_PRICE          = float(os.environ.get("MAX_PRICE",       "5000"))
MIN_AVG_VOLUME     = float(os.environ.get("MIN_AVG_VOLUME",  "300000"))
MIN_CONFIDENCE     = int(os.environ.get("MIN_CONFIDENCE",    "55"))
MIN_RR_RATIO       = float(os.environ.get("MIN_RR_RATIO",    "1.5"))

SL_ATR_MULT        = float(os.environ.get("SL_ATR_MULT",    "1.5"))
TP1_RR             = float(os.environ.get("TP1_RR",          "1.5"))
TP2_RR             = float(os.environ.get("TP2_RR",          "2.5"))
TP3_RR             = float(os.environ.get("TP3_RR",          "3.5"))

IST = pytz.timezone("Asia/Kolkata")


# ════════════════════════════════════════════════════════════════════
#  STOCK UNIVERSE
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

ALL_STOCKS  = list(dict.fromkeys(NIFTY_50 + NIFTY_100_EXTRA + FNO_EXTRA))
NSE_SYMBOLS = [s + ".NS" for s in ALL_STOCKS]


# ════════════════════════════════════════════════════════════════════
#  LOGGING  — stdout so Railway log viewer captures everything
# ════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

log = logging.getLogger("nse_scanner")


# ════════════════════════════════════════════════════════════════════
#  TELEGRAM  — with retry + rate-limit handling
# ════════════════════════════════════════════════════════════════════

def telegram_ok() -> bool:
    return TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE"


def send_telegram(text: str, retries: int = 3) -> bool:
    if not telegram_ok():
        print("\n" + "─" * 60)
        print("📨  TELEGRAM (not configured — console output):")
        print(text)
        print("─" * 60 + "\n", flush=True)
        return True

    url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":                  TELEGRAM_CHAT_ID,
        "text":                     text,
        "parse_mode":               "HTML",
        "disable_web_page_preview": True,
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                return True
            if r.status_code == 429:
                wait = r.json().get("parameters", {}).get("retry_after", 5)
                log.warning(f"Telegram rate-limited — waiting {wait}s …")
                time.sleep(wait)
                continue
            log.warning(f"Telegram {r.status_code}: {r.text[:120]}")
        except requests.RequestException as e:
            log.warning(f"Telegram attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                time.sleep(3)
    return False


# ════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS  (pure pandas/numpy — no extra deps)
# ════════════════════════════════════════════════════════════════════

def ind_rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ind_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ef  = series.ewm(span=fast,   adjust=False).mean()
    es  = series.ewm(span=slow,   adjust=False).mean()
    lin = ef - es
    sig = lin.ewm(span=signal, adjust=False).mean()
    return lin, sig, lin - sig


def ind_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def ind_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()


def ind_atr(df: pd.DataFrame, period=14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()


def ind_bollinger(series: pd.Series, period=20, std=2.0):
    mid = series.rolling(period).mean()
    sd  = series.rolling(period).std()
    return mid + std * sd, mid, mid - std * sd


def ind_adx(df: pd.DataFrame, period=14) -> pd.Series:
    hi, lo  = df["High"], df["Low"]
    pdm     = hi.diff().clip(lower=0)
    mdm     = (-lo.diff()).clip(lower=0)
    pdm[pdm < mdm] = 0
    mdm[mdm < pdm] = 0
    atr_1   = ind_atr(df, 1).rolling(period).mean().replace(0, np.nan)
    pdi     = 100 * pdm.rolling(period).mean() / atr_1
    mdi     = 100 * mdm.rolling(period).mean() / atr_1
    dx      = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.rolling(period).mean()


def ind_supertrend(df: pd.DataFrame, period=10, mult=3.0) -> pd.Series:
    at     = ind_atr(df, period)
    hl2    = (df["High"] + df["Low"]) / 2
    upper  = hl2 + mult * at
    lower  = hl2 - mult * at
    close  = df["Close"]
    d      = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if   close.iloc[i] > upper.iloc[i - 1]: d.iloc[i] =  1
        elif close.iloc[i] < lower.iloc[i - 1]: d.iloc[i] = -1
        else:                                    d.iloc[i] =  d.iloc[i - 1]
    return d


# ════════════════════════════════════════════════════════════════════
#  SIGNAL DETECTION ENGINE
# ════════════════════════════════════════════════════════════════════

def _fv(series: pd.Series, idx: int = -1) -> float:
    return float(series.iloc[idx])


def detect_signals(df: pd.DataFrame, symbol: str) -> dict | None:
    if df is None or len(df) < 55:
        return None

    # Flatten yfinance multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    close  = df["Close"]
    volume = df["Volume"]

    # ── Indicators ────────────────────────────────────────────────────
    rsi_s                 = ind_rsi(close)
    macd_l, sig_l, hist_s = ind_macd(close)
    ema9_s                = ind_ema(close, 9)
    ema21_s               = ind_ema(close, 21)
    ema50_s               = ind_ema(close, 50)
    vwap_s                = ind_vwap(df)
    atr_s                 = ind_atr(df)
    bb_up, _, bb_lo       = ind_bollinger(close)
    adx_s                 = ind_adx(df)
    st_d                  = ind_supertrend(df)

    # ── Snapshot ──────────────────────────────────────────────────────
    ltp       = round(_fv(close),    2)
    rsi_v     = round(_fv(rsi_s),    2)
    rsi_p     =       _fv(rsi_s, -2)
    hist_v    = round(_fv(hist_s),   4)
    hist_p    =       _fv(hist_s,-2)
    macd_v    = round(_fv(macd_l),   4)
    sig_v     = round(_fv(sig_l),    4)
    ema9_v    =       _fv(ema9_s)
    ema21_v   =       _fv(ema21_s)
    ema50_v   =       _fv(ema50_s)
    vwap_v    = round(_fv(vwap_s),   2)
    vwap_p    =       _fv(vwap_s,-2)
    atr_v     =       _fv(atr_s)
    adx_v     = round(_fv(adx_s),    2)
    st_now    =   int(_fv(st_d))
    bb_up_v   =       _fv(bb_up)
    bb_lo_v   =       _fv(bb_lo)

    avg_vol   = float(volume.rolling(20).mean().iloc[-1])
    cur_vol   = float(volume.iloc[-1])
    vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 0
    vol_pct   = round((vol_ratio - 1) * 100, 1)

    # ── Pre-filters ───────────────────────────────────────────────────
    if not (MIN_PRICE <= ltp <= MAX_PRICE):   return None
    if avg_vol < MIN_AVG_VOLUME:              return None
    if np.isnan(atr_v) or atr_v <= 0:        return None
    if np.isnan(adx_v):                       return None

    # ── Signal scoring ────────────────────────────────────────────────
    found = []
    score = 0

    # 1. BREAKOUT
    resistance = float(close.rolling(20).max().iloc[-2])
    if ltp > resistance and vol_ratio >= 1.5:
        found.append("🚀 Price Breakout above 20-period resistance")
        score += 25

    if _fv(close, -2) < _fv(bb_up, -2) and ltp > bb_up_v and vol_ratio >= 1.3:
        found.append("🚀 Bollinger Band Upper Breakout")
        score += 15

    # 2. TREND-FOLLOWING
    ema_bull       = ema9_v > ema21_v > ema50_v
    macd_fresh     = hist_v > 0 and hist_p <= 0
    if ema_bull and macd_fresh:
        found.append("📈 Trend: EMA stack + fresh MACD crossover")
        score += 28
    elif ema_bull and macd_v > sig_v:
        found.append("📈 Trend: EMA bullish stack")
        score += 16

    # 3. RSI REVERSAL
    if 35 <= rsi_v <= 58 and rsi_v > rsi_p and rsi_p < 50:
        found.append("🔄 RSI Reversal — rising from oversold zone")
        score += 20

    # 4. VWAP BOUNCE
    if _fv(close, -2) < vwap_p and ltp > vwap_v and vol_ratio >= 1.0:
        found.append("💹 VWAP Bounce — price crossed above VWAP")
        score += 22

    # 5. VOLUME SPIKE
    if vol_ratio >= 2.0:
        found.append(f"🔊 Volume Spike: +{vol_pct}% above 20D avg")
        score += 15
    elif vol_ratio >= 1.5:
        found.append(f"🔊 Volume Surge: +{vol_pct}% above 20D avg")
        score += 8

    # Bonus modifiers
    if adx_v  > 25:       score += 8
    if st_now == 1:       score += 10
    if ltp    > vwap_v:   score += 5
    if ema9_v > ema21_v:  score += 4

    score = min(score, 100)

    if not found or score < MIN_CONFIDENCE:
        return None

    # ── Trade plan ────────────────────────────────────────────────────
    sl   = round(ltp - SL_ATR_MULT * atr_v, 2)
    risk = ltp - sl
    if risk <= 0:
        return None

    tp1 = round(ltp + TP1_RR * risk, 2)
    tp2 = round(ltp + TP2_RR * risk, 2)
    tp3 = round(ltp + TP3_RR * risk, 2)

    if (tp1 - ltp) / risk < MIN_RR_RATIO:
        return None

    return {
        "symbol":     symbol.replace(".NS", ""),
        "ltp":        ltp,
        "signals":    found,
        "score":      score,
        "rsi":        rsi_v,
        "macd":       macd_v,
        "macd_hist":  hist_v,
        "ema9":       round(ema9_v,  2),
        "ema21":      round(ema21_v, 2),
        "ema50":      round(ema50_v, 2),
        "vwap":       vwap_v,
        "adx":        adx_v,
        "supertrend": "🟢 BUY" if st_now == 1 else "🔴 SELL",
        "above_vwap": ltp > vwap_v,
        "vol_pct":    vol_pct,
        "cur_vol":    int(cur_vol),
        "avg_vol":    int(avg_vol),
        "bb_upper":   round(bb_up_v, 2),
        "bb_lower":   round(bb_lo_v, 2),
        "entry":      ltp,
        "sl":         sl,
        "tp1":        tp1,
        "tp2":        tp2,
        "tp3":        tp3,
        "atr":        round(atr_v,  2),
        "risk":       round(risk,   2),
    }


# ════════════════════════════════════════════════════════════════════
#  TELEGRAM MESSAGE FORMATTERS
# ════════════════════════════════════════════════════════════════════

def pct_diff(new_val, base):
    return round(((new_val - base) / base) * 100, 2)


def format_alert(r: dict) -> str:
    bar      = "━" * 32
    stars    = "⭐" * (r["score"] // 20)
    sigs     = "\n    ".join(r["signals"])
    now_str  = datetime.now(IST).strftime("%d %b %Y  %I:%M %p IST")
    vwap_lbl = "✅ Above VWAP"  if r["above_vwap"] else "❌ Below VWAP"
    adx_lbl  = "✅ Strong"      if r["adx"] > 25   else "〰️ Weak"
    rsi_lbl  = "✅"             if r["rsi"] < 60   else "⚠️ Overbought"
    sl_pct   = abs(pct_diff(r["sl"],  r["ltp"]))
    tp1_pct  = pct_diff(r["tp1"], r["ltp"])
    tp2_pct  = pct_diff(r["tp2"], r["ltp"])
    tp3_pct  = pct_diff(r["tp3"], r["ltp"])

    return f"""
🇮🇳 <b>NSE INTRADAY SIGNAL 🔔</b>
{bar}
📌 <b>Stock  :</b> <code>{r["symbol"]}</code>  |  NSE
💰 <b>LTP    :</b> ₹{r["ltp"]}
📅 <b>Time   :</b> {now_str}
{bar}
<b>📡 SIGNALS TRIGGERED</b>
    {sigs}
{bar}
<b>🎯 TRADE PLAN  (15-min chart)</b>

  ➤ <b>Entry :</b> ₹{r["entry"]}
  🛡 <b>SL    :</b> ₹{r["sl"]}   <i>(-{sl_pct}%  |  ATR ×{SL_ATR_MULT})</i>
  🎯 <b>TP 1  :</b> ₹{r["tp1"]}  <i>(+{tp1_pct}%  |  RR 1:{TP1_RR})</i>
  🎯 <b>TP 2  :</b> ₹{r["tp2"]}  <i>(+{tp2_pct}%  |  RR 1:{TP2_RR})</i>
  🎯 <b>TP 3  :</b> ₹{r["tp3"]}  <i>(+{tp3_pct}%  |  RR 1:{TP3_RR})</i>
  📏 <b>ATR   :</b> ₹{r["atr"]}   |  Risk/share: ₹{r["risk"]}
{bar}
<b>📊 INDICATORS</b>

  RSI (14)       : {r["rsi"]}  {rsi_lbl}
  MACD           : {r["macd"]}  (Hist: {r["macd_hist"]})
  EMA  9/21/50   : {r["ema9"]} / {r["ema21"]} / {r["ema50"]}
  VWAP           : ₹{r["vwap"]}  {vwap_lbl}
  ADX (14)       : {r["adx"]}  {adx_lbl}
  Supertrend     : {r["supertrend"]}
  BB Upper/Lower : {r["bb_upper"]} / {r["bb_lower"]}
  Volume (curr)  : {r["cur_vol"]:,}
  Vol vs 20D avg : +{r["vol_pct"]}%
{bar}
<b>🧠 CONFIDENCE: {r["score"]} / 100</b>  {stars}
{bar}
⚠️ <i>Educational use only. Not SEBI-registered advice.
   Always manage your own risk.</i>""".strip()


def format_summary(results: list, scan_time: str) -> str:
    lines = [
        f"📊 <b>NSE Scan Complete</b> — {scan_time}",
        f"Scanned <b>{len(NSE_SYMBOLS)}</b> stocks  |  Found <b>{len(results)}</b> signal(s)\n",
    ]
    for r in results:
        stars = "⭐" * (r["score"] // 20)
        lines.append(
            f"{stars} <code>{r['symbol']}</code>  ₹{r['ltp']}"
            f"  Score:<b>{r['score']}</b>"
            f"  SL:₹{r['sl']}  TP1:₹{r['tp1']}"
        )
    return "\n".join(lines)


def format_no_signals(scan_time: str) -> str:
    return (
        f"📊 <b>NSE Scan</b> — {scan_time}\n"
        f"No qualifying signals this cycle.\n"
        f"(Threshold: {MIN_CONFIDENCE}/100)"
    )


def format_startup() -> str:
    return (
        f"🟢 <b>NSE Scanner Online</b>\n"
        f"Universe : {len(ALL_STOCKS)} stocks (Nifty50 + Nifty100 + F&O)\n"
        f"Timeframe: {TIMEFRAME}  |  Every {SCAN_INTERVAL_MIN} min\n"
        f"Market   : Mon–Fri  09:15–15:30 IST\n"
        f"Min score: {MIN_CONFIDENCE}/100  |  Min R:R 1:{MIN_RR_RATIO}\n"
        f"Price    : ₹{int(MIN_PRICE)}–₹{int(MAX_PRICE)}\n"
        f"SL: ATR×{SL_ATR_MULT}  TP: {TP1_RR}/{TP2_RR}/{TP3_RR}"
    )


# ════════════════════════════════════════════════════════════════════
#  MAIN SCAN
# ════════════════════════════════════════════════════════════════════

def run_scan() -> list:
    now_ist   = datetime.now(IST)
    scan_time = now_ist.strftime("%d %b %Y  %I:%M %p IST")
    log.info(f"🔍  Scan — {scan_time} — {len(NSE_SYMBOLS)} stocks")

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
            r = detect_signals(df, sym)
            if r:
                results.append(r)
                short = " + ".join(s.split(" ")[1] for s in r["signals"])
                log.info(f"  ✅  {r['symbol']:16s}  ₹{r['ltp']:>8.2f}  Score:{r['score']:>3}  [{short}]")
        except Exception:
            errors += 1
            log.debug(f"  ⚠️  {sym}: {traceback.format_exc(limit=1)}")
        time.sleep(0.25)

    results.sort(key=lambda x: x["score"], reverse=True)
    log.info(f"✅  Scan done — {len(results)} signals | {errors} errors")

    if not results:
        send_telegram(format_no_signals(scan_time))
        return results

    send_telegram(format_summary(results, scan_time))
    time.sleep(1)
    for r in results:
        send_telegram(format_alert(r))
        time.sleep(0.8)

    return results


# ════════════════════════════════════════════════════════════════════
#  MARKET HOURS HELPERS
# ════════════════════════════════════════════════════════════════════

def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    o = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    c = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return o <= now <= c


def secs_to_open() -> int:
    now = datetime.now(IST)
    o   = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now < o:
        return int((o - now).total_seconds())
    return 0


# ════════════════════════════════════════════════════════════════════
#  STARTUP DISPLAY
# ════════════════════════════════════════════════════════════════════

def print_banner():
    print(f"""
╔══════════════════════════════════════════════════════════╗
║    🇮🇳  NSE Intraday Stock Scanner  —  v2.0             ║
║  Universe : {len(ALL_STOCKS)} stocks  |  Timeframe : {TIMEFRAME}              ║
║  Hosting  : Railway.app  |  Data : yfinance (free)     ║
╚══════════════════════════════════════════════════════════╝
""", flush=True)


def log_config():
    log.info("── Config ────────────────────────────────────────────────")
    log.info(f"  Telegram    : {'✅ Configured' if telegram_ok() else '❌ Not configured (console only)'}")
    log.info(f"  Universe    : {len(ALL_STOCKS)} stocks")
    log.info(f"  Timeframe   : {TIMEFRAME}  |  Interval: {SCAN_INTERVAL_MIN} min")
    log.info(f"  Price       : ₹{int(MIN_PRICE)} – ₹{int(MAX_PRICE)}")
    log.info(f"  Min volume  : {int(MIN_AVG_VOLUME):,}")
    log.info(f"  Min score   : {MIN_CONFIDENCE}/100")
    log.info(f"  Min R:R     : 1:{MIN_RR_RATIO}")
    log.info(f"  SL / TPs    : ATR×{SL_ATR_MULT}  |  1:{TP1_RR}  1:{TP2_RR}  1:{TP3_RR}")
    log.info("──────────────────────────────────────────────────────────")


def warn_no_telegram():
    if not telegram_ok():
        log.warning("⚠️  TELEGRAM NOT CONFIGURED — alerts go to console/Railway logs only.")
        log.warning("   Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as Railway env vars.")


# ════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🇮🇳 NSE Intraday Scanner")
    parser.add_argument("--loop",  action="store_true",
                        help="Auto-scan every N min during market hours (use on Railway)")
    parser.add_argument("--force", action="store_true",
                        help="Force scan outside market hours (for testing)")
    args = parser.parse_args()

    print_banner()
    log_config()
    warn_no_telegram()

    # ── Single scan ───────────────────────────────────────────────────
    if not args.loop:
        if not is_market_open() and not args.force:
            now = datetime.now(IST)
            log.info(f"⏰  Market closed ({now.strftime('%H:%M %Z  %A')}).")
            log.info("   Use --force to scan anyway, or --loop for auto-scheduling.")
            sys.exit(0)
        run_scan()
        sys.exit(0)

    # ── Loop mode (Railway) ───────────────────────────────────────────
    log.info(f"🔁  Loop mode — every {SCAN_INTERVAL_MIN} min  |  Mon–Fri 09:15–15:30 IST")
    log.info("    Ctrl+C to stop.\n")

    if telegram_ok():
        send_telegram(format_startup())

    consecutive_errors = 0

    while True:
        try:
            if is_market_open() or args.force:
                run_scan()
                consecutive_errors = 0
                log.info(f"💤  Next scan in {SCAN_INTERVAL_MIN} min …\n")
                time.sleep(SCAN_INTERVAL_MIN * 60)
            else:
                now  = datetime.now(IST)
                secs = secs_to_open()
                if 0 < secs < 8 * 3600:
                    mins = secs // 60
                    log.info(f"⏰  Market opens in {mins} min — sleeping until then …")
                    time.sleep(max(secs - 30, 60))
                else:
                    log.info(f"⏰  Market closed ({now.strftime('%H:%M IST  %A')}) — checking in 5 min …")
                    time.sleep(5 * 60)

        except KeyboardInterrupt:
            log.info("👋  Scanner stopped.")
            if telegram_ok():
                send_telegram("🔴 <b>NSE Scanner stopped.</b>")
            break

        except Exception as e:
            consecutive_errors += 1
            log.error(f"❌  Error #{consecutive_errors}: {e}")
            log.error(traceback.format_exc())
            if consecutive_errors >= 5:
                send_telegram(f"❌ <b>Scanner: {consecutive_errors} errors in a row</b>\n{str(e)[:200]}")
                consecutive_errors = 0
            time.sleep(60)
