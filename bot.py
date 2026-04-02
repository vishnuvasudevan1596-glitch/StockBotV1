import yfinance as yf
import pandas as pd
import pandas_ta as ta
import pytz
import requests
import time
from datetime import datetime
import traceback
import os

# ==================== CONFIG ====================
CONFIG = {
    "SCAN_INTERVAL_SECONDS": 90,
    "CANDLE_TIMEFRAME": "5m",
    "LIQUID_SYMBOLS": [
        "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS","SBIN.NS","BHARTIARTL.NS","HINDUNILVR.NS","ITC.NS","LT.NS",
        "AXISBANK.NS","KOTAKBANK.NS","MARUTI.NS","SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","HCLTECH.NS","ASIANPAINTS.NS","BAJFINANCE.NS","ADANIENT.NS",
        "BAJAJFINSV.NS","DMART.NS","INDUSINDBK.NS","TECHM.NS","WIPRO.NS","NESTLEIND.NS","POWERGRID.NS","NTPC.NS","TATASTEEL.NS","JSWSTEEL.NS",
        "GRASIM.NS","COALINDIA.NS","HEROMOTOCO.NS","HDFCLIFE.NS","CIPLA.NS","BRITANNIA.NS","ADANIPORTS.NS","APOLLOHOSP.NS","DRREDDY.NS","EICHERMOT.NS",
        "TATAMOTORS.NS","HINDALCO.NS","SBILIFE.NS","DIVISLAB.NS","JSWENERGY.NS","TRENT.NS","VEDL.NS","M&M.NS","ONGC.NS","BPCL.NS","IOC.NS","GAIL.NS",
        "IRCTC.NS","PIDILITIND.NS","HAVELLS.NS","DABUR.NS","GODREJCP.NS","COLPAL.NS","MARICO.NS","BERGEPAINT.NS"
    ],
    "SUPER_TREND_PERIOD": 10,
    "SUPER_TREND_MULTIPLIER": 3.0,
    "EMA_FAST": 9,
    "EMA_SLOW": 21,
    "RSI_PERIOD": 14,
    "RSI_LONG_THRESHOLD": 55,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "VOLUME_MULTIPLIER": 1.5,
    "ATR_PERIOD": 14,
    "ATR_SL_MULTIPLIER": 2.0,
    "MIN_RR": 2.0,
    "CONFLUENCE_THRESHOLD": 5,
    "SIMULATION_CAPITAL": 100000.0,
    "RISK_PER_TRADE": 0.01,
    "MAX_CALLS_PER_DAY": 8,
    "TELEGRAM_TOKEN": None,
    "TELEGRAM_CHAT_ID": None,
    "HEALTH_PING_MINUTES": 30,
    "SEND_MARKET_OPEN": True,
    "COOLDOWN_MINUTES": 30,
}

IST = pytz.timezone('Asia/Kolkata')

# ========== GLOBALS ==========
capital = CONFIG["SIMULATION_CAPITAL"]
daily_calls = 0
last_health_ping = None
last_daily_reset = None
data_cache = {}
open_positions = []          # each: dict with symbol, entry, sl, target, qty, entry_time
last_alert_time = {}         # symbol -> datetime
market_open_sent = False

def send_telegram(message):
    print(message)
    if not CONFIG["TELEGRAM_TOKEN"] or not CONFIG["TELEGRAM_CHAT_ID"]:
        return
    url = f"https://api.telegram.org/bot{CONFIG['TELEGRAM_TOKEN']}/sendMessage"
    payload = {"chat_id": CONFIG["TELEGRAM_CHAT_ID"], "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except:
        pass

def is_market_open():
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_time <= now <= close_time

# ========== BATCH DATA FETCH ==========
def fetch_all_data():
    """Download latest 2 days of 5-min data for all symbols in one batch."""
    global data_cache
    symbols_str = " ".join(CONFIG["LIQUID_SYMBOLS"])
    try:
        new_data = yf.download(symbols_str, period="2d", interval=CONFIG["CANDLE_TIMEFRAME"],
                               group_by='ticker', progress=False, auto_adjust=True)
        if new_data.empty:
            return
        for symbol in CONFIG["LIQUID_SYMBOLS"]:
            if symbol in new_data.columns.levels[0]:
                sym_data = new_data[symbol].dropna()
                if sym_data.empty:
                    continue
                if symbol in data_cache:
                    df = pd.concat([data_cache[symbol], sym_data]).drop_duplicates()
                else:
                    df = sym_data
                df = df.tail(100)
                data_cache[symbol] = df
    except Exception as e:
        send_telegram(f"⚠️ Batch download error: {str(e)[:100]}")

def get_latest_data(symbol):
    return data_cache.get(symbol, None)

# ========== INDICATORS ==========
def calculate_indicators(df):
    df = df.copy()
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'], anchor="D")
    
    supertrend = ta.supertrend(df['High'], df['Low'], df['Close'],
                               length=CONFIG["SUPER_TREND_PERIOD"],
                               multiplier=CONFIG["SUPER_TREND_MULTIPLIER"])
    st_dir_col = f"SUPERTd_{CONFIG['SUPER_TREND_PERIOD']}_{CONFIG['SUPER_TREND_MULTIPLIER']}"
    df['SUPERT_DIR'] = supertrend.get(st_dir_col, 0)
    
    df['EMA9'] = ta.ema(df['Close'], length=CONFIG["EMA_FAST"])
    df['EMA21'] = ta.ema(df['Close'], length=CONFIG["EMA_SLOW"])
    df['RSI'] = ta.rsi(df['Close'], length=CONFIG["RSI_PERIOD"])
    macd = ta.macd(df['Close'], fast=CONFIG["MACD_FAST"], slow=CONFIG["MACD_SLOW"], signal=CONFIG["MACD_SIGNAL"])
    df = pd.concat([df, macd], axis=1)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=CONFIG["ATR_PERIOD"])
    df['VOL_AVG'] = df['Volume'].rolling(20).mean().fillna(0)
    return df

# ========== SIGNAL GENERATION ==========
def generate_signal(symbol, df):
    global daily_calls
    if daily_calls >= CONFIG["MAX_CALLS_PER_DAY"]:
        return None
    now = datetime.now(IST)
    if symbol in last_alert_time:
        elapsed = (now - last_alert_time[symbol]).total_seconds() / 60
        if elapsed < CONFIG["COOLDOWN_MINUTES"]:
            return None
    
    # Relaxed NaN check: only essential columns for scoring
    required_cols = ['Close', 'SUPERT_DIR', 'EMA9', 'EMA21', 'RSI', 'ATR', 'VOL_AVG']
    if df[required_cols].iloc[-1].isna().any() or df[required_cols].iloc[-2].isna().any():
        return None
    
    row = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    if row['SUPERT_DIR'] == 1: score += 1
    if row['EMA9'] > row['EMA21']: score += 1
    if row['Close'] > row.get('VWAP', row['Close']): score += 1
    if row['RSI'] > CONFIG["RSI_LONG_THRESHOLD"]: score += 1
    
    macd_col = f"MACDh_{CONFIG['MACD_FAST']}_{CONFIG['MACD_SLOW']}_{CONFIG['MACD_SIGNAL']}"
    if macd_col in row.index and macd_col in prev.index:
        if row[macd_col] > 0 and prev[macd_col] < 0:
            score += 1
    elif 'MACDh_12_26_9' in row.index:
        if row['MACDh_12_26_9'] > 0 and prev['MACDh_12_26_9'] < 0:
            score += 1
    
    if row['Volume'] > row['VOL_AVG'] * CONFIG["VOLUME_MULTIPLIER"]: score += 1
    if row['Close'] > row['Open']: score += 1
    
    if score < CONFIG["CONFLUENCE_THRESHOLD"]:
        return None
    
    entry = row['Close']
    atr = row['ATR']
    sl = entry - (atr * CONFIG["ATR_SL_MULTIPLIER"])
    target = entry + (atr * CONFIG["ATR_SL_MULTIPLIER"] * CONFIG["MIN_RR"])
    risk = entry - sl
    if risk <= 0:
        return None
    
    risk_amount = capital * CONFIG["RISK_PER_TRADE"]
    qty = int(risk_amount / risk)
    if qty < 1:
        return None
    
    daily_calls += 1
    last_alert_time[symbol] = now
    return {
        "symbol": symbol.replace(".NS", ""),
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "target": round(target, 2),
        "qty": qty,
        "rr": CONFIG["MIN_RR"],
        "entry_time": now
    }

# ========== REAL SIMULATION ==========
def check_open_positions(df_dict):
    """Monitor open positions against current price; close if SL/target hit or market end."""
    global capital, open_positions
    now = datetime.now(IST)
    closed_positions = []
    for pos in open_positions[:]:
        sym = pos['symbol'] + ".NS"
        df = df_dict.get(sym)
        if df is None or df.empty:
            continue
        current_price = df['Close'].iloc[-1]
        # Check SL hit
        if current_price <= pos['sl']:
            pnl = (current_price - pos['entry']) * pos['qty']
            capital += pnl
            closed_positions.append(f"🛑 {pos['symbol']} SL hit | P&L: ₹{pnl:,.0f}")
            open_positions.remove(pos)
        # Check target hit
        elif current_price >= pos['target']:
            pnl = (pos['target'] - pos['entry']) * pos['qty']
            capital += pnl
            closed_positions.append(f"🎯 {pos['symbol']} Target hit | P&L: ₹{pnl:,.0f}")
            open_positions.remove(pos)
        # Market close exit (15:25 to 15:30 IST)
        elif now.hour == 15 and now.minute >= 25:
            pnl = (current_price - pos['entry']) * pos['qty']
            capital += pnl
            closed_positions.append(f"⏰ {pos['symbol']} Market close exit | P&L: ₹{pnl:,.0f}")
            open_positions.remove(pos)
    for msg in closed_positions:
        send_telegram(msg)

def simulate_trade(signal):
    open_positions.append({
        "symbol": signal['symbol'],
        "entry": signal['entry'],
        "sl": signal['sl'],
        "target": signal['target'],
        "qty": signal['qty'],
        "entry_time": signal['entry_time']
    })
    print(f"[SIMULATION] Opened {signal['symbol']} BUY @ {signal['entry']} | SL {signal['sl']} | Target {signal['target']}")

# ========== MAIN LOOP ==========
def main_loop():
    global last_health_ping, daily_calls, last_daily_reset, capital, data_cache, market_open_sent, open_positions
    send_telegram("🚀 <b>Intraday Call Bot Started</b>\nBatch fetch + Real simulation + Cooldown (FINAL)")
    
    while True:
        now = datetime.now(IST)
        
        # Daily reset only on weekdays
        if (last_daily_reset is None or now.date() != last_daily_reset.date()) and now.weekday() < 5:
            daily_calls = 0
            capital = CONFIG["SIMULATION_CAPITAL"]
            data_cache = {}
            open_positions = []
            last_alert_time.clear()
            market_open_sent = False
            last_daily_reset = now
            send_telegram("🌅 <b>New trading day – reset (calls, capital, positions)</b>")
        
        if not is_market_open():
            time.sleep(60)
            continue
        
        # Market open alert (once)
        if CONFIG["SEND_MARKET_OPEN"] and not market_open_sent and now.hour == 9 and now.minute == 15:
            send_telegram("🌅 <b>Market Open – Scanning started</b>")
            market_open_sent = True
        
        # Health ping
        if last_health_ping is None or (now - last_health_ping).total_seconds() > CONFIG["HEALTH_PING_MINUTES"] * 60:
            send_telegram(f"✅ <b>Bot Healthy</b> | Capital: ₹{capital:,.0f} | Calls today: {daily_calls} | Open positions: {len(open_positions)}")
            last_health_ping = now
        
        # 1. Batch fetch all data
        fetch_all_data()
        
        # 2. Check open positions against latest prices
        check_open_positions(data_cache)
        
        # 3. Generate new signals
        for symbol in CONFIG["LIQUID_SYMBOLS"]:
            try:
                df = get_latest_data(symbol)
                if df is None or len(df) < 50:
                    continue
                df = calculate_indicators(df)
                signal = generate_signal(symbol, df)
                if signal:
                    send_telegram(
                        f"🔥 <b>INTRADAY CALL</b>\n"
                        f"📌 {signal['symbol']} - BUY @ {signal['entry']}\n"
                        f"🛑 SL: {signal['sl']} | 🎯 Target: {signal['target']}\n"
                        f"📦 Qty: {signal['qty']} | RR: 1:{signal['rr']}\n"
                        f"🕒 {now.strftime('%H:%M:%S')} IST"
                    )
                    simulate_trade(signal)
            except Exception as e:
                error_msg = f"❌ Error on {symbol}: {str(e)[:100]}"
                send_telegram(error_msg)
        
        time.sleep(CONFIG["SCAN_INTERVAL_SECONDS"])

if __name__ == "__main__":
    CONFIG["TELEGRAM_TOKEN"] = os.getenv("TELEGRAM_TOKEN")
    CONFIG["TELEGRAM_CHAT_ID"] = os.getenv("TELEGRAM_CHAT_ID")
    try:
        main_loop()
    except Exception as e:
        error = f"💥 Bot Crashed:\n{traceback.format_exc()}"
        send_telegram(error)
        print(error)
        time.sleep(10)
