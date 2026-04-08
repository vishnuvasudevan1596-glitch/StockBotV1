import yfinance as yf
import pandas as pd
import pandas_ta as ta
import pytz
import requests
import time
from datetime import datetime, timedelta
import traceback
import os
import sys

# ==================== NSEPY IMPORTS WITH TRY-EXCEPT ====================
try:
    from nsepy import get_history as nsepy_get_history
    NSEPY_AVAILABLE = True
    print("✅ nsepy loaded successfully", flush=True)
except ImportError:
    NSEPY_AVAILABLE = False
    print("⚠️ nsepy not installed. Install with: pip install nsepy", flush=True)

print("DEBUG: bot.py started (imports done)", flush=True)

# ==================== CONFIG ====================
CONFIG = {
    "SCAN_INTERVAL_SECONDS": 90,
    "CANDLE_TIMEFRAME": "5m",
    "NIFTY500_CSV_URL": "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "MAX_SYMBOLS": 100,
    "MIN_AVG_DAILY_VOL": 800000,
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
    "ATR_SL_MULTIPLIER": 2.5,
    "MIN_RR": 1.5,
    "ADX_THRESHOLD": 25,
    "TRADING_START_HOUR": 9,
    "TRADING_START_MIN": 30,                # Changed from 45 to 30 (9:30 AM)
    "TRADING_END_HOUR": 15,
    "TRADING_END_MIN": 30,
    "SIMULATION_CAPITAL": 100000.0,
    "RISK_PER_TRADE": 0.01,
    "MAX_CALLS_PER_DAY": 5000,
    "COOLDOWN_MINUTES": 45,
    "SLIPPAGE_PCT": 0.15,
    "ROUNDTRIP_COST_PCT": 0.06,
    "INDIA_VIX_THRESHOLD": 22,
    "MIN_VIX": 14,
    "MAX_QTY_PCT_OF_ADV": 0.005,
    "SEND_MARKET_OPEN": True,
    "HEALTH_PING_MINUTES": 30,
    "HISTORICAL_TEST_DATE": "",   # leave empty for live
    "SIGNAL_SCORE_THRESHOLD": 80,
    "SIGNAL_WINDOW": {
        "ENABLED": True,
        "START_HOUR": 9,                    # Changed from 11
        "START_MIN": 30,                    # 9:30 AM
        "END_HOUR": 15,                     # Changed from 14
        "END_MIN": 30,                      # 3:30 PM
    },
    "DAILY_MARKET_FILTER": {
        "ENABLED": True,
        "NIFTY_SYMBOL": "^NSEI",
        "MIN_ADX": 25,
        "ADX_PERIOD": 14,
    },
    "TRAILING_STOP": {
        "ENABLED": True,
        "PARTIAL_EXIT_PCT": 0.5,
        "BREAKEVEN_AFTER_PARTIAL": True,
        "TRAILING_ACTIVATION_PCT": 0.75,
        "TRAILING_DISTANCE_ATR": 1.0,
    },
    "REGIME_RISK_MAP": {
        "TRENDING_UP": 1.0,
        "TRENDING_DOWN": 0.8,
        "RANGING": 0.5,
        "HIGH_VOL": 0.6
    }
}

# ==================== VIX FETCH ====================
def get_india_vix(use_cache=True):
    cache_file = "last_vix.txt"
    if use_cache:
        try:
            with open(cache_file, 'r') as f:
                return float(f.read().strip())
        except:
            pass
    try:
        from nselib import capital_market
        vix_data = capital_market.india_vix_data()
        if vix_data is not None and not vix_data.empty:
            val = float(vix_data['VIX'].iloc[-1])
            with open(cache_file, 'w') as f:
                f.write(str(val))
            return val
    except:
        pass
    try:
        vix = yf.download("^INDIAVIX", period="1d", interval="5m", progress=False)
        if not vix.empty:
            val = float(vix['Close'].iloc[-1])
            with open(cache_file, 'w') as f:
                f.write(str(val))
            return val
    except:
        pass
    return 20.0

# ==================== DATA CLEANING UTILITY ====================
def clean_dataframe(df):
    """Remove duplicate indices, sort, and ensure no NaT or NaN in index."""
    if df is None or df.empty:
        return df
    # Remove duplicate index entries (keep last)
    df = df[~df.index.duplicated(keep='last')]
    # Sort index
    df = df.sort_index()
    return df

# ==================== NIFTY500 SYMBOLS ====================
def get_nifty500_symbols():
    FALLBACK_SYMBOLS = [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
        "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS",
        "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","TITAN.NS",
        "SUNPHARMA.NS","BAJFINANCE.NS","WIPRO.NS","ULTRACEMCO.NS","NESTLEIND.NS",
        "POWERGRID.NS","NTPC.NS","TECHM.NS","HCLTECH.NS","ONGC.NS",
        "BAJAJFINSV.NS","JSWSTEEL.NS","TATAMOTORS.NS","TATASTEEL.NS","ADANIENT.NS",
        "ADANIPORTS.NS","COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS",
        "GRASIM.NS","HEROMOTOCO.NS","HINDALCO.NS","INDUSINDBK.NS","M&M.NS",
        "CIPLA.NS","BPCL.NS","BRITANNIA.NS","APOLLOHOSP.NS","TATACONSUM.NS",
        "SBILIFE.NS","HDFCLIFE.NS","BAJAJ-AUTO.NS","UPL.NS","VEDL.NS"
    ]
    try:
        import io
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com"}
        resp = requests.get(CONFIG["NIFTY500_CSV_URL"], headers=headers, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbols = [f"{sym.strip().upper()}.NS" for sym in df['Symbol'].astype(str) if sym.strip()]
        if len(symbols) > 10:
            print(f"✅ Loaded {len(symbols)} Nifty500 symbols", flush=True)
            return symbols
        raise ValueError
    except Exception as e:
        print(f"⚠️ Using fallback symbols: {e}", flush=True)
        return FALLBACK_SYMBOLS

def filter_liquid_symbols(symbols):
    try:
        data = yf.download(" ".join(symbols[:300]), period="5d", interval="1d", progress=False, group_by='ticker')
        avg_vol = {}
        for sym in symbols[:300]:
            if sym in data.columns.levels[0]:
                avg_vol[sym] = data[sym]['Volume'].mean()
        liquid = [sym for sym, v in avg_vol.items() if v > CONFIG["MIN_AVG_DAILY_VOL"]]
        liquid = liquid[:CONFIG["MAX_SYMBOLS"]]
        print(f"✅ Filtered to {len(liquid)} liquid stocks", flush=True)
        return liquid
    except:
        return symbols[:100]

IST = pytz.timezone('Asia/Kolkata')

# ========== GLOBALS ==========
capital = CONFIG["SIMULATION_CAPITAL"]
daily_calls = 0
last_health_ping = None
last_daily_reset = None
data_cache = {}
open_positions = []
last_alert_time = {}
market_open_sent = False
trades_history = []
LIQUID_SYMBOLS = []
market_ok_for_day = True

def send_telegram(message):
    print(f"Telegram: {message[:50]}...", flush=True)
    if not CONFIG.get("TELEGRAM_TOKEN") or not CONFIG.get("TELEGRAM_CHAT_ID"):
        return
    url = f"https://api.telegram.org/bot{CONFIG['TELEGRAM_TOKEN']}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CONFIG["TELEGRAM_CHAT_ID"], "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}", flush=True)

def is_market_open():
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=9, minute=15, second=0)
    close_time = now.replace(hour=15, minute=30, second=0)
    return open_time <= now <= close_time

def fetch_all_data():
    global data_cache
    if not LIQUID_SYMBOLS:
        return
    symbols_str = " ".join(LIQUID_SYMBOLS)
    try:
        new_data = yf.download(symbols_str, period="2d", interval=CONFIG["CANDLE_TIMEFRAME"],
                               group_by='ticker', progress=False, auto_adjust=True)
        if new_data.empty:
            return
        for symbol in LIQUID_SYMBOLS:
            if symbol in new_data.columns.levels[0]:
                sym_data = new_data[symbol].dropna()
                if sym_data.empty:
                    continue
                sym_data = clean_dataframe(sym_data)
                if symbol in data_cache:
                    existing = clean_dataframe(data_cache[symbol])
                    combined = pd.concat([existing, sym_data]).drop_duplicates()
                    combined = clean_dataframe(combined)
                    data_cache[symbol] = combined.tail(100)
                else:
                    data_cache[symbol] = sym_data.tail(100)
    except Exception as e:
        send_telegram(f"⚠️ Download error: {str(e)[:100]}")

def get_latest_data(symbol):
    df = data_cache.get(symbol, None)
    if df is not None and not df.empty:
        df = clean_dataframe(df)
    return df

def calculate_indicators(df):
    if df is None or df.empty:
        return df
    df = clean_dataframe(df.copy())
    try:
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'], anchor="D")
        supertrend = ta.supertrend(df['High'], df['Low'], df['Close'],
                                   length=CONFIG["SUPER_TREND_PERIOD"],
                                   multiplier=CONFIG["SUPER_TREND_MULTIPLIER"])
        st_col = f"SUPERTd_{CONFIG['SUPER_TREND_PERIOD']}_{CONFIG['SUPER_TREND_MULTIPLIER']}"
        df['SUPERT_DIR'] = supertrend.get(st_col, 0)
        df['EMA9'] = ta.ema(df['Close'], length=CONFIG["EMA_FAST"])
        df['EMA21'] = ta.ema(df['Close'], length=CONFIG["EMA_SLOW"])
        df['RSI'] = ta.rsi(df['Close'], length=CONFIG["RSI_PERIOD"])
        macd = ta.macd(df['Close'], fast=CONFIG["MACD_FAST"], slow=CONFIG["MACD_SLOW"], signal=CONFIG["MACD_SIGNAL"])
        df = pd.concat([df, macd], axis=1)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=CONFIG["ATR_PERIOD"])
        df['VOL_AVG'] = df['Volume'].rolling(20).mean().fillna(0)
        adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df = pd.concat([df, adx_df], axis=1)
        df['SMA20'] = df['Close'].rolling(20).mean()
    except Exception as e:
        print(f"Indicator calculation error: {e}", flush=True)
        return None
    return clean_dataframe(df)

def update_daily_market_filter():
    global market_ok_for_day
    if not CONFIG["DAILY_MARKET_FILTER"]["ENABLED"]:
        market_ok_for_day = True
        return
    try:
        nifty = yf.download(CONFIG["DAILY_MARKET_FILTER"]["NIFTY_SYMBOL"], period="5d", interval="5m", progress=False)
        if nifty.empty or len(nifty) < 50:
            market_ok_for_day = True
            return
        nifty = clean_dataframe(nifty)
        nifty_indicators = calculate_indicators(nifty)
        if nifty_indicators is None:
            market_ok_for_day = True
            return
        adx = nifty_indicators['ADX_14'].iloc[-1]
        if adx < CONFIG["DAILY_MARKET_FILTER"]["MIN_ADX"]:
            market_ok_for_day = False
            send_telegram(f"⚠️ Market filter: Nifty ADX = {adx:.1f} (< {CONFIG['DAILY_MARKET_FILTER']['MIN_ADX']}) -> No trades today")
        else:
            market_ok_for_day = True
            print(f"Market filter: Nifty ADX = {adx:.1f} -> trading allowed", flush=True)
    except Exception as e:
        print(f"Market filter error: {e}", flush=True)
        market_ok_for_day = True

def calculate_signal_score(df):
    if df is None or df.empty:
        return 0
    row = df.iloc[-1]
    score = 0
    if row['SUPERT_DIR'] == 1 and row['EMA9'] > row['EMA21']:
        score += 30
    elif row['EMA9'] > row['EMA21']:
        score += 15
    if row['RSI'] > CONFIG["RSI_LONG_THRESHOLD"]:
        score += 30
    elif row['RSI'] > 50:
        score += 15
    macd_col = f"MACDh_{CONFIG['MACD_FAST']}_{CONFIG['MACD_SLOW']}_{CONFIG['MACD_SIGNAL']}"
    if macd_col in row.index and row[macd_col] > 0:
        score += 10
    if row['Volume'] > row['VOL_AVG'] * CONFIG["VOLUME_MULTIPLIER"]:
        score += 20
    elif row['Volume'] > row['VOL_AVG']:
        score += 10
    atr_ratio = row['ATR'] / df['ATR'].rolling(10).mean().iloc[-1] if len(df) >= 10 else 1
    if atr_ratio > 1.2:
        score += 10
    if row['ADX_14'] > CONFIG["ADX_THRESHOLD"]:
        score += 10
    return min(score, 100)

def is_within_signal_window(now):
    if not CONFIG["SIGNAL_WINDOW"]["ENABLED"]:
        return True
    start = now.replace(hour=CONFIG["SIGNAL_WINDOW"]["START_HOUR"], minute=CONFIG["SIGNAL_WINDOW"]["START_MIN"])
    end = now.replace(hour=CONFIG["SIGNAL_WINDOW"]["END_HOUR"], minute=CONFIG["SIGNAL_WINDOW"]["END_MIN"])
    return start <= now <= end

def generate_signal(symbol, df, now=None, vix=None):
    global daily_calls
    if now is None:
        now = datetime.now(IST)
    if not market_ok_for_day:
        return None
    if not is_within_signal_window(now):
        return None
    if daily_calls >= CONFIG["MAX_CALLS_PER_DAY"]:
        return None
    if now.hour < CONFIG["TRADING_START_HOUR"] or (now.hour == CONFIG["TRADING_START_HOUR"] and now.minute < CONFIG["TRADING_START_MIN"]):
        return None
    if now.hour > CONFIG["TRADING_END_HOUR"] or (now.hour == CONFIG["TRADING_END_HOUR"] and now.minute > CONFIG["TRADING_END_MIN"]):
        return None
    if symbol in last_alert_time:
        if (now - last_alert_time[symbol]).total_seconds() / 60 < CONFIG["COOLDOWN_MINUTES"]:
            return None
    if df is None or df.empty or len(df) < 50:
        return None
    required = ['Close', 'SUPERT_DIR', 'EMA9', 'EMA21', 'RSI', 'ATR', 'VOL_AVG', 'ADX_14', 'SMA20']
    if df[required].iloc[-1].isna().any() or df[required].iloc[-2].isna().any():
        return None
    row = df.iloc[-1]
    if row['SUPERT_DIR'] != 1:
        return None
    if row['EMA9'] <= row['EMA21']:
        return None
    if row['ADX_14'] < CONFIG["ADX_THRESHOLD"]:
        return None
    _vix = vix if vix is not None else get_india_vix()
    if _vix > CONFIG["INDIA_VIX_THRESHOLD"] or _vix < CONFIG["MIN_VIX"]:
        return None
    if row['Close'] < row['SMA20']:
        return None
    score = calculate_signal_score(df)
    if score < CONFIG["SIGNAL_SCORE_THRESHOLD"]:
        return None
    entry = row['Close']
    atr = row['ATR']
    sl = entry - (atr * CONFIG["ATR_SL_MULTIPLIER"])
    risk = entry - sl
    if risk <= 0:
        return None
    regime_mult = 1.0
    if row['ADX_14'] > 30:
        regime_mult = 1.0
    elif row['ADX_14'] < 20:
        regime_mult = 0.5
    else:
        regime_mult = 0.8
    risk_amount = capital * CONFIG["RISK_PER_TRADE"] * regime_mult
    qty = int(risk_amount / risk)
    if qty < 1:
        return None
    target = entry + (atr * CONFIG["ATR_SL_MULTIPLIER"] * CONFIG["MIN_RR"])
    max_qty = int(row['VOL_AVG'] * CONFIG["MAX_QTY_PCT_OF_ADV"])
    if max_qty > 0:
        qty = min(qty, max_qty)
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
        "entry_time": now,
        "atr": atr,
        "vix": _vix,
        "score": score,
        "partial_exit_done": False
    }

def check_open_positions(df_dict, now=None):
    global capital, open_positions, trades_history
    if now is None:
        now = datetime.now(IST)
    closed = []
    for pos in open_positions[:]:
        sym = pos['symbol'] + ".NS"
        df = df_dict.get(sym)
        if df is None or df.empty:
            continue
        df = clean_dataframe(df)
        curr = df['Close'].iloc[-1]
        pnl_gross = 0
        exit_reason = None
        if CONFIG["TRAILING_STOP"]["ENABLED"] and not pos.get("partial_exit_done", False):
            half_target = pos['entry'] + (pos['target'] - pos['entry']) * 0.5
            if curr >= half_target:
                half_qty = pos['qty'] // 2
                if half_qty > 0:
                    pnl_gross_half = (half_target - pos['entry']) * half_qty
                    cost = abs(pnl_gross_half) * (CONFIG["ROUNDTRIP_COST_PCT"] + CONFIG["SLIPPAGE_PCT"]) / 100
                    pnl_net_half = pnl_gross_half - cost
                    capital += pnl_net_half
                    trades_history.append({"win": pnl_net_half > 0, "pnl": pnl_net_half, "symbol": pos['symbol']})
                    pos['qty'] -= half_qty
                    pos['partial_exit_done'] = True
                    send_telegram(f"📌 {pos['symbol']} Partial profit taken | +₹{pnl_net_half:,.0f} | Remaining qty: {pos['qty']}")
                    if CONFIG["TRAILING_STOP"]["BREAKEVEN_AFTER_PARTIAL"]:
                        pos['sl'] = pos['entry']
                    continue
        if CONFIG["TRAILING_STOP"]["ENABLED"] and pos.get("partial_exit_done", False):
            activation_price = pos['entry'] + (pos['target'] - pos['entry']) * CONFIG["TRAILING_STOP"]["TRAILING_ACTIVATION_PCT"]
            if curr >= activation_price:
                atr_val = df['ATR'].iloc[-1]
                new_sl = curr - (atr_val * CONFIG["TRAILING_STOP"]["TRAILING_DISTANCE_ATR"])
                if new_sl > pos['sl']:
                    pos['sl'] = new_sl
        if curr <= pos['sl']:
            pnl_gross = (curr - pos['entry']) * pos['qty']
            exit_reason = f"🛑 {pos['symbol']} SL hit"
        elif curr >= pos['target']:
            pnl_gross = (pos['target'] - pos['entry']) * pos['qty']
            exit_reason = f"🎯 {pos['symbol']} Target hit"
        elif now.hour == 15 and now.minute >= 25:
            pnl_gross = (curr - pos['entry']) * pos['qty']
            exit_reason = f"⏰ {pos['symbol']} Market close exit"
        if pnl_gross != 0:
            cost = abs(pnl_gross) * (CONFIG["ROUNDTRIP_COST_PCT"] + CONFIG["SLIPPAGE_PCT"]) / 100
            pnl_net = pnl_gross - cost
            capital += pnl_net
            trades_history.append({"win": pnl_net > 0, "pnl": pnl_net, "symbol": pos['symbol']})
            closed.append(f"{exit_reason} | P&L net: ₹{pnl_net:,.0f}")
            open_positions.remove(pos)
    for msg in closed:
        send_telegram(msg)

def simulate_trade(signal):
    open_positions.append({
        "symbol": signal['symbol'],
        "entry": signal['entry'],
        "sl": signal['sl'],
        "target": signal['target'],
        "qty": signal['qty'],
        "entry_time": signal['entry_time'],
        "partial_exit_done": False
    })
    print(f"[SIM] Opened {signal['symbol']} @ {signal['entry']}", flush=True)

# ==================== HYBRID DATA FETCHING (for historical) ====================
def fetch_5m_data_nsepy(symbol_clean, start_date, end_date):
    if not NSEPY_AVAILABLE:
        return None
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        data = nsepy_get_history(symbol=symbol_clean,
                                 start=start_dt,
                                 end=end_dt)
        if data is not None and not data.empty:
            data = data.resample('5T').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            print(f"nsepy: Fetched {len(data)} bars for {symbol_clean}", flush=True)
            return data
    except Exception as e:
        print(f"nsepy error for {symbol_clean}: {e}", flush=True)
    return None

def fetch_5m_data_yfinance(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date,
                           interval="5m", progress=False, auto_adjust=True)
        if data is not None and not data.empty:
            print(f"yfinance: Fetched {len(data)} bars for {symbol}", flush=True)
            return data
    except Exception as e:
        print(f"yfinance error for {symbol}: {e}", flush=True)
    return None

def fetch_5m_data_hybrid(symbol, start_date, end_date):
    symbol_clean = symbol.replace(".NS", "")
    data = fetch_5m_data_nsepy(symbol_clean, start_date, end_date)
    if data is not None and not data.empty:
        return data
    data = fetch_5m_data_yfinance(symbol, start_date, end_date)
    if data is not None and not data.empty:
        return data
    return pd.DataFrame()

def fetch_5m_data_batch_hybrid(symbols, start_date, end_date):
    result = {}
    total = len(symbols)
    print(f"Fetching 5m data for {total} symbols using hybrid method...", flush=True)
    for idx, symbol in enumerate(symbols):
        if idx % 10 == 0:
            print(f"  Progress: {idx+1}/{total}", flush=True)
        data = fetch_5m_data_hybrid(symbol, start_date, end_date)
        if not data.empty:
            result[symbol] = data
        time.sleep(0.2)
    print(f"✅ Successfully fetched data for {len(result)}/{total} symbols", flush=True)
    return result

def run_backtest(days=30):
    print("🚀 Running 30-day backtest...", flush=True)
    symbols = filter_liquid_symbols(get_nifty500_symbols())[:50]
    trades = 0
    wins = 0
    for sym in symbols:
        try:
            df = yf.download(sym, period=f"{days}d", interval=CONFIG["CANDLE_TIMEFRAME"], progress=False)
            if len(df) < 100:
                continue
            df = clean_dataframe(df)
            df = calculate_indicators(df)
            if df is None:
                continue
            for i in range(50, len(df)-1):
                bar = df.iloc[i].name
                if not bar.tzinfo:
                    bar = IST.localize(bar)
                sig = generate_signal(sym, df.iloc[:i+1], now=bar)
                if sig:
                    trades += 1
                    exit_price = df.iloc[i+1]['Close']
                    if exit_price > sig['entry']:
                        wins += 1
        except:
            continue
    wr = wins/trades*100 if trades else 0
    print(f"✅ Backtest: {trades} trades, win rate {wr:.1f}%", flush=True)
    send_telegram(f"📈 Backtest: {trades} trades, {wr:.1f}% win-rate")

def run_historical_replay(test_date_str):
    global capital, daily_calls, open_positions, last_alert_time, trades_history, market_ok_for_day
    try:
        test_date = datetime.strptime(test_date_str, "%Y-%m-%d").date()
        print(f"🚀 Replay for {test_date_str}...", flush=True)
        send_telegram(f"📅 Historical Replay Started for {test_date_str}")

        capital = CONFIG["SIMULATION_CAPITAL"]
        daily_calls = 0
        open_positions = []
        last_alert_time.clear()
        trades_history.clear()
        market_ok_for_day = True

        update_daily_market_filter()

        all_symbols = get_nifty500_symbols()
        symbols = all_symbols[:30]
        print(f"Using {len(symbols)} symbols", flush=True)

        start = (pd.to_datetime(test_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        end = (pd.to_datetime(test_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

        print(f"Downloading 5m data: {start} to {end}...", flush=True)
        data_dict = fetch_5m_data_batch_hybrid(symbols, start, end)

        if not data_dict:
            send_telegram("❌ No 5m data available from any source.")
            print("❌ No 5m data downloaded.", flush=True)
            return

        bar_times = pd.date_range(start=f"{test_date} 09:15", end=f"{test_date} 15:30", freq="5min", tz=IST)
        vix = get_india_vix()
        print(f"VIX: {vix:.1f}", flush=True)
        start_time = time.time()

        for idx, bar in enumerate(bar_times):
            now = bar
            if not (9 <= now.hour <= 15):
                continue
            for symbol, df in data_dict.items():
                if df is None or df.empty:
                    continue
                data_cache[symbol] = clean_dataframe(df)
            check_open_positions(data_cache, now=now)
            for symbol in symbols:
                if symbol not in data_cache:
                    continue
                df = data_cache.get(symbol)
                if df is None or len(df) < 40:
                    continue
                df = calculate_indicators(df)
                if df is None:
                    continue
                sig = generate_signal(symbol, df, now=now, vix=vix)
                if sig:
                    send_telegram(
                        f"🔥 <b>INTRADAY CALL [HISTORICAL {test_date_str}]</b>\n"
                        f"📌 {sig['symbol']} - BUY @ {sig['entry']}\n"
                        f"🛑 SL: {sig['sl']} | 🎯 Target: {sig['target']}\n"
                        f"📦 Qty: {sig['qty']} | RR: 1:{sig['rr']}\n"
                        f"📊 ATR: {sig['atr']:.2f} | VIX: {sig['vix']:.1f} | Score: {sig['score']}/100\n"
                        f"🕒 {now.strftime('%H:%M:%S')} IST"
                    )
                    simulate_trade(sig)
            if idx % 10 == 0:
                print(f"Progress: {idx+1}/{len(bar_times)} bars", flush=True)

        check_open_positions(data_cache)
        wins = sum(1 for t in trades_history if t["win"])
        total = len(trades_history)
        net = sum(t["pnl"] for t in trades_history)
        wr = wins/total*100 if total else 0
        send_telegram(f"📊 Summary {test_date_str}\nTrades: {total} | Win%: {wr:.1f}%\nNet P&L: ₹{net:,.0f} | Capital: ₹{capital:,.0f}")
        print(f"Replay done in {time.time()-start_time:.1f}s", flush=True)
    except Exception as e:
        send_telegram(f"❌ Replay failed: {str(e)}")
        print(traceback.format_exc(), flush=True)

def main_loop():
    global last_health_ping, daily_calls, last_daily_reset, capital, data_cache, market_open_sent, open_positions, LIQUID_SYMBOLS, market_ok_for_day
    print("Entering main loop", flush=True)
    send_telegram("🚀 Bot started | Signal window 9:30-15:30 | Daily Nifty ADX filter | Trailing stop")
    while True:
        now = datetime.now(IST)
        if (last_daily_reset is None or now.date() != last_daily_reset.date()) and now.weekday() < 5:
            daily_calls = 0
            capital = CONFIG["SIMULATION_CAPITAL"]
            data_cache = {}
            open_positions = []
            last_alert_time.clear()
            market_open_sent = False
            trades_history.clear()
            last_daily_reset = now
            LIQUID_SYMBOLS = filter_liquid_symbols(get_nifty500_symbols())
            update_daily_market_filter()
            send_telegram("🌅 New trading day – reset")
        if not is_market_open():
            time.sleep(60)
            continue
        if CONFIG["SEND_MARKET_OPEN"] and not market_open_sent and now.hour == 9 and now.minute == 15:
            send_telegram("🌅 Market Open – scanning")
            market_open_sent = True
        if last_health_ping is None or (now - last_health_ping).total_seconds() > CONFIG["HEALTH_PING_MINUTES"]*60:
            send_telegram(f"✅ Healthy | Capital: ₹{capital:,.0f} | Calls: {daily_calls} | Open: {len(open_positions)} | Market OK: {market_ok_for_day}")
            last_health_ping = now
        fetch_all_data()
        check_open_positions(data_cache, now=now)
        if market_ok_for_day:
            for sym in LIQUID_SYMBOLS:
                try:
                    df = get_latest_data(sym)
                    if df is None or len(df) < 50:
                        continue
                    df = calculate_indicators(df)
                    if df is None:
                        continue
                    sig = generate_signal(sym, df, now=now)
                    if sig:
                        send_telegram(
                            f"🔥 <b>INTRADAY CALL</b>\n"
                            f"📌 {sig['symbol']} - BUY @ {sig['entry']}\n"
                            f"🛑 SL: {sig['sl']} | 🎯 Target: {sig['target']}\n"
                            f"📦 Qty: {sig['qty']} | RR: 1:{sig['rr']}\n"
                            f"📊 ATR: {sig['atr']:.2f} | VIX: {sig['vix']:.1f} | Score: {sig['score']}/100\n"
                            f"🕒 {now.strftime('%H:%M:%S')} IST"
                        )
                        simulate_trade(sig)
                except Exception as e:
                    send_telegram(f"❌ Error {sym}: {str(e)[:100]}")
        if now.hour == 15 and now.minute == 35 and trades_history:
            wins = sum(1 for t in trades_history if t["win"])
            total = len(trades_history)
            net = sum(t["pnl"] for t in trades_history)
            wr = wins/total*100 if total else 0
            send_telegram(f"📊 Day Summary\nTrades: {total} | Win%: {wr:.1f}%\nP&L: ₹{net:,.0f} | Capital: ₹{capital:,.0f}")
        time.sleep(CONFIG["SCAN_INTERVAL_SECONDS"])

if __name__ == "__main__":
    CONFIG["TELEGRAM_TOKEN"] = os.getenv("TELEGRAM_TOKEN")
    CONFIG["TELEGRAM_CHAT_ID"] = os.getenv("TELEGRAM_CHAT_ID")
    print(f"TOKEN set: {bool(CONFIG['TELEGRAM_TOKEN'])}", flush=True)
    hist_dates = os.getenv("HISTORICAL_DATES")
    if hist_dates and hist_dates.strip():
        for d in [x.strip() for x in hist_dates.split(",") if x.strip()]:
            run_historical_replay(d)
        sys.exit(0)
    elif CONFIG.get("HISTORICAL_TEST_DATE") and CONFIG["HISTORICAL_TEST_DATE"].strip():
        run_historical_replay(CONFIG["HISTORICAL_TEST_DATE"])
        sys.exit(0)
    else:
        print("Starting live simulation mode...", flush=True)
        try:
            main_loop()
        except Exception as e:
            send_telegram(f"💥 Crash: {traceback.format_exc()}")
            print(traceback.format_exc(), flush=True)
            time.sleep(10)
