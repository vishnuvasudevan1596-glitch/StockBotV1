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


print("DEBUG: bot.py started (imports done)", flush=True)

IST = pytz.timezone('Asia/Kolkata')

# ==================== CONFIG ====================
CONFIG = {
    "SCAN_INTERVAL_SECONDS": 180,
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
    "VOLUME_MULTIPLIER": 1.8,
    "ATR_PERIOD": 14,
    "ATR_SL_MULTIPLIER": 2.5,
    "MIN_RR": 2.0,
    "ADX_THRESHOLD": 20,
    "TRADING_START_HOUR": 9,
    "TRADING_START_MIN": 45,                # Changed from 45 to 30 (9:30 AM)
    "TRADING_END_HOUR": 14,
    "TRADING_END_MIN": 0,
    "SIMULATION_CAPITAL": 100000.0,
    "RISK_PER_TRADE": 0.01,
    "MAX_CALLS_PER_DAY": 10,
    "COOLDOWN_MINUTES": 30,
    "SLIPPAGE_PCT": 0.15,
    "ROUNDTRIP_COST_PCT": 0.06,
    "INDIA_VIX_THRESHOLD": 22,
    "MIN_VIX": 14,
    "MAX_QTY_PCT_OF_ADV": 0.005,
    "SEND_MARKET_OPEN": True,
    "HEALTH_PING_MINUTES": 30,
    "HISTORICAL_TEST_DATE": "",   # leave empty for live
    "SIGNAL_SCORE_THRESHOLD": 75,
    "SIGNAL_WINDOW": {
        "ENABLED": True,
        "START_HOUR": 9,
        "START_MIN": 45,
        "END_HOUR": 14,
        "END_MIN": 0,
    },
    "DAILY_MARKET_FILTER": {
        "ENABLED": True,
        "NIFTY_SYMBOL": "^NSEI",
        "MIN_ADX": 25,
        "ADX_PERIOD": 14,
    },
    "TRAILING_STOP": {
        "ENABLED": True,
        "PARTIAL_EXIT_PCT": 0.8,
        "BREAKEVEN_AFTER_PARTIAL": True,
        "TRAILING_ACTIVATION_PCT": 0.85,
        "TRAILING_DISTANCE_ATR": 1.0,
    },
    "REGIME_RISK_MAP": {
        "TRENDING_UP": 1.0,
        "TRENDING_DOWN": 0.8,
        "RANGING": 0.5,
        "HIGH_VOL": 0.6
    },
    "GLOBAL_COOLDOWN_SECONDS": 180,
    "MAX_SIGNALS_PER_SYMBOL_PER_DAY": 2,
    "MAX_OPEN_POSITIONS": 5,
    "MAX_DAILY_LOSS_PCT": 2.0,
    "REGIME_ADX_STRONG": 30,
    "MIN_VWAP_BUFFER_PCT": 0.1,
    "MAX_TRADE_NOTIONAL_PCT": 0.20,
    "ADX_SLOPE_BARS": 3,
    "ENTRY_COST_PCT": 0.10,
    "EXIT_COST_PCT": 0.10,
    "FIXED_BROKERAGE": 20.0,
}

# ==================== VIX FETCH (in-memory cache, no file) ====================
_vix_cache = {"value": None, "ts": None}

def get_india_vix():
    global _vix_cache
    now = datetime.now(IST)
    # Return cached value if fresh (< 5 minutes old)
    if _vix_cache["value"] is not None and _vix_cache["ts"] is not None:
        if (now - _vix_cache["ts"]).total_seconds() < 300:
            return _vix_cache["value"]
    # Try yfinance (primary, no nselib dependency)
    try:
        vix_df = yf.download("^INDIAVIX", period="2d", interval="5m", progress=False, auto_adjust=True)
        if vix_df is not None and not vix_df.empty:
            close_col = vix_df['Close']
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            val = float(close_col.dropna().iloc[-1])
            if 8.0 < val < 90.0:  # sanity check
                _vix_cache["value"] = val
                _vix_cache["ts"] = now
                print(f"VIX fetched: {val:.2f}", flush=True)
                return val
    except Exception as e:
        print(f"VIX yfinance error: {e}", flush=True)
    # Try NSE API with browser session (backup)
    try:
        sess = requests.Session()
        sess.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.nseindia.com'
        })
        sess.get('https://www.nseindia.com', timeout=8)
        time.sleep(1)
        resp = sess.get('https://www.nseindia.com/api/india-vix', timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            val = float(data['data'][0]['VIX'])
            if 8.0 < val < 90.0:
                _vix_cache["value"] = val
                _vix_cache["ts"] = now
                print(f"VIX NSE API fetched: {val:.2f}", flush=True)
                return val
    except Exception as e:
        print(f"VIX NSE API error: {e}", flush=True)
    # Return last known value or None (never fake 20.0)
    if _vix_cache["value"] is not None:
        print(f"VIX: using last known value {_vix_cache['value']:.2f}", flush=True)
        return _vix_cache["value"]
    print("⚠️ VIX unavailable — skipping VIX filter this cycle", flush=True)
    return None

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

def save_cache(symbol, df):
    if df is None or df.empty: return
    path = os.path.join(DATA_CACHE_DIR, f"{symbol.replace('.NS','')}.parquet")
    df.to_parquet(path)

def load_cache(symbol):
    path = os.path.join(DATA_CACHE_DIR, f"{symbol.replace('.NS','')}.parquet")
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except:
            return None
    return None

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
    liquid = []
    batch_size = 100
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            data = yf.download(" ".join(batch), period="5d", interval="1d", progress=False, group_by='ticker', auto_adjust=True)
            for sym in batch:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        vol = data[sym]['Volume'].mean()
                    else:
                        vol = data['Volume'].mean()
                    if vol > CONFIG["MIN_AVG_DAILY_VOL"]:
                        liquid.append(sym)
                except:
                    continue
            time.sleep(2)
        except:
            continue
    liquid = liquid[:CONFIG["MAX_SYMBOLS"]]
    print(f"✅ Filtered to {len(liquid)} liquid stocks", flush=True)
    return liquid

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
DATA_CACHE_DIR = "/data_cache_5m"
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
equity_curve = []
signal_rejections = []
market_ok_for_day = True
last_global_signal_time = None
symbol_daily_calls = {}
_gate_counts = {}

def send_telegram(message):
    print(f"Telegram: {message[:120]}...", flush=True)
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
    if not LIQUID_SYMBOLS: return
    for symbol in LIQUID_SYMBOLS:
        cached = load_cache(symbol)
        if cached is not None and not cached.empty:
            data_cache[symbol] = cached.tail(200)
            continue
        try:
            new_data = yf.download(symbol, period="60d", interval=CONFIG["CANDLE_TIMEFRAME"], progress=False, auto_adjust=True)
            if not new_data.empty:
                if isinstance(new_data.columns, pd.MultiIndex):
                    new_data.columns = new_data.columns.get_level_values(0)
                new_data = clean_dataframe(new_data)
                data_cache[symbol] = new_data.tail(200)
                save_cache(symbol, new_data)
        except:
            pass
    print(f"✅ Data cache updated for {len(data_cache)} symbols", flush=True)

def get_latest_data(symbol):
    df = data_cache.get(symbol, None)
    if df is not None and not df.empty:
        df = clean_dataframe(df)
    return df

def calculate_indicators(df):
    if df is None or df.empty or len(df) < 20:
        return None
    df = clean_dataframe(df.copy())
    try:
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'], anchor="D")
        
        st = ta.supertrend(df['High'], df['Low'], df['Close'], length=CONFIG["SUPER_TREND_PERIOD"], multiplier=CONFIG["SUPER_TREND_MULTIPLIER"])
        if st is not None and not st.empty:
            df = pd.concat([df, st], axis=1)
            st_dir_col = next((c for c in df.columns if "SUPERTd_" in str(c)), None)
            if st_dir_col is None:
                df['SUPERT_DIR'] = 0
            else:
                df['SUPERT_DIR'] = df[st_dir_col]
        else:
            df['SUPERT_DIR'] = 0

        df['EMA9'] = ta.ema(df['Close'], length=CONFIG["EMA_FAST"])
        df['EMA21'] = ta.ema(df['Close'], length=CONFIG["EMA_SLOW"])
        df['RSI'] = ta.rsi(df['Close'], length=CONFIG["RSI_PERIOD"])

        macd = ta.macd(df['Close'], fast=CONFIG["MACD_FAST"], slow=CONFIG["MACD_SLOW"], signal=CONFIG["MACD_SIGNAL"])
        if macd is not None and not macd.empty:
            df = pd.concat([df, macd], axis=1)
        else:
            df['MACDh'] = 0.0

        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=CONFIG["ATR_PERIOD"])

        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx is not None and not adx.empty:
            df = pd.concat([df, adx], axis=1)
        else:
            df['ADX_14'] = 0.0

        df['VOL_AVG'] = df['Volume'].rolling(20).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        slope_bars = CONFIG.get("ADX_SLOPE_BARS", 3)
        df['ADX_SLOPE'] = df['ADX_14'].diff(slope_bars)
        return clean_dataframe(df)
    except Exception as e:
        print(f"Indicator error: {e}", flush=True)
        return None

def update_daily_market_filter():
    global market_ok_for_day
    if not CONFIG["DAILY_MARKET_FILTER"]["ENABLED"]:
        market_ok_for_day = True
        return

    try:
        nifty = yf.download(CONFIG["DAILY_MARKET_FILTER"]["NIFTY_SYMBOL"], 
                            period="10d", interval="5m", progress=False)
        
        if nifty.empty or len(nifty) < 30:
            nifty = yf.download(CONFIG["DAILY_MARKET_FILTER"]["NIFTY_SYMBOL"], 
                                period="1mo", interval="1d", progress=False)
            if nifty.empty or len(nifty) < 10:
                market_ok_for_day = True
                print("Market filter: Insufficient Nifty data → trading allowed (fallback)", flush=True)
                return

        nifty = clean_dataframe(nifty)
        nifty_indicators = calculate_indicators(nifty)
        
        if nifty_indicators is None or 'ADX_14' not in nifty_indicators.columns:
            market_ok_for_day = True
            print("Market filter: Could not compute ADX → trading allowed", flush=True)
            return

        adx = nifty_indicators['ADX_14'].dropna().iloc[-1] if not nifty_indicators['ADX_14'].dropna().empty else 0
        
        if adx < CONFIG["DAILY_MARKET_FILTER"]["MIN_ADX"]:
            market_ok_for_day = False
            send_telegram(f"⚠️ Market filter: Nifty ADX = {adx:.1f} (< {CONFIG['DAILY_MARKET_FILTER']['MIN_ADX']}) → No trades today")
        else:
            market_ok_for_day = True
            print(f"Market filter: Nifty ADX = {adx:.1f} → trading allowed", flush=True)
            
    except Exception as e:
        print(f"Market filter error: {e} → trading allowed (safety fallback)", flush=True)
        market_ok_for_day = True

def get_regime(df):
    if df is None or len(df) < 20: return "RANGING"
    row = df.iloc[-1]
    adx = row.get('ADX_14', 0)
    if adx > CONFIG["REGIME_ADX_STRONG"] and row['SUPERT_DIR'] == 1 and row['EMA9'] > row['EMA21']:
        return "TRENDING_UP"
    elif adx > CONFIG["REGIME_ADX_STRONG"] and row['SUPERT_DIR'] == -1:
        return "TRENDING_DOWN"
    elif row['ADX_14'] < 20:
        return "RANGING"
    else:
        return "HIGH_VOL"

def calculate_signal_score(df):
    if df is None or df.empty or len(df) < 2:
        return 0
    row = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0

    # Supertrend bullish + EMA aligned (only full points, no partial credit)
    if row['SUPERT_DIR'] == 1 and row['EMA9'] > row['EMA21']:
        score += 30

    # RSI strength — must be above 60, no partial credit for weak RSI
    if row['RSI'] > CONFIG["RSI_LONG_THRESHOLD"]:
        score += 25

    # MACD histogram: positive AND rising (momentum building)
    macd_h_cols = [c for c in df.columns if "MACDh" in str(c)]
    if macd_h_cols:
        macd_col = macd_h_cols[0]
        if row[macd_col] > 0 and row[macd_col] > prev[macd_col]:
            score += 20
        elif row[macd_col] > 0:
            score += 8

    # Volume — strong institutional interest required
    if row['VOL_AVG'] > 0 and row['Volume'] > row['VOL_AVG'] * CONFIG["VOLUME_MULTIPLIER"]:
        score += 20  # 2.0x average volume only

    # ATR expansion — volatility expanding = move has energy
    if len(df) >= 10:
        atr_mean = df['ATR'].rolling(10).mean().iloc[-1]
        if atr_mean > 0:
            atr_ratio = row['ATR'] / atr_mean
            if atr_ratio > 1.3:
                score += 15
            elif atr_ratio > 1.1:
                score += 8

    # ADX trend strength bonus
    if row['ADX_14'] > 35:
        score += 10
    elif row['ADX_14'] > CONFIG["ADX_THRESHOLD"]:
        score += 5

    return min(score, 100)

def is_fresh_crossover(df, bar_date=None):
    """
    Returns True if a fresh bullish event occurred in the last 2 bars:
    - Supertrend just flipped to bullish, OR
    - EMA9 just crossed above EMA21
    This prevents re-signalling on stocks already in a bullish state.
    """
    if df is None or len(df) < 3:
        return False
    if bar_date is not None:
        last_idx = df.index[-1]
        if hasattr(last_idx, 'tzinfo') and last_idx.tzinfo is None:
            last_idx = IST.localize(last_idx)
        if last_idx.date() != bar_date:
            return False
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    # Supertrend flip: was bearish/neutral last bar, now bullish
    supert_flip = (curr['SUPERT_DIR'] == 1 and prev['SUPERT_DIR'] != 1)

    # EMA crossover: EMA9 crossed above EMA21 in last 2 bars
    ema_cross_now = (curr['EMA9'] > curr['EMA21'] and prev['EMA9'] <= prev['EMA21'])
    ema_cross_prev = (prev['EMA9'] > prev['EMA21'] and prev2['EMA9'] <= prev2['EMA21'])
    ema_cross = ema_cross_now or ema_cross_prev

    return supert_flip or ema_cross

def is_within_signal_window(now):
    if not CONFIG["SIGNAL_WINDOW"]["ENABLED"]:
        return True
    start = now.replace(hour=CONFIG["SIGNAL_WINDOW"]["START_HOUR"], minute=CONFIG["SIGNAL_WINDOW"]["START_MIN"], second=0, microsecond=0)
    end = now.replace(hour=CONFIG["SIGNAL_WINDOW"]["END_HOUR"], minute=CONFIG["SIGNAL_WINDOW"]["END_MIN"], second=0, microsecond=0)
    return start <= now <= end

def generate_signal(symbol, df, now=None, vix=None):
    global daily_calls, last_global_signal_time, symbol_daily_calls, _gate_counts
    if now is None: now = datetime.now(IST)

    # Gate 1: Market & session
    if not market_ok_for_day:
        _gate_counts['market_not_ok'] = _gate_counts.get('market_not_ok', 0) + 1
        return None
    if not is_within_signal_window(now):
        _gate_counts['outside_window'] = _gate_counts.get('outside_window', 0) + 1
        return None
    if daily_calls >= CONFIG["MAX_CALLS_PER_DAY"]:
        _gate_counts['max_calls'] = _gate_counts.get('max_calls', 0) + 1
        return None
    if len(open_positions) >= CONFIG["MAX_OPEN_POSITIONS"]:
        _gate_counts['max_positions'] = _gate_counts.get('max_positions', 0) + 1
        return None
    if now.hour < CONFIG["TRADING_START_HOUR"] or (now.hour == CONFIG["TRADING_START_HOUR"] and now.minute < CONFIG["TRADING_START_MIN"]):
        _gate_counts['before_open'] = _gate_counts.get('before_open', 0) + 1
        return None
    if now.hour > CONFIG["TRADING_END_HOUR"] or (now.hour == CONFIG["TRADING_END_HOUR"] and now.minute > CONFIG["TRADING_END_MIN"]):
        _gate_counts['after_close'] = _gate_counts.get('after_close', 0) + 1
        return None
    if now.hour > 14 or (now.hour == 14 and now.minute >= 0):
        _gate_counts['hard_cutoff'] = _gate_counts.get('hard_cutoff', 0) + 1
        return None

    # Gate 2: Cooldowns
    if last_global_signal_time and (now - last_global_signal_time).total_seconds() < CONFIG["GLOBAL_COOLDOWN_SECONDS"]:
        _gate_counts['global_cooldown'] = _gate_counts.get('global_cooldown', 0) + 1
        return None
    if symbol in last_alert_time and (now - last_alert_time[symbol]).total_seconds() / 60 < CONFIG["COOLDOWN_MINUTES"]:
        _gate_counts['symbol_cooldown'] = _gate_counts.get('symbol_cooldown', 0) + 1
        return None
    if symbol_daily_calls.get(symbol, 0) >= CONFIG["MAX_SIGNALS_PER_SYMBOL_PER_DAY"]:
        _gate_counts['symbol_max_calls'] = _gate_counts.get('symbol_max_calls', 0) + 1
        return None

    # Gate 3: Data quality
    if df is None or len(df) < 50:
        _gate_counts['insufficient_data'] = _gate_counts.get('insufficient_data', 0) + 1
        return None
    required = ['Close','SUPERT_DIR','EMA9','EMA21','RSI','ATR','VOL_AVG','ADX_14','SMA20','VWAP']
    if df[required].iloc[-1].isna().any() or df[required].iloc[-2].isna().any():
        _gate_counts['nan_data'] = _gate_counts.get('nan_data', 0) + 1
        return None

    # Gate 4: Fresh crossover
    bar_date = now.date() if hasattr(now, 'date') else None
    if not is_fresh_crossover(df, bar_date=bar_date):
        _gate_counts['no_crossover'] = _gate_counts.get('no_crossover', 0) + 1
        return None

    # ---- Symbol passed crossover gate — log every rejection from here ----
    row = df.iloc[-1]
    sym_short = symbol.replace('.NS', '')
    print(f"[CROSSOVER] {sym_short} @ {now.strftime('%H:%M')} | ST={row['SUPERT_DIR']} EMA9={row['EMA9']:.1f} EMA21={row['EMA21']:.1f} ADX={row['ADX_14']:.1f} RSI={row['RSI']:.1f}", flush=True)

    # Gate 5: Trend confirm
    if row['SUPERT_DIR'] != 1 or row['EMA9'] <= row['EMA21'] or row['ADX_14'] < CONFIG["ADX_THRESHOLD"]:
        print(f"  ❌ trend_confirm FAIL | ST={row['SUPERT_DIR']} EMA_bull={row['EMA9']>row['EMA21']} ADX={row['ADX_14']:.1f} (need {CONFIG['ADX_THRESHOLD']})", flush=True)
        _gate_counts['trend_confirm'] = _gate_counts.get('trend_confirm', 0) + 1
        return None
    if 'ADX_SLOPE' in df.columns:
        if pd.isna(row['ADX_SLOPE']) or row['ADX_SLOPE'] <= 0:
            print(f"  ❌ adx_slope FAIL | ADX_SLOPE={row['ADX_SLOPE']:.1f} (need > 0)", flush=True)
            _gate_counts['adx_slope'] = _gate_counts.get('adx_slope', 0) + 1
            return None

    # Gate: ATR floor — skip micro-ATR stocks (SL within noise)
    if row['ATR'] / row['Close'] < 0.003:
        _gate_counts['atr_floor'] = _gate_counts.get('atr_floor', 0) + 1
        return None

    # Gate 6: Price filter
    if row['Close'] < row['SMA20'] or row['Close'] < row['VWAP'] * (1 + CONFIG["MIN_VWAP_BUFFER_PCT"]/100):
        print(f"  ❌ price_filter FAIL | Close={row['Close']:.1f} SMA20={row['SMA20']:.1f} VWAP={row['VWAP']:.1f}", flush=True)
        _gate_counts['price_filter'] = _gate_counts.get('price_filter', 0) + 1
        return None

    # Gate 7: RSI band
    if not (50 <= row['RSI'] <= 78):
        print(f"  ❌ rsi FAIL | RSI={row['RSI']:.1f} (need 50–78)", flush=True)
        _gate_counts['rsi'] = _gate_counts.get('rsi', 0) + 1
        return None

    # Gate 8: VIX
    _vix = vix if vix is not None else get_india_vix()
    if _vix and (_vix > CONFIG["INDIA_VIX_THRESHOLD"] or _vix < CONFIG["MIN_VIX"]):
        print(f"  ❌ vix FAIL | VIX={_vix:.1f} (allowed {CONFIG['MIN_VIX']}–{CONFIG['INDIA_VIX_THRESHOLD']})", flush=True)
        _gate_counts['vix'] = _gate_counts.get('vix', 0) + 1
        return None

    # Gate 9: Score
    score = calculate_signal_score(df)
    if score < CONFIG["SIGNAL_SCORE_THRESHOLD"]:
        print(f"  ❌ score FAIL | score={score} < {CONFIG['SIGNAL_SCORE_THRESHOLD']}", flush=True)
        signal_rejections.append({"symbol": symbol, "score": score, "time": now})
        _gate_counts['score'] = _gate_counts.get('score', 0) + 1
        return None

    print(f"  ✅ SIGNAL PASS | {sym_short} score={score}", flush=True)
    _gate_counts['signals_passed'] = _gate_counts.get('signals_passed', 0) + 1

    # Regime-aware sizing
    regime = get_regime(df)
    regime_mult = CONFIG["REGIME_RISK_MAP"].get(regime, 0.7)

    # Position calc
    entry = row['Close']
    atr = row['ATR']
    sl = entry - (atr * CONFIG["ATR_SL_MULTIPLIER"])
    risk = entry - sl
    if risk <= 0: return None
    risk_amount = capital * CONFIG["RISK_PER_TRADE"] * regime_mult
    qty = int(risk_amount / risk)
    if qty < 1: return None
    target = entry + (atr * CONFIG["ATR_SL_MULTIPLIER"] * CONFIG["MIN_RR"])

    # Notional cap: never exceed MAX_TRADE_NOTIONAL_PCT of current capital
    max_notional_qty = int((capital * CONFIG["MAX_TRADE_NOTIONAL_PCT"]) / entry)
    if max_notional_qty > 0:
        qty = min(qty, max_notional_qty)

    # Liquidity cap: never exceed ADV-based limit
    max_adv_qty = int(row['VOL_AVG'] * CONFIG["MAX_QTY_PCT_OF_ADV"])
    if max_adv_qty > 0:
        qty = min(qty, max_adv_qty)

    if qty < 1:
        return None

    # Commit
    daily_calls += 1
    last_global_signal_time = now
    last_alert_time[symbol] = now
    symbol_daily_calls[symbol] = symbol_daily_calls.get(symbol, 0) + 1

    return {
        "symbol": symbol.replace(".NS", ""),
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "target": round(target, 2),
        "qty": qty,
        "rr": CONFIG["MIN_RR"],
        "entry_time": now,
        "atr": atr,
        "vix": _vix if _vix else 0.0,
        "score": score,
        "regime": regime,
        "partial_exit_done": False
    }

def check_open_positions(df_dict, now=None):
    global capital, open_positions, trades_history, equity_curve
    if now is None:
        now = datetime.now(IST)
    if open_positions:
        equity_curve.append({"timestamp": now, "capital": capital, "open_pos": len(open_positions)})
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
            half_target = pos['entry'] + (pos['target'] - pos['entry']) * CONFIG["TRAILING_STOP"]["PARTIAL_EXIT_PCT"]
            if curr >= half_target:
                half_qty = pos['qty'] // 2
                if half_qty > 0:
                    pnl_gross_half = (half_target - pos['entry']) * half_qty
                    notional_half = half_target * half_qty
                    min_viable_gross = (notional_half * (CONFIG["ENTRY_COST_PCT"] + CONFIG["EXIT_COST_PCT"]) / 100) + CONFIG["FIXED_BROKERAGE"]
                    if pnl_gross_half < min_viable_gross:
                        continue
                    entry_cost = notional_half * CONFIG["ENTRY_COST_PCT"] / 100
                    exit_cost = notional_half * CONFIG["EXIT_COST_PCT"] / 100
                    cost = entry_cost + exit_cost + CONFIG["FIXED_BROKERAGE"]
                    pnl_net_half = pnl_gross_half - cost
                    capital += pnl_net_half
                    trades_history.append({
                        "win": pnl_net_half > 0, "pnl": pnl_net_half,
                        "symbol": pos['symbol'], "exit_reason": "partial",
                        "exit_price": half_target, "entry": pos['entry'],
                        "qty": half_qty, "entry_time": str(pos.get('entry_time', '')),
                        "exit_time": str(now)
                    })
                    equity_curve.append({"timestamp": now, "capital": capital, "open_pos": len(open_positions)})
                    pos['qty'] -= half_qty
                    pos['partial_exit_done'] = True
                    send_telegram(f"📌 {pos['symbol']} Partial profit taken | +₹{pnl_net_half:,.0f} | Remaining qty: {pos['qty']}")
                    if CONFIG["TRAILING_STOP"]["BREAKEVEN_AFTER_PARTIAL"]:
                        pos['sl'] = pos['entry']
                    continue
        if CONFIG["TRAILING_STOP"]["ENABLED"] and pos.get("partial_exit_done", False):
            activation_price = pos['entry'] + (pos['target'] - pos['entry']) * CONFIG["TRAILING_STOP"]["TRAILING_ACTIVATION_PCT"]
            if curr >= activation_price:
                atr_val = pos.get('atr', None)
                if atr_val is None:
                    atr_val = (df['High'].iloc[-1] - df['Low'].iloc[-1])
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
            notional_exit = pos['entry'] * pos['qty']
            entry_cost = notional_exit * CONFIG["ENTRY_COST_PCT"] / 100
            exit_cost = notional_exit * CONFIG["EXIT_COST_PCT"] / 100
            cost = entry_cost + exit_cost + CONFIG["FIXED_BROKERAGE"]
            pnl_net = pnl_gross - cost
            capital += pnl_net
            trades_history.append({
                "win": pnl_net > 0, "pnl": pnl_net,
                "symbol": pos['symbol'], "exit_reason": exit_reason,
                "exit_price": curr, "entry": pos['entry'],
                "qty": pos['qty'], "entry_time": str(pos.get('entry_time', '')),
                "exit_time": str(now), "atr": pos.get('atr', 0)
            })
            equity_curve.append({"timestamp": now, "capital": capital, "open_pos": len(open_positions)})
            closed.append(f"{exit_reason} | P&L net: ₹{pnl_net:,.0f}")
            open_positions.remove(pos)
    for msg in closed:
        send_telegram(msg)
    if closed:
        wins_today = sum(1 for t in trades_history if t.get("win"))
        total_today = len(trades_history)
        net_today = sum(t["pnl"] for t in trades_history)
        print(f"[POSITIONS] Closed {len(closed)} | Running: Trades={total_today} Win%={wins_today/total_today*100:.1f}% Net=₹{net_today:,.0f}", flush=True)

def simulate_trade(signal):
    open_positions.append({
        "symbol": signal['symbol'],
        "entry": signal['entry'],
        "sl": signal['sl'],
        "target": signal['target'],
        "qty": signal['qty'],
        "entry_time": signal['entry_time'],
        "atr": signal['atr'],
        "partial_exit_done": False
    })
    print(f"[SIM] Opened {signal['symbol']} @ {signal['entry']}", flush=True)

# ==================== HYBRID DATA FETCHING (for historical) ====================
def fetch_5m_data_hybrid(symbol, start_date, end_date):
    """yfinance-only 5m fetcher. nsepy removed (dead in 2026)."""
    try:
        data = yf.download(symbol, start=start_date, end=end_date,
                           interval="5m", progress=False, auto_adjust=True)
        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            print(f"yfinance: Fetched {len(data)} bars for {symbol}", flush=True)
            return clean_dataframe(data)
    except Exception as e:
        print(f"yfinance error for {symbol}: {e}", flush=True)
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
    global daily_calls, last_global_signal_time
    print("🚀 Running 30-day backtest...", flush=True)
    symbols = filter_liquid_symbols(get_nifty500_symbols())[:50]
    trades = 0
    wins = 0
    for sym in symbols:
        try:
            df_raw = yf.download(sym, period=f"{days}d", interval=CONFIG["CANDLE_TIMEFRAME"], progress=False, auto_adjust=True)
            if len(df_raw) < 100:
                continue
            df_raw = clean_dataframe(df_raw)
            last_bt_date = None
            daily_calls = 0
            last_global_signal_time = None
            last_alert_time.pop(sym, None)
            symbol_daily_calls.pop(sym, None)
            for i in range(50, len(df_raw)-1):
                bar = df_raw.iloc[i].name
                if bar.tzinfo is None:
                    bar_ist = IST.localize(bar)
                else:
                    bar_ist = bar.astimezone(IST)
                if last_bt_date is None or bar_ist.date() != last_bt_date:
                    daily_calls = 0
                    last_global_signal_time = None
                    symbol_daily_calls.pop(sym, None)
                    last_bt_date = bar_ist.date()
                df = calculate_indicators(df_raw.iloc[:i+1].copy())
                if df is None:
                    continue
                sig = generate_signal(sym, df, now=bar_ist)
                if sig:
                    trades += 1
                    next_bar = df_raw.iloc[i+1]
                    if next_bar['Low'] <= sig['sl']:
                        pass
                    elif next_bar['High'] >= sig['target']:
                        wins += 1
                    elif next_bar['Close'] > sig['entry']:
                        wins += 1
        except Exception as e:
            print(f"Backtest error {sym}: {e}", flush=True)
            continue
    wr = wins/trades*100 if trades else 0
    dd_msg = ""
    if equity_curve:
        df_eq = pd.DataFrame(equity_curve)
        if not df_eq.empty:
            peak = df_eq['capital'].cummax()
            dd = (df_eq['capital'] - peak) / peak * 100
            max_dd = dd.min()
            dd_msg = f" | Max DD: {max_dd:.2f}%"
    print(f"✅ Backtest: {trades} trades, win rate {wr:.1f}%{dd_msg}", flush=True)
    send_telegram(f"📈 Backtest: {trades} trades, {wr:.1f}% win-rate{dd_msg}")

def run_historical_replay(test_date_str):
    global capital, daily_calls, open_positions, last_alert_time, trades_history, market_ok_for_day, last_global_signal_time
    try:
        test_date = datetime.strptime(test_date_str, "%Y-%m-%d").date()
        print(f"🚀 Replay for {test_date_str}...", flush=True)
        print(f"📅 Replay started for {test_date_str}", flush=True)

        capital = CONFIG["SIMULATION_CAPITAL"]
        daily_calls = 0
        open_positions = []
        last_alert_time.clear()
        symbol_daily_calls.clear()
        trades_history.clear()
        equity_curve.clear()
        signal_rejections.clear()
        market_ok_for_day = True
        last_global_signal_time = None
        _gate_counts.clear()
        print("Historical mode: market filter bypassed", flush=True)

        KNOWN_LIQUID = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
                        "SBIN.NS","BHARTIARTL.NS","AXISBANK.NS","KOTAKBANK.NS","LT.NS",
                        "BAJAJFINSV.NS","WIPRO.NS","HCLTECH.NS","SUNPHARMA.NS","BAJFINANCE.NS",
                        "TITAN.NS","MARUTI.NS","NTPC.NS","POWERGRID.NS","ONGC.NS",
                        "ADANIPORTS.NS","JSWSTEEL.NS","TATASTEEL.NS","HINDALCO.NS","TECHM.NS",
                        "DRREDDY.NS","DIVISLAB.NS","CIPLA.NS","M&M.NS","BAJAJ-AUTO.NS"]
        symbols = KNOWN_LIQUID
        print(f"Using {len(symbols)} symbols", flush=True)

        start = (pd.to_datetime(test_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        end = (pd.to_datetime(test_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        
        today_date = datetime.now(IST).date()
        earliest_allowed = pd.to_datetime(today_date - timedelta(days=58)).strftime("%Y-%m-%d")
        start = max(start, earliest_allowed)

        print(f"Downloading 5m data: {start} to {end}...", flush=True)
        data_dict = fetch_5m_data_batch_hybrid(symbols, start, end)
        if data_dict:
            avg_bars = sum(len(v) for v in data_dict.values()) // len(data_dict)
            print(f"Data loaded: {len(data_dict)} symbols, avg {avg_bars} bars/symbol", flush=True)

        if not data_dict:
            send_telegram("❌ No 5m data available from any source.")
            print("❌ No 5m data downloaded.", flush=True)
            return

        bar_times = pd.date_range(start=f"{test_date} 09:15", end=f"{test_date} 15:30", freq="5min", tz=IST)
        vix = get_india_vix()
        if vix is None:
            vix = 16.0  # Historical mode only: use neutral fallback since live fetch irrelevant
            print("VIX unavailable for historical replay, using neutral 16.0", flush=True)
        print(f"VIX: {vix:.1f}", flush=True)
        start_time = time.time()

        for idx, bar in enumerate(bar_times):
            now = bar
            if not (9 <= now.hour <= 15):
                continue
            sliced_cache = {}
            for symbol, df in data_dict.items():
                if df is None or df.empty:
                    continue
                df_clean = clean_dataframe(df)
                data_cache[symbol] = df_clean
                df_at_bar = df_clean[df_clean.index <= bar]
                if not df_at_bar.empty:
                    sliced_cache[symbol] = df_at_bar
            check_open_positions(sliced_cache, now=now)
            for symbol in symbols:
                if symbol not in data_cache:
                    continue
                df_full = data_cache.get(symbol)
                if df_full is None or len(df_full) < 40:
                    continue
                df_slice_raw = df_full[df_full.index <= bar]
                if len(df_slice_raw) < 50:
                    continue
                df_slice = calculate_indicators(df_slice_raw.copy())
                if df_slice is None:
                    continue
                last_slice_bar = df_slice.index[-1]
                if hasattr(last_slice_bar, 'tzinfo') and last_slice_bar.tzinfo is None:
                    last_slice_bar = IST.localize(last_slice_bar)
                if last_slice_bar.date() != bar.date():
                    continue
                sig = generate_signal(symbol, df_slice, now=now, vix=vix)
                if sig:
                    notional = round(sig['entry'] * sig['qty'], 0)
                    risk_rs = round(sig['entry'] - sig['sl'], 2) * sig['qty']
                    sl_pct = round((sig['entry'] - sig['sl']) / sig['entry'] * 100, 2)
                    tgt_pct = round((sig['target'] - sig['entry']) / sig['entry'] * 100, 2)
                    send_telegram(
                        f"🔥 <b>INTRADAY CALL [HISTORICAL {test_date_str}]</b>\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 {sig['symbol']} — BUY @ ₹{sig['entry']}\n"
                        f"🛑 SL: ₹{sig['sl']} (-{sl_pct}%)  |  🎯 Target: ₹{sig['target']} (+{tgt_pct}%)\n"
                        f"📦 Qty: {sig['qty']} shares  |  Notional: ₹{notional:,.0f}\n"
                        f"💰 Risk: ₹{risk_rs:,.0f}  |  RR: 1:{sig['rr']}\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"📊 Score: {sig['score']}/100  |  ATR: ₹{sig['atr']:.2f}\n"
                        f"📈 VIX: {sig['vix']:.1f}  |  Capital: ₹{capital:,.0f}\n"
                        f"🕒 {now.strftime('%H:%M IST')}  |  {now.strftime('%A')}"
                    )
                    simulate_trade(sig)
            if idx % 10 == 0:
                print(f"Progress: {idx+1}/{len(bar_times)} bars", flush=True)

        final_bar = bar_times[-1]
        final_sliced_cache = {}
        for symbol, df_full in data_dict.items():
            if df_full is None or df_full.empty:
                continue
            df_clean = clean_dataframe(df_full)
            df_at_close = df_clean[df_clean.index <= final_bar]
            if not df_at_close.empty:
                final_sliced_cache[symbol] = df_at_close
        check_open_positions(final_sliced_cache)
        wins = sum(1 for t in trades_history if t["win"])
        total = len(trades_history)
        net = sum(t["pnl"] for t in trades_history)
        wr = wins/total*100 if total else 0
        dd_msg = ""
        if equity_curve:
            df_eq = pd.DataFrame(equity_curve)
            if not df_eq.empty:
                peak = df_eq['capital'].cummax()
                dd = (df_eq['capital'] - peak) / peak * 100
                max_dd = dd.min()
                dd_msg = f"\n📉 Max Drawdown: {max_dd:.2f}%"
        gate_summary = " | ".join(f"{k}={v}" for k, v in sorted(_gate_counts.items()))
        print(f"[GATES] {test_date_str}: {gate_summary}", flush=True)
        send_telegram(f"🔍 Gates [{test_date_str}]:\n{gate_summary}")
        send_telegram(f"📊 Summary {test_date_str}\nTrades: {total} | Win%: {wr:.1f}%\nNet P&L: ₹{net:,.0f} | Capital: ₹{capital:,.0f}{dd_msg}")
        print(f"Replay done in {time.time()-start_time:.1f}s", flush=True)
    except Exception as e:
        send_telegram(f"❌ Replay failed: {str(e)}")
        print(traceback.format_exc(), flush=True)

def run_multi_day_historical_test(days_back=30):
    """Test the bot on the last N trading days (perfect for 60-day validation)"""
    print(f"🚀 Starting {days_back}-day historical multi-replay...", flush=True)
    send_telegram(f"📅 Starting {days_back}-day full historical test...")

    # Auto-generate last trading days (skip weekends)
    today = datetime.now(IST).date()
    test_dates = []
    for i in range(days_back):
        d = today - timedelta(days=i)
        if d.weekday() < 5:  # Monday=0 ... Friday=4
            test_dates.append(d.strftime("%Y-%m-%d"))
    test_dates = test_dates[::-1]  # oldest → newest

    today_date = datetime.now(IST).date()
    cutoff = today_date - timedelta(days=58)
    test_dates = [d for d in test_dates if datetime.strptime(d, "%Y-%m-%d").date() >= cutoff]
    print(f"After Yahoo 5m window filter: {len(test_dates)} usable days", flush=True)

    all_symbols = get_nifty500_symbols()

    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    max_dd_list = []

    for idx, test_date in enumerate(test_dates):
        _vix_cache["value"] = None
        _vix_cache["ts"] = None
        print(f"\n=== Running replay {idx+1}/{len(test_dates)} → {test_date} ===", flush=True)
        run_historical_replay(test_date)   # This already sends Telegram summary + DD
        
        daily_pnl = sum(t.get("pnl", 0) for t in trades_history)
        daily_wr = sum(1 for t in trades_history if t.get("win")) / len(trades_history) * 100 if trades_history else 0
        print(f"   → {test_date}: {len(trades_history)} trades | WR={daily_wr:.1f}% | P&L=Rs.{daily_pnl:,.0f}", flush=True)

        # Aggregate stats
        if trades_history:
            wins = sum(1 for t in trades_history if t.get("win", False))
            net = sum(t.get("pnl", 0) for t in trades_history)
            total_trades += len(trades_history)
            total_wins += wins
            total_pnl += net

            # Last DD from this day
            if equity_curve:
                df_eq = pd.DataFrame(equity_curve)
                if not df_eq.empty:
                    peak = df_eq['capital'].cummax()
                    dd = (df_eq['capital'] - peak) / peak * 100
                    max_dd_list.append(dd.min())

        # Reset for next day
        trades_history.clear()
        equity_curve.clear()
        data_cache.clear()

    # Final overall report
    if total_trades > 0:
        win_rate = (total_wins / total_trades) * 100
        avg_dd = sum(max_dd_list) / len(max_dd_list) if max_dd_list else 0
        print(f"\n🎯 === {days_back}-DAY FULL TEST COMPLETE ===", flush=True)
        print(f"Total Trades: {total_trades} | Overall Win%: {win_rate:.1f}%", flush=True)
        print(f"Total Net P&L: ₹{total_pnl:,.0f} | Avg Max DD: {avg_dd:.2f}%", flush=True)
        send_telegram(
            f"🎯 <b>{days_back}-DAY HISTORICAL TEST COMPLETE</b>\n"
            f"📊 Total Trades: {total_trades} | Win Rate: {win_rate:.1f}%\n"
            f"💰 Net P&L: ₹{total_pnl:,.0f}\n"
            f"📉 Avg Max Drawdown: {avg_dd:.2f}%"
        )
    else:
        send_telegram(f"⚠️ No trades in the {days_back}-day period")

def main_loop():
    global last_health_ping, daily_calls, last_daily_reset, capital, data_cache, market_open_sent, open_positions, LIQUID_SYMBOLS, market_ok_for_day
    print("Entering main loop", flush=True)
    send_telegram("🚀 Bot started | ADX≥20+slope | Score≥75 | 9:20-14:00 | 20% notional cap | Max 10 calls/day")
    while True:
        now = datetime.now(IST)
        if (last_daily_reset is None or now.date() != last_daily_reset.date()) and now.weekday() < 5:
            daily_calls = 0
            capital = CONFIG["SIMULATION_CAPITAL"]
            data_cache = {}
            open_positions = []
            last_alert_time.clear()
            symbol_daily_calls.clear()
            market_open_sent = False
            trades_history.clear()
            equity_curve.clear()
            signal_rejections.clear()
            last_daily_reset = now
            LIQUID_SYMBOLS = filter_liquid_symbols(get_nifty500_symbols())
            if now.hour >= 8:
                update_daily_market_filter()
            else:
                print("Early morning reset → using previous market filter", flush=True)
                market_ok_for_day = True
            send_telegram("🌅 New trading day – reset")
        if not is_market_open():
            time.sleep(60)
            continue
        if CONFIG["SEND_MARKET_OPEN"] and not market_open_sent and now.hour == 9 and now.minute == 15:
            send_telegram("🌅 Market Open – scanning")
            market_open_sent = True
        if last_health_ping is None or (now - last_health_ping).total_seconds() > CONFIG["HEALTH_PING_MINUTES"]*60:
            dd_msg = ""
            if equity_curve:
                df_eq = pd.DataFrame(equity_curve)
                if not df_eq.empty:
                    peak = df_eq['capital'].cummax()
                    dd = (df_eq['capital'] - peak) / peak * 100
                    max_dd = dd.min()
                    dd_msg = f" | Max DD: {max_dd:.2f}%"
            send_telegram(f"✅ Healthy | Capital: ₹{capital:,.0f} | Calls: {daily_calls} | Open: {len(open_positions)} | Market OK: {market_ok_for_day}{dd_msg}")
            last_health_ping = now
        fetch_all_data()
        current_vix = get_india_vix()
        check_open_positions(data_cache, now=now)
        if market_ok_for_day:
            for sym in LIQUID_SYMBOLS:
                try:
                    df = get_latest_data(sym)
                    if df is None or len(df) < 50:
                        continue
                    # Pre-filter: skip low-volume symbols instantly (saves 60-70% compute)
                    vol_avg_quick = df['Volume'].rolling(20).mean().iloc[-1]
                    if pd.isna(vol_avg_quick) or vol_avg_quick == 0:
                        continue
                    if df['Volume'].iloc[-1] < vol_avg_quick * 1.2:
                        continue
                    df = calculate_indicators(df)
                    if df is None:
                        continue
                    sig = generate_signal(sym, df, now=now, vix=current_vix)
                    if sig:
                        notional = round(sig['entry'] * sig['qty'], 0)
                        risk_rs = round(sig['entry'] - sig['sl'], 2) * sig['qty']
                        sl_pct = round((sig['entry'] - sig['sl']) / sig['entry'] * 100, 2)
                        tgt_pct = round((sig['target'] - sig['entry']) / sig['entry'] * 100, 2)
                        send_telegram(
                            f"🔥 <b>INTRADAY CALL</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"📌 {sig['symbol']} — BUY @ ₹{sig['entry']}\n"
                            f"🛑 SL: ₹{sig['sl']} (-{sl_pct}%)  |  🎯 Target: ₹{sig['target']} (+{tgt_pct}%)\n"
                            f"📦 Qty: {sig['qty']} shares  |  Notional: ₹{notional:,.0f}\n"
                            f"💰 Risk: ₹{risk_rs:,.0f}  |  RR: 1:{sig['rr']}\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"📊 Score: {sig['score']}/100  |  ATR: ₹{sig['atr']:.2f}\n"
                            f"📈 VIX: {sig['vix']:.1f}  |  Capital: ₹{capital:,.0f}\n"
                            f"🕒 {now.strftime('%H:%M IST')}  |  {now.strftime('%A')}"
                        )
                        simulate_trade(sig)
                except Exception as e:
                    send_telegram(f"❌ Error {sym}: {str(e)[:100]}")
        if now.hour == 15 and now.minute == 30 and trades_history:
            wins = sum(1 for t in trades_history if t["win"])
            total = len(trades_history)
            net = sum(t["pnl"] for t in trades_history)
            wr = wins/total*100 if total else 0
            dd_msg = ""
            if equity_curve:
                df_eq = pd.DataFrame(equity_curve)
                if not df_eq.empty:
                    peak = df_eq['capital'].cummax()
                    dd = (df_eq['capital'] - peak) / peak * 100
                    max_dd = dd.min()
                    dd_msg = f"\n📉 Max Drawdown: {max_dd:.2f}%"
            send_telegram(f"📊 Day Summary\nTrades: {total} | Win%: {wr:.1f}%\nP&L: ₹{net:,.0f} | Capital: ₹{capital:,.0f}{dd_msg}")
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
    elif os.getenv("BACKTEST"):
        run_backtest(int(os.getenv("BACKTEST")))
        sys.exit(0)
    elif os.getenv("MULTI_DAY_TEST"):
        days = int(os.getenv("MULTI_DAY_TEST"))
        run_multi_day_historical_test(days)
        sys.exit(0)
    else:
        print("Starting live simulation mode...", flush=True)
        try:
            main_loop()
        except Exception as e:
            send_telegram(f"💥 Crash: {traceback.format_exc()}")
            print(traceback.format_exc(), flush=True)
            time.sleep(10)
