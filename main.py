from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import time
import os  
import math
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Secured Markov Screener Engine with Risk Profiles")

# ==========================================
# SECURITY: API KEY VALIDATION
# ==========================================
SECRET_KEY = os.environ.get("API_SECRET_KEY")

async def verify_token(authorization: str = Header(...)):
    if not SECRET_KEY or authorization != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing API Key")

# ==========================================
# CORS SETUP
# ==========================================
origins = [
    "https://www.stockscreen.art",
    "https://stockscreen.art"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global caches
INDIVIDUAL_CACHE = {}
SCREENER_CACHE = {"data": None, "timestamp": 0}
CACHE_EXPIRY_SECONDS = 3600  

SCREENER_TICKERS = [
    "SPY", "QQQ", "DIA", "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", 
    "TSLA", "NFLX", "XOM", "JPM", "V", "MA", "LLY", "UNH", "COST", "TPL", "FIX",
    "XIU.TO", "CSU.TO", "TOI.TO", "LMN.TO", "SHOP.TO", "ATD.TO", "BN.TO", "CNQ.TO", 
    "CP.TO", "CNR.TO", "TD.TO", "RY.TO", "BMO.TO", "CCO.TO", "BCE.TO", "ENB.TO"
]

# ==========================================
# SYSTEM RISK PROFILES SPECIFICATION CONFIG
# ==========================================
RISK_MULTIPLIERS = {
    "conservative": {"stop": 1.5, "target": 2.5, "buffer": 0.998},
    "moderate":     {"stop": 2.0, "target": 3.5, "buffer": 1.000},
    "aggressive":   {"stop": 3.0, "target": 5.0, "buffer": 1.002}
}

# ==========================================
# TECHNICAL INDICATOR CALCULATIONS
# ==========================================

def calculate_rsi(series, period=14):
    if series.empty or len(series) < period:
        return np.nan
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.iloc[-1]

def calculate_atr(df, period=14):
    if len(df) < period:
        return np.nan
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_series.iloc[-1]

def calculate_macd_signal_str(series, fast=12, slow=26, signal=9):
    if series.empty or len(series) < slow + signal:
        return "N/A"
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    c_macd, c_sig = macd_line.iloc[-1], signal_line.iloc[-1]
    p_macd, p_sig = macd_line.iloc[-2], signal_line.iloc[-2]
    
    if c_macd > c_sig and p_macd <= p_sig:
        return "Bullish Crossover"
    elif c_macd < c_sig and p_macd >= p_sig:
        return "Bearish Crossover"
    else:
        return "Bullish Trend" if c_macd > c_sig else "Bearish Trend"

# ==========================================
# TRADING SIGNAL SYSTEM LOGIC
# ==========================================

def generate_trading_signal(regime, rsi, macd_str, current_price, atr, risk_profile="moderate"):
    """
    Quant Matrix v2: Computes dynamic risk boundaries by multiplying active 
    asset ATR profiles against selected user safety metrics.
    """
    try:
        current_price = float(current_price)
        rsi = float(rsi) if (rsi != "N/A" and not pd.isna(rsi)) else 50.0
        atr = float(atr) if (atr != "N/A" and not pd.isna(atr)) else (current_price * 0.02)
        
        profile = str(risk_profile).lower().strip()
        if profile not in RISK_MULTIPLIERS:
            profile = "moderate"
    except:
        return {"action": "HOLD", "color": "#cbd5e1", "entry": "N/A", "stop": "N/A", "target": "N/A"}

    cfg = RISK_MULTIPLIERS[profile]
    action = "HOLD"
    color = "#cbd5e1" 
    
    # Calculate Risk Boundaries using Profile Multipliers
    stop_loss = round(current_price - (cfg["stop"] * atr), 2)
    target_price = round(current_price + (cfg["target"] * atr), 2)
    suggested_entry = round(current_price * cfg["buffer"], 2)

    # 1. Profile: STRONG BUY
    if regime == "Bull" and rsi < 48 and "Bullish" in macd_str:
        action = "STRONG BUY"
        color = "#22c55e"
        
    # 2. Profile: BUY
    elif regime == "Bull" and rsi < 65:
        action = "BUY"
        color = "#4ade80"

    # 3. Profile: RANGE ACCUMULATION
    elif regime == "Sideways" and rsi < 38:
        action = "ACCUMULATE"
        color = "#00d4ff"
        stop_loss = round(current_price - (max(1.0, cfg["stop"] - 0.5) * atr), 2) 

    # 4. Profile: EXHAUSTION
    elif rsi > 76:
        action = "TAKE PROFIT"
        color = "#eab308"
        stop_loss = round(current_price * 0.97, 2)
        target_price = "N/A"
        suggested_entry = "N/A"

    # 5. Profile: BEARISH AVOID
    elif regime == "Bear" and ("Bearish" in macd_str or rsi > 60):
        action = "STRONG SELL / AVOID"
        color = "#ef4444"
        suggested_entry = "N/A"
        stop_loss = "N/A"
        target_price = "N/A"

    return {
        "action": f"{action}",
        "color": color,
        "entry": f"${suggested_entry}" if suggested_entry != "N/A" else "N/A",
        "stop": f"${stop_loss}" if stop_loss != "N/A" else "N/A",
        "target": f"${target_price}" if target_price != "N/A" else "N/A"
    }

# ==========================================
# CORE MARKOV REGIME MATHEMATICS PIPELINE
# ==========================================

def calculate_single_markov(ticker, window=20, threshold=0.012, risk_profile="moderate"):
    try:
        df = yf.download(ticker, period="1y", progress=False, group_by=False)
        if df.empty or len(df) < window: return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        if 'adj close' in df.columns and 'close' not in df.columns:
            df['close'] = df['adj close']
            
        df['close'] = pd.to_numeric(df['close'], errors='coerce').astype(float)
        df['high'] = pd.to_numeric(df['high'], errors='coerce').astype(float)
        df['low'] = pd.to_numeric(df['low'], errors='coerce').astype(float)
        df = df.dropna(subset=['close', 'high', 'low'])
        
        latest_rsi = calculate_rsi(df['close'])
        latest_atr = calculate_atr(df)
        macd_desc = calculate_macd_signal_str(df['close'])
        
        df['Log_Ret'] = np.log(df['close'] / df['close'].shift(1))
        df['Roll_Mom'] = df['Log_Ret'].rolling(window).sum()
        
        def classify(val):
            if pd.isna(val): return 'Sideways'
            return 'Bull' if val > threshold else ('Bear' if val < -threshold else 'Sideways')
            
        df['State'] = df['Roll_Mom'].apply(classify)
        df['Next_State'] = df['State'].shift(-1)
        
        states = ['Bull', 'Sideways', 'Bear']
        matrix = {s: {next_s: 0.0 for next_s in states} for s in states}
        for _, row in df.dropna(subset=['State', 'Next_State']).iterrows():
            matrix[row['State']][row['Next_State']] += 1.0
            
        for s in states:
            total = sum(matrix[s].values())
            if total > 0:
                for next_s in states: matrix[s][next_s] = round(matrix[s][next_s] / total, 3)
            else: matrix[s][s] = 1.0
                
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or df['close'].iloc[-1] or 0.0
        year_high = info.get('fiftyTwoWeekHigh') or df['close'].max() or 0.0
        year_low = info.get('fiftyTwoWeekLow') or df['close'].min() or 0.0
        target_mean = info.get('targetMeanPrice') or current_price
        regime_score = round(((target_mean / current_price) - 1) * 100, 2) if current_price > 0 else 0.0
        current_state = df['State'].values[-1] if hasattr(df['State'], 'values') else df['State'].iloc[-1]
        trailing_return = float((df['close'].values[-1] / df['close'].values[0]) - 1)
        
        # Compute volatility signal with profile selection parameters
        signal_data = generate_trading_signal(current_state, latest_rsi, macd_desc, current_price, latest_atr, risk_profile)
        
        return {
            "ticker": str(ticker), 
            "current_regime": str(current_state), 
            "p_bull_bull": float(matrix['Bull']['Bull']),
            "transition_matrix": matrix, 
            "trailing_return": round(trailing_return * 100, 2), 
            "sample_days": int(len(df)),
            "current_price": float(current_price), 
            "year_high": float(year_high), 
            "year_low": float(year_low), 
            "regime_score": regime_score,
            "rsi": round(float(latest_rsi), 2) if (not pd.isna(latest_rsi) and not math.isnan(latest_rsi)) else "N/A",
            "atr": round(float(latest_atr), 2) if (not pd.isna(latest_atr) and not math.isnan(latest_atr)) else "N/A",
            "macd_signal": str(macd_desc), 
            "trade_signal": signal_data
        }
    except Exception as e:
        with open("indicator_error_log.txt", "a") as f:
            f.write(f"Error parsing calculations for {ticker}: {str(e)}\n")
        return None

# ==========================================
# SECURED ENDPOINTS
# ==========================================

@app.get("/api/screener", dependencies=[Depends(verify_token)])
def get_top_screener():
    current_time = time.time()
    if SCREENER_CACHE["data"] and (current_time - SCREENER_CACHE["timestamp"] < CACHE_EXPIRY_SECONDS):
        return SCREENER_CACHE["data"]
        
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        computed = executor.map(lambda t: calculate_single_markov(t, risk_profile="moderate"), SCREENER_TICKERS)
        for res in computed:
            if res: results.append(res)
                
    sorted_results = sorted(results, key=lambda x: (-x['p_bull_bull'], -x['trailing_return']))
    top_25 = sorted_results[:25]
    
    SCREENER_CACHE["data"] = top_25
    SCREENER_CACHE["timestamp"] = current_time
    return top_25

@app.get("/api/regime", dependencies=[Depends(verify_token)])
def get_individual_regime(ticker: str, window: int = 20, threshold: float = 0.012, risk: str = "moderate"):
    ticker_key = f"{str(ticker).upper().strip()}_{risk.lower()}"
    current_time = time.time()
    
    if ticker_key in INDIVIDUAL_CACHE:
        payload, ts = INDIVIDUAL_CACHE[ticker_key]
        if current_time - ts < CACHE_EXPIRY_SECONDS: 
            return payload
            
    res = calculate_single_markov(str(ticker).upper().strip(), window, threshold, risk)
    if not res: 
        raise HTTPException(status_code=404, detail="Ticker data calculation failed.")
        
    INDIVIDUAL_CACHE[ticker_key] = (res, current_time)
    return res
