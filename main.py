from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import time
import os  
import math
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Secured Markov Screener Engine")

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
# NEW: TECHNICAL INDICATOR CALCULATION ENGINE
# ==========================================

def calculate_rsi(prices_series, period=14):
    """Calculates RSI using raw pandas series input."""
    if len(prices_series) < period:
        return np.nan
    delta = prices_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.iloc[-1]

def calculate_atr(df, period=14):
    """Calculates ATR matching common charting platforms."""
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

def calculate_macd_signal_str(prices_series, fast=12, slow=26, signal=9):
    """Calculates MACD trends and notes structural directional crossings."""
    if len(prices_series) < slow + signal:
        return "N/A"
    fast_ema = prices_series.ewm(span=fast, adjust=False).mean()
    slow_ema = prices_series.ewm(span=slow, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Check current and prior cross states
    c_macd, c_sig = macd_line.iloc[-1], signal_line.iloc[-1]
    p_macd, p_sig = macd_line.iloc[-2], signal_line.iloc[-2]
    
    if c_macd > c_sig and p_macd <= p_sig:
        return "Bullish Crossover"
    elif c_macd < c_sig and p_macd >= p_sig:
        return "Bearish Crossover"
    else:
        return "Bullish Trend" if c_macd > c_sig else "Bearish Trend"

# ==========================================
# CORE MARKOV REGIME MATHEMATICS PIPELINE
# ==========================================

def calculate_single_markov(ticker, window=20, threshold=0.012):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty or len(df) < window: return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(col).lower() for col in df.columns]
        
        # Enforce pure float conversions on primary Series to safely handle math computations
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        
        # Calculate Technical Indicators
        latest_rsi = calculate_rsi(df['close'])
        latest_atr = calculate_atr(df)
        macd_desc = calculate_macd_signal_str(df['close'])
        
        # Existing Markov Array Calculations
        df['Log_Ret'] = np.log(df['close'] / df['close'].shift(1))
        df['Roll_Mom'] = df['Log_Ret'].rolling(window).sum()
        
        def classify(val):
            if pd.isna(val): return 'Sideways'
            if val > threshold: return 'Bull'
            elif val < -threshold: return 'Bear'
            return 'Sideways'
            
        df['State'] = df['Roll_Mom'].apply(classify)
        df['Next_State'] = df['State'].shift(-1)
        
        states = ['Bull', 'Sideways', 'Bear']
        matrix = {s: {next_s: 0.0 for next_s in states} for s in states}
        
        for _, row in df.dropna(subset=['State', 'Next_State']).iterrows():
            matrix[row['State']][row['Next_State']] += 1.0
            
        for s in states:
            total = sum(matrix[s].values())
            if total > 0:
                for next_s in states:
                    matrix[s][next_s] = round(matrix[s][next_s] / total, 3)
            else:
                matrix[s][s] = 1.0
                
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0.0
        year_high = info.get('fiftyTwoWeekHigh') or 0.0
        year_low = info.get('fiftyTwoWeekLow') or 0.0
        target_mean = info.get('targetMeanPrice') or current_price
        regime_score = round(((target_mean / current_price) - 1) * 100, 2) if current_price > 0 else 0.0
        
        current_state = df['State'].values[-1] if hasattr(df['State'], 'values') else df['State'].iloc[-1]
        trailing_return = float((df['close'].values[-1] / df['close'].values[0]) - 1)
        
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
            
            # Formatted Indicator Sub-payload
            "rsi": round(float(latest_rsi), 2) if not pd.isna(latest_rsi) else "N/A",
            "atr": round(float(latest_atr), 2) if not pd.isna(latest_atr) else "N/A",
            "macd_signal": macd_desc
        }
    except Exception as e:
        print(f"Error computing math matrix for {ticker}: {str(e)}")
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
        computed = executor.map(calculate_single_markov, SCREENER_TICKERS)
        for res in computed:
            if res: results.append(res)
                
    sorted_results = sorted(results, key=lambda x: (-x['p_bull_bull'], -x['trailing_return']))
    top_25 = sorted_results[:25]
    
    SCREENER_CACHE["data"] = top_25
    SCREENER_CACHE["timestamp"] = current_time
    return top_25

@app.get("/api/regime", dependencies=[Depends(verify_token)])
def get_individual_regime(ticker: str, window: int = 20, threshold: float = 0.012):
    ticker_key = str(ticker).upper().strip()
    current_time = time.time()
    
    if ticker_key in INDIVIDUAL_CACHE:
        payload, ts = INDIVIDUAL_CACHE[ticker_key]
        if current_time - ts < CACHE_EXPIRY_SECONDS: 
            return payload
            
    res = calculate_single_markov(ticker_key, window, threshold)
    if not res: 
        raise HTTPException(status_code=404, detail=f"Ticker {ticker_key} data generation failed.")
        
    INDIVIDUAL_CACHE[ticker_key] = (res, current_time)
    return res
