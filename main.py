from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import time
import os  
import math
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging to track failures
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Secured Markov Screener Engine")

SECRET_KEY = os.environ.get("API_SECRET_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.stockscreen.art", "https://stockscreen.art"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory caches
INDIVIDUAL_CACHE = {}
SCREENER_CACHE = {"data": None, "timestamp": 0}
CACHE_EXPIRY_SECONDS = 3600  

SCREENER_TICKERS = [
    "SPY", "QQQ", "DIA", "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", 
    "TSLA", "NFLX", "XOM", "JPM", "V", "MA", "LLY", "UNH", "COST", "TPL", "FIX",
    "XIU.TO", "CSU.TO", "TOI.TO", "LMN.TO", "SHOP.TO", "ATD.TO", "BN.TO", "CNQ.TO", 
    "CP.TO", "CNR.TO", "TD.TO", "RY.TO", "BMO.TO", "CCO.TO", "BCE.TO", "ENB.TO"
]

RISK_MULTIPLIERS = {
    "conservative": {"stop": 1.5, "target": 2.5, "buffer": 0.995},
    "moderate":     {"stop": 2.0, "target": 3.5, "buffer": 1.000},
    "aggressive":   {"stop": 3.0, "target": 5.0, "buffer": 1.005}
}

# --- DEFENSIVE CALCULATION FUNCTIONS ---

def calculate_rsi(series, period=14):
    if series.empty or len(series) < period: return np.nan
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.iloc[-1]

def calculate_atr(df, period=14):
    if len(df) < period: return np.nan
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean().iloc[-1]

def calculate_macd_signal_str(series, fast=12, slow=26, signal=9):
    if series.empty or len(series) < slow + signal: return "N/A"
    macd_line = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]: return "Bullish Crossover"
    return "Bullish Trend" if macd_line.iloc[-1] > signal_line.iloc[-1] else "Bearish Trend"

def generate_trading_signal(regime, rsi, macd_str, current_price, atr, risk_profile="moderate"):
    try:
        rsi = float(rsi) if isinstance(rsi, (int, float)) else 50.0
        atr = float(atr) if isinstance(atr, (int, float)) and atr > 0 else current_price * 0.025
        cfg = RISK_MULTIPLIERS.get(risk_profile, RISK_MULTIPLIERS["moderate"])
        
        stop_loss = round(current_price - (cfg["stop"] * atr), 2)
        target_price = round(current_price + (cfg["target"] * atr), 2)
        suggested_entry = round(current_price * cfg["buffer"], 2)

        action, color = "HOLD", "#cbd5e1"
        if regime == "Bull" and rsi < 48 and "Bullish" in macd_str: action, color = "STRONG BUY", "#22c55e"
        elif regime == "Bull" and rsi < 65: action, color = "BUY", "#4ade80"
        elif rsi > 76: action, color = "TAKE PROFIT", "#eab308"
        
        return {"action": action, "color": color, "entry": f"${suggested_entry}", "stop": f"${stop_loss}", "target": f"${target_price}"}
    except Exception:
        return {"action": "HOLD", "color": "#cbd5e1", "entry": "N/A", "stop": "N/A", "target": "N/A"}

# --- MAIN ENGINE PIPELINE ---

def calculate_single_markov(ticker, window=20, threshold=0.012, risk_profile="moderate"):
    try:
        # Sanitize and fetch
        clean_ticker = ticker.strip().upper().replace("NYSE:", "")
        df = yf.download(clean_ticker, period="1y", interval="1d", progress=False)
        
        # Defensive check
        if df is None or df.empty or len(df) < window:
            logger.warning(f"Ticker {clean_ticker} failed validation (Empty/Insufficient Data)")
            return None
            
        # Standardize
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(-1)
        df.columns = [str(col).lower() for col in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        
        df[['close', 'high', 'low']] = df[['close', 'high', 'low']].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=['close', 'high', 'low'])
        
        # Calculate
        latest_rsi = calculate_rsi(df['close'])
        latest_atr = calculate_atr(df)
        macd_desc = calculate_macd_signal_str(df['close'])
        
        # Markov Logic
        df['Log_Ret'] = np.log(df['close'] / df['close'].shift(1))
        df['State'] = df['Log_Ret'].rolling(window).sum().apply(lambda v: 'Bull' if v > threshold else ('Bear' if v < -threshold else 'Sideways'))
        
        # Return payload
        current_price = float(df['close'].iloc[-1])
        return {
            "ticker": clean_ticker,
            "current_regime": str(df['State'].iloc[-1]),
            "current_price": current_price,
            "rsi": round(float(latest_rsi), 2) if not pd.isna(latest_rsi) else "N/A",
            "trade_signal": generate_trading_signal(str(df['State'].iloc[-1]), latest_rsi, macd_desc, current_price, latest_atr, risk_profile)
        }
    except Exception as e:
        logger.error(f"Pipeline failure for {ticker}: {e}")
        return None

@app.get("/api/screener")
def get_top_screener(token: str = Query(None)):
    if not SECRET_KEY or token != SECRET_KEY: raise HTTPException(status_code=403)
    # ... (Keep existing cache logic here) ...
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for res in executor.map(lambda t: calculate_single_markov(t), SCREENER_TICKERS):
            if res: results.append(res)
    return results

@app.get("/api/regime")
def get_individual_regime(ticker: str, token: str = Query(None)):
    if not SECRET_KEY or token != SECRET_KEY: raise HTTPException(status_code=403)
    res = calculate_single_markov(ticker)
    if not res: raise HTTPException(status_code=404, detail="Ticker data unavailable.")
    return res
