from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Secured Markov Screener Engine")

# --- CORS & SECURITY ---
SECRET_KEY = os.environ.get("API_SECRET_KEY")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.stockscreen.art", "https://stockscreen.art"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIG ---
SCREENER_TICKERS = ["SPY", "QQQ", "DIA", "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", "TSLA", "NFLX", "XOM", "JPM", "V", "MA", "LLY", "UNH", "COST", "TPL", "FIX", "XIU.TO", "CSU.TO", "TOI.TO", "LMN.TO", "SHOP.TO", "ATD.TO", "BN.TO", "CNQ.TO", "CP.TO", "CNR.TO", "TD.TO", "RY.TO", "BMO.TO", "CCO.TO", "BCE.TO", "ENB.TO"]
RISK_MULTIPLIERS = {"conservative": {"stop": 1.5, "target": 2.5, "buffer": 0.995}, "moderate": {"stop": 2.0, "target": 3.5, "buffer": 1.000}, "aggressive": {"stop": 3.0, "target": 5.0, "buffer": 1.005}}

# --- 1. THE GATEKEEPER ---
def standardize_ticker_data(df, ticker):
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(-1)
    df.columns = [str(c).lower().strip() for c in df.columns]
    mapping = {'adj close': 'close', 'close': 'close', 'high': 'high', 'low': 'low'}
    df = df.rename(columns=mapping)
    if not all(col in df.columns for col in ['close', 'high', 'low']): return None
    df = df[['close', 'high', 'low']].apply(pd.to_numeric, errors='coerce').dropna()
    return df

# --- 2. TECHNICAL INDICATOR CALCULATIONS ---
def calculate_rsi(series, period=14):
    if series.empty or len(series) < period: return np.nan
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period).mean()
    avg_loss = loss.ewm(alpha=1/period).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).iloc[-1]

def calculate_atr(df, period=14):
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period).mean().iloc[-1]

def calculate_macd_signal_str(series, fast=12, slow=26, signal=9):
    macd = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    if macd.iloc[-1] > sig.iloc[-1] and macd.iloc[-2] <= sig.iloc[-2]: return "Bullish Crossover"
    return "Bullish Trend" if macd.iloc[-1] > sig.iloc[-1] else "Bearish Trend"

def generate_trading_signal(regime, rsi, macd_str, current_price, atr, risk="moderate"):
    cfg = RISK_MULTIPLIERS.get(risk, RISK_MULTIPLIERS["moderate"])
    stop = round(current_price - (cfg["stop"] * atr), 2)
    target = round(current_price + (cfg["target"] * atr), 2)
    entry = round(current_price * cfg["buffer"], 2)
    action, color = "HOLD", "#cbd5e1"
    if regime == "Bull" and rsi < 48 and "Bullish" in macd_str: action, color = "STRONG BUY", "#22c55e"
    return {"action": action, "color": color, "entry": f"${entry}", "stop": f"${stop}", "target": f"${target}"}

# --- 3. CORE PIPELINE ---
def calculate_single_markov(ticker, window=20, threshold=0.012, risk="moderate"):
    try:
        clean_ticker = ticker.strip().upper().replace("NYSE:", "").replace("$", "")
        raw_df = yf.download(clean_ticker, period="1y", interval="1d", progress=False)
        df = standardize_ticker_data(raw_df, clean_ticker)
        if df is None or len(df) < window: return None
        
        latest_rsi = calculate_rsi(df['close'])
        latest_atr = calculate_atr(df)
        macd = calculate_macd_signal_str(df['close'])
        
        df['State'] = np.log(df['close'] / df['close'].shift(1)).rolling(window).sum().apply(lambda v: 'Bull' if v > threshold else ('Bear' if v < -threshold else 'Sideways'))
        
        return {
            "ticker": clean_ticker,
            "current_regime": str(df['State'].iloc[-1]),
            "current_price": float(df['close'].iloc[-1]),
            "rsi": round(float(latest_rsi), 2),
            "trade_signal": generate_trading_signal(str(df['State'].iloc[-1]), latest_rsi, macd, float(df['close'].iloc[-1]), latest_atr, risk)
        }
    except Exception as e:
        logger.error(f"Pipeline failure for {ticker}: {e}")
        return None

# --- 4. ENDPOINTS ---
@app.get("/api/screener")
def get_top_screener(token: str = Query(None)):
    if not SECRET_KEY or token != SECRET_KEY: raise HTTPException(status_code=403)
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = [res for res in executor.map(lambda t: calculate_single_markov(t), SCREENER_TICKERS) if res]
    return results

@app.get("/api/regime")
def get_regime(ticker: str, token: str = Query(None)):
    if not SECRET_KEY or token != SECRET_KEY: raise HTTPException(status_code=403)
    res = calculate_single_markov(ticker)
    if not res: raise HTTPException(status_code=404, detail="Data unavailable")
    return res
