import logging
import os
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# 1. Setup Logging & App
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Markov Regime Screener API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("API_SECRET_KEY", "your_fallback_dev_key")

# 2. Standardize Ticker Data
def standardize_ticker_data(df, ticker):
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().strip() for c in df.columns]
    mapping = {'adj close': 'close', 'close': 'close', 'high': 'high', 'low': 'low', 'open': 'open'}
    df = df.rename(columns=mapping)
    if 'close' not in df.columns: return None
    df = df.ffill().bfill()
    return df[['close', 'high', 'low', 'open']]

# 3. Indicators
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1: return 50.0
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return (100 - (100 / (1 + rs))).iloc[-1]

def generate_trading_signal(regime, rsi, price, p_bull, risk_profile="moderate"):
    signal = {"action": "HOLD", "entry": f"${price:.2f}", "stop": "N/A", "target": "N/A"}
    m = {"conservative": {"stop": 0.03, "target": 0.06}, "moderate": {"stop": 0.05, "target": 0.10}, "aggressive": {"stop": 0.08, "target": 0.18}}.get(risk_profile)
    signal["stop"] = f"${(price * (1 - m['stop'])):.2f}"
    signal["target"] = f"${(price * (1 + m['target'])):.2f}"
    if regime == "Bull" and rsi < 70 and p_bull > 0.60: signal["action"] = "BUY"
    elif rsi > 75: signal["action"] = "EXHAUSTED"
    elif regime == "Bear": signal["action"] = "BEAR_HOLD"
    return signal

# 4. Core Markov Logic (FIXED FOR FREQUENCY)
def calculate_single_markov(ticker, window=20, risk="moderate"):
    clean_ticker = ticker.strip().upper().replace(".US", "").replace("$", "")
    try:
        raw_df = yf.download(clean_ticker, period="1y", interval="1d", progress=False, multi_level_index=False)
        df = standardize_ticker_data(raw_df, clean_ticker)
        if df is None or len(df) < window: return None
        
        # --- FIX: Ensure Frequency for statsmodels ---
        df.index = pd.to_datetime(df.index)
        df = df.asfreq('B')  # Business day frequency
        df = df.ffill()      # Fill gaps
        
        close_series = df['close']
        returns = close_series.pct_change().dropna()
        
        model = MarkovRegression(returns, k_regimes=2, switching_variance=True)
        res = model.fit(disp=False)
        
        smoothed_probs = res.smoothed_marginal_probabilities
        current_regime = "Bull" if smoothed_probs[1].iloc[-1] > smoothed_probs[0].iloc[-1] else "Bear"
        p_bull_bull = float(res.regime_transition_matrix[1, 1])
        
        latest_price = float(close_series.iloc[-1])
        latest_rsi = calculate_rsi(close_series)
        trade_signal = generate_trading_signal(current_regime, latest_rsi, latest_price, p_bull_bull, risk)
        
        history_df = df[['close']].tail(60).reset_index()
        history_df.columns = ['date', 'close']
        
        return {
            "ticker": clean_ticker,
            "current_price": latest_price,
            "current_regime": current_regime,
            "p_bull_bull": p_bull_bull,
            "rsi": round(latest_rsi, 1),
            "trade_signal": trade_signal,
            "history_dates": history_df['date'].dt.strftime('%Y-%m-%d').tolist(),
            "history_data": history_df['close'].tolist()
        }
    except Exception as e:
        logger.error(f"Error {clean_ticker}: {e}")
        return None

# 5. API Endpoints
@app.get("/api/screener")
def get_screener_matrix(token: str = Query(None)):
    if token != SECRET_KEY: raise HTTPException(status_code=403, detail="Unauthorized")
    return [d for d in [calculate_single_markov(t) for t in ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD"]] if d]

@app.get("/api/regime")
def get_individual_regime(ticker: str, token: str = Query(None)):
    if token != SECRET_KEY: raise HTTPException(status_code=403, detail="Unauthorized")
    data = calculate_single_markov(ticker)
    if not data: raise HTTPException(status_code=404, detail="Processing failed")
    return data
