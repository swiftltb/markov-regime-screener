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

# 2. Forgiving Gatekeeper Configuration
def standardize_ticker_data(df, ticker):
    if df is None or df.empty:
        return None

    # MultiIndex Safety Guard: Grab Level 0 if yfinance packages it with multi-levels
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Normalize column text to lowercase strings
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    # Loose mapping configuration
    mapping = {
        'adj close': 'close', 
        'close': 'close', 
        'high': 'high', 
        'low': 'low', 
        'open': 'open'
    }
    df = df.rename(columns=mapping)
    
    # Absolute minimum requirement survival check
    if 'close' not in df.columns:
        logger.warning(f"Ticker {ticker} rejected: missing 'close' price column. Found columns: {list(df.columns)}")
        return None
        
    # Fallback assignment to prevent structural crashes
    if 'high' not in df.columns: df['high'] = df['close']
    if 'low' not in df.columns: df['low'] = df['close']
    if 'open' not in df.columns: df['open'] = df['close']
        
    # Seamless gap filling
    df = df.ffill().bfill()
    return df[['close', 'high', 'low', 'open']]

# 3. Indicator & Strategy Matrix Calculations
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def generate_trading_signal(regime, rsi, price, p_bull, risk_profile):
    signal = {"action": "HOLD", "entry": f"${price:.2f}", "stop": "N/A", "target": "N/A"}
    
    multipliers = {
        "conservative": {"stop": 0.03, "target": 0.06},
        "moderate": {"stop": 0.05, "target": 0.10},
        "aggressive": {"stop": 0.08, "target": 0.18}
    }
    m = multipliers.get(risk_profile, multipliers["moderate"])

    # ALWAYS POPULATE THRESHOLDS regardless of regime context
    signal["stop"] = f"${(price * (1 - m['stop'])):.2f}"
    signal["target"] = f"${(price * (1 + m['target'])):.2f}"

    # Determine explicit Action recommendation tags
    if regime == "Bull" and rsi < 70 and p_bull > 0.60:
        signal["action"] = "BUY"
    elif rsi > 75:
        signal["action"] = "EXHAUSTED"
    elif regime == "Bear":
        signal["action"] = "BEAR_HOLD"
        
    return signal

# 4. Core Markov Logic Circuit
def calculate_single_markov(ticker, window=20, threshold=0.012, risk="moderate"):
    clean_ticker = ticker.strip().upper()
    clean_ticker = clean_ticker.replace(".US", "").replace("NYSE:", "").replace("NASDAQ:", "").replace("$", "")
    
    try:
        # FIXED: Explicitly tell yfinance to return flat tables instead of multi-level indices
        raw_df = yf.download(clean_ticker, period="1y", interval="1d", progress=False, multi_level_index=False)
        
        df = standardize_ticker_data(raw_df, clean_ticker)
        if df is None or len(df) < window:
            return None
            
        close_series = df['close'].astype(float)
        returns = close_series.pct_change().dropna()
        
        if len(returns) < window:
            return None

        # Execute Markov Switching Regression
        model = MarkovRegression(returns, k_regimes=2, switching_variance=True)
        res = model.fit(disp=False)
        
        smoothed_probs = res.smoothed_marginal_probabilities
        current_regime_index = np.argmax([smoothed_probs[0].iloc[-1], smoothed_probs[1].iloc[-1]])
        
        regime_labels = {0: "Bear", 1: "Bull"}
        current_regime = regime_labels.get(current_regime_index, "Unknown")
        
        p_bull_bull = float(res.regime_transition_matrix[1, 1]) if hasattr(res, 'regime_transition_matrix') else 0.50
        
        latest_price = float(close_series.iloc[-1])
        trailing_return = float((close_series.iloc[-1] / close_series.iloc[-window] - 1) * 100)
        latest_rsi = calculate_rsi(close_series)
        
        trade_signal = generate_trading_signal(current_regime, latest_rsi, latest_price, p_bull_bull, risk)
        
        history_df = df[['close']].tail(60).reset_index()
        history_df.columns = ['date', 'close']
        history_df['date'] = history_df['date'].dt.strftime('%Y-%m-%d')
        history_list = history_df.values.tolist()

        return {
            "ticker": clean_ticker,
            "current_price": latest_price,
            "current_regime": current_regime,
            "p_bull_bull": p_bull_bull,
            "trailing_return": round(trailing_return, 2),
            "rsi": round(latest_rsi, 1),
            "trade_signal": trade_signal,
            "history": history_list
        }
        
    except Exception as e:
        logger.error(f"Error compiling Markov execution metrics for {clean_ticker}: {e}")
        return None

# 5. API Core Endpoint Directives
@app.get("/api/screener")
def get_screener_matrix(token: str = Query(None)):
    if not SECRET_KEY or token != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized API Token Access")
        
    default_universe = ["SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "BTC-USD", "LEU"]
    compiled_results = []
    
    for ticker in default_universe:
        data = calculate_single_markov(ticker)
        if data:
            compiled_results.append(data)
            
    return compiled_results

@app.get("/api/regime")
def get_individual_regime(ticker: str, token: str = Query(None)):
    if not SECRET_KEY or token != SECRET_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized API Token Access")
        
    if not ticker:
        raise HTTPException(status_code=400, detail="Missing required ticker parameter query")
        
    data = calculate_single_markov(ticker)
    if not data:
        raise HTTPException(status_code=404, detail="Ticker processing failed or symbol is missing")
        
    return data
