import logging, os
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory store
cache = {}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_markov_data(ticker):
    clean_ticker = str(ticker).strip().upper()
    if clean_ticker in cache: return cache[clean_ticker]
    
    try:
        # Use a single fetch attempt
        df = yf.download(clean_ticker, period="1y", interval="1d", progress=False)
        
        # Validate data
        if df is None or df.empty or 'Close' not in df.columns:
            logger.error(f"Invalid data for {clean_ticker}")
            return None
        
        # Process and flatten
        close_data = df['Close'].copy()
        if isinstance(close_data, pd.DataFrame):
            close_data = close_data.iloc[:, 0]
            
        data_clean = close_data.asfreq('B').ffill().dropna()
        returns = data_clean.pct_change().dropna()
        
        # Fit model
        model = MarkovRegression(returns, k_regimes=2, switching_variance=True).fit(disp=False)
        p1 = model.smoothed_marginal_probabilities[1].iloc[-1]
        
        regime = "Bull" if p1 > 0.5 else "Bear"
        
        data = {
            "ticker": clean_ticker, 
            "current_price": float(data_clean.iloc[-1]),
            "current_regime": regime, 
            "rsi": 50,
            "trade_signal": {"action": "BUY" if regime == "Bull" else "HOLD", "target": "N/A"},
            "history_dates": data_clean.tail(60).index.strftime('%Y-%m-%d').tolist(),
            "history_data": data_clean.tail(60).tolist()
        }
        cache[clean_ticker] = data
        return data
    except Exception as e:
        logger.error(f"Model failure for {clean_ticker}: {str(e)}")
        return None

@app.get("/api/screener")
def screener(token: str = Query(None)):
    if token != os.getenv("API_SECRET_KEY"): raise HTTPException(403)
    return [d for d in [get_markov_data(t) for t in ["SPY", "QQQ", "IWM", "TLT", "GLD"]] if d]

@app.get("/api/regime")
def regime(ticker: str, token: str = Query(None)):
    if token != os.getenv("API_SECRET_KEY"): raise HTTPException(403)
    data = get_markov_data(ticker)
    if not data: raise HTTPException(404, detail="Ticker not found")
    return data
