import logging, os, json
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cache = {}
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_markov_data(ticker):
    clean_ticker = str(ticker).strip().upper()
    if clean_ticker in cache: return cache[clean_ticker]
    try:
        raw_df = yf.download(clean_ticker, period="1y", interval="1d", progress=False)
        if raw_df is None or raw_df.empty or 'Close' not in raw_df.columns:
            return None
        
        df = raw_df[['Close']].copy()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={'close': 'Close'}).asfreq('B').ffill().dropna()
        
        returns = df['Close'].pct_change().dropna()
        model = MarkovRegression(returns, k_regimes=2, switching_variance=True).fit(disp=False)
        p1 = model.smoothed_marginal_probabilities[1].iloc[-1]
        regime = "Bull" if p1 > 0.5 else "Bear"
        
        data = {
            "ticker": clean_ticker, 
            "current_price": float(df['Close'].iloc[-1]),
            "current_regime": regime, 
            "rsi": 50,
            "trade_signal": {"action": "BUY" if regime == "Bull" else "HOLD", "target": "N/A"},
            "history_dates": df.tail(60).index.strftime('%Y-%m-%d').tolist(),
            "history_data": df['Close'].tail(60).tolist()
        }
        cache[clean_ticker] = data
        return data
    except Exception as e:
        logger.error(f"Critical error for {clean_ticker}: {str(e)}")
        return None

@app.get("/api/screener")
def screener(token: str = Query(None)):
    if token != os.getenv("API_SECRET_KEY"): raise HTTPException(403)
    results = []
    for t in ["SPY", "QQQ", "IWM", "TLT", "GLD"]:
        data = get_markov_data(t)
        if data: results.append(data)
    
    # Log what we are actually sending
    logger.info(f"Screener returning {len(results)} items")
    return results

@app.get("/api/regime")
def regime(ticker: str, token: str = Query(None)):
    if token != os.getenv("API_SECRET_KEY"): raise HTTPException(403)
    data = get_markov_data(ticker)
    if not data: raise HTTPException(404, detail="Ticker not found")
    return data
