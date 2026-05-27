import logging, os
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_markov_data(ticker):
    clean_ticker = str(ticker).strip().upper()
    try:
        # Simplest possible fetch
        df = yf.download(clean_ticker, period="1y", interval="1d", progress=False)
        if df is None or df.empty or 'Close' not in df.columns:
            return None
        
        # Ensure flat columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        data = df[['Close']].copy().ffill().dropna()
        returns = data['Close'].pct_change().dropna()
        
        model = MarkovRegression(returns, k_regimes=2, switching_variance=True).fit(disp=False)
        p1 = model.smoothed_marginal_probabilities[1].iloc[-1]
        regime = "Bull" if p1 > 0.5 else "Bear"
        
        return {
            "ticker": clean_ticker, 
            "current_price": float(data['Close'].iloc[-1]),
            "current_regime": regime, 
            "rsi": 50,
            "trade_signal": {"action": "BUY" if regime == "Bull" else "HOLD", "target": "N/A"},
            "history_dates": data.tail(60).index.strftime('%Y-%m-%d').tolist(),
            "history_data": data['Close'].tail(60).tolist()
        }
    except Exception as e:
        logger.error(f"Error for {clean_ticker}: {str(e)}")
        return None

@app.get("/api/screener")
def screener(token: str = Query(None)):
    if token != os.getenv("API_SECRET_KEY"): raise HTTPException(403)
    # Testing with just two to confirm data flow
    return [d for d in [get_markov_data("SPY"), get_markov_data("QQQ")] if d]

@app.get("/api/regime")
def regime(ticker: str, token: str = Query(None)):
    if token != os.getenv("API_SECRET_KEY"): raise HTTPException(403)
    data = get_markov_data(ticker)
    if not data: raise HTTPException(404, detail="Ticker not found")
    return data
