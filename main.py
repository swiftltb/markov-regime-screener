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
    try:
        raw_df = yf.download(ticker, period="1y", interval="1d", progress=False)
        df = raw_df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns: return None
        df = df.rename(columns={'close': 'Close'})
        df = df.asfreq('B').ffill()
        returns = df['Close'].pct_change().dropna()
        model = MarkovRegression(returns, k_regimes=2, switching_variance=True).fit(disp=False)
        p1 = model.smoothed_marginal_probabilities[1].iloc[-1]
        current_regime = "Bull" if p1 > 0.5 else "Bear"
        return {
            "ticker": ticker, "current_price": float(df['Close'].iloc[-1]),
            "current_regime": current_regime, "rsi": 50,
            "trade_signal": {"action": "BUY" if current_regime == "Bull" else "HOLD", "target": "N/A"},
            "history_dates": df.tail(60).index.strftime('%Y-%m-%d').tolist(),
            "history_data": df['Close'].tail(60).tolist()
        }
    except Exception as e:
        logger.error(f"CRITICAL ERROR for {ticker}: {str(e)}")
        return None

@app.get("/api/screener")
def screener(token: str = Query(None)):
    if token != os.getenv("API_SECRET_KEY"): raise HTTPException(403)
    return [d for d in [get_markov_data(t) for t in ["SPY", "QQQ", "IWM", "TLT", "GLD"]] if d]

@app.get("/api/regime")
def regime(ticker: str, risk: str = "moderate", token: str = Query(None)):
    if token != os.getenv("API_SECRET_KEY"): raise HTTPException(403)
    data = get_markov_data(ticker)
    if not data: raise HTTPException(404, detail="Ticker not found")
    return data
