import logging, os, random
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of User-Agents to rotate
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
]

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_markov_data(ticker):
    clean_ticker = str(ticker).strip().upper()
    try:
        # Rotate user agent for every request
        ua = random.choice(USER_AGENTS)
        raw_df = yf.download(clean_ticker, period="1y", interval="1d", progress=False, headers={'User-Agent': ua})
        
        if raw_df is None or raw_df.empty or 'Close' not in raw_df.columns:
            logger.warning(f"Yahoo returned empty data for {clean_ticker}")
            return None
        
        df = raw_df[['Close']].copy()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={'close': 'Close'}).asfreq('B').ffill().dropna()
        
        returns = df['Close'].pct_change().dropna()
        model = MarkovRegression(returns, k_regimes=2, switching_variance=True).fit(disp=False)
        p1 = model.smoothed_marginal_probabilities[1].iloc[-1]
        regime = "Bull" if p1 > 0.5 else "Bear"
        
        return {
            "ticker": clean_ticker, "current_price": float(df['Close'].iloc[-1]),
            "current_regime": regime, "rsi": 50,
            "trade_signal": {"action": "BUY" if regime == "Bull" else "HOLD", "target": "N/A"},
            "history_dates": df.tail(60).index.strftime('%Y-%m-%d').tolist(),
            "history_data": df['Close'].tail(60).tolist()
        }
    except Exception as e:
        logger.error(f"Critical error for {clean_ticker}: {str(e)}")
        return None

@app.get("/api/screener")
def screener(token: str = Query(None)):
    if token != os.getenv("API_SECRET_KEY"): raise HTTPException(403)
    results = []
    # Test with just one to see if it unblocks
    for t in ["SPY", "QQQ"]:
        data = get_markov_data(t)
        if data: results.append(data)
    return results

@app.get("/api/regime")
def regime(ticker: str, token: str = Query(None)):
    if token != os.getenv("API_SECRET_KEY"): raise HTTPException(403)
    data = get_markov_data(ticker)
    if not data: raise HTTPException(404, detail="Ticker not found")
    return data
