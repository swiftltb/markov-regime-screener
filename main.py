from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import requests
import functools
import uvicorn
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

app = FastAPI()

# --- HARDENED CORS: Always attached, even on crash ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_cors_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

# --- ROBUST ANALYSIS ENGINE ---
def run_markov_analysis(ticker):
    try:
        # Fetching with 45s hard timeout
        worker_url = f"https://raspy-recipe-da41.arthur-barabash.workers.dev/?ticker={ticker}"
        response = requests.get(worker_url, timeout=45)
        response.raise_for_status()
        data = response.json()

        # Data Validation
        if not data.get('closes') or len(data['closes']) < 30:
            return {"ticker": ticker, "price": 0, "rsi": 0, "macd": 0, "regime": "Low Data"}

        df = pd.DataFrame({'Close': data['closes']})
        # Calculate Returns: Ensure no NaNs or Infs
        returns = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        
        # Guard: Check for degenerate data (flatlines)
        if returns.std() < 1e-9:
            regime = "Neutral"
        else:
            # Model Fitting: Explicit scope for convergence
            model = MarkovAutoregression(returns, k_regimes=2, order=1, trend='c')
            res = model.fit(disp=False, method='nm', maxiter=2000)
            regime = "Bull" if res.filtered_marginal_probabilities[0].iloc[-1] > 0.5 else "Bear"

        # Indicators
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1]
        macd = (df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()).iloc[-1]

        return {
            "ticker": ticker,
            "price": float(df['Close'].iloc[-1]),
            "rsi": round(float(rsi), 2),
            "macd": round(float(macd), 2),
            "regime": regime
        }
    except Exception as e:
        return {"ticker": ticker, "price": 0, "rsi": 0, "macd": 0, "regime": "Error"}

# --- CACHE ---
@functools.lru_cache(maxsize=1)
def get_cached_data():
    tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "AMD", "GOOGL", "AMZN", "META", "NFLX", "INTC", "CSCO", "PEP", "ADBE", "QCOM", "AVGO"]
    return [run_markov_analysis(t) for t in tickers]

# --- ENDPOINTS ---
@app.get("/screener-data")
async def get_screener_data():
    return {"status": "success", "data": get_cached_data()}

@app.get("/api/health")
async def health():
    return {"status": "online"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
