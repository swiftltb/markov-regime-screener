import gc
import pandas as pd
import numpy as np
import httpx
from fastapi import FastAPI
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

app = FastAPI()

# Configuration
DREAMHOST_API = "https://stockscreen.art/update_cache.php"
WORKER_URL = "https://raspy-recipe-da41.arthur-barabash.workers.dev/"
TICKERS = ["AAPL", "NVDA", "MSFT", "TSLA", "AMD", "GOOGL", "AMZN", "META", "NFLX", "AVGO", "INTC", "CSCO", "PEP", "ADBE", "COST"]
HEADERS = {"X-Secret-Key": "k7P9vR2WxM4zLqN1jB5vH8cF3tD6yS9a"}

def run_math(df):
    """Memory-efficient math logic with robust statsmodels attribute access."""
    # RSI & MACD logic
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2

    # Markov logic
    returns = df['Close'].pct_change().dropna().tail(500)
    model = MarkovAutoregression(returns, k_regimes=2, order=1)
    res = model.fit(disp=False)
    
    state = res.smoothed_marginal_probabilities.iloc[-1].idxmax()
    
    # Robust attribute access
    if hasattr(res, 'regimes'):
        matrix = res.regimes.transition_matrix
    else:
        matrix = res.regime_transition_matrix
        
    persistence = matrix[state, state]
    
    return {
        "regime": "Bullish" if state == 1 else "Bearish",
        "persistence": round(float(persistence), 2),
        "rsi": round(float(rsi.iloc[-1]), 2),
        "macd": round(float(macd.iloc[-1]), 2)
    }

async def process_ticker(ticker, client):
    try:
        # 1. Fetch data
        resp = await client.get(f"{WORKER_URL}?ticker={ticker}", timeout=30.0)
        data = resp.json()
        
        # 2. Convert and inspect data immediately in the logs
        df = pd.DataFrame({'Close': data['closes']})
        returns = df['Close'].pct_change().dropna()
        
        print(f"DIAGNOSTIC: Ticker {ticker}")
        print(f"DIAGNOSTIC: Data rows: {len(returns)}")
        print(f"DIAGNOSTIC: Any nulls? {returns.isnull().any()}")
        print(f"DIAGNOSTIC: Any infinities? {np.isinf(returns).any()}")

        # 3. Attempt math and report the specific error type
        try:
            model = MarkovAutoregression(returns, k_regimes=2, order=1)
            res = model.fit(disp=False)
            print(f"MATH SUCCESS: {ticker}")
        except Exception as e:
            print(f"MATH FAILURE: {ticker} | Error Type: {type(e).__name__} | Details: {str(e)}")
            raise # This ensures the stack trace shows up in your Render dashboard

    except Exception as e:
        print(f"PIPELINE CRASH on {ticker}: {str(e)}")

@app.get("/run-refresh")
async def run_full_refresh():
    async with httpx.AsyncClient() as client:
        for ticker in TICKERS:
            await process_ticker(ticker, client)
    return {"status": "Complete"}
