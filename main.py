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
    """
    Optimized math engine using a 60-day window to reduce 
    memory footprint and prevent SIGABRT (Code 134) crashes.
    """
    # 1. Configurable window size
    WINDOW_SIZE = 60 
    
    # 2. Slice and clean data
    # Ensure we have enough data to form a valid regime
    data_subset = df['Close'].pct_change().tail(WINDOW_SIZE).dropna().astype('float32')
    
    if len(data_subset) < 40:
        return {"regime": "Neutral", "persistence": 0.0, "rsi": 0.0, "macd": 0.0}
    
    # 3. Clear memory before allocation
    gc.collect()
    
    try:
        # 4. Use memory-efficient fitting
        model = MarkovAutoregression(data_subset, k_regimes=2, order=1, switching_variance=True)
        res = model.fit(disp=False, maxiter=100, method='nm')
        
        # 5. Extract state and persistence
        state = res.smoothed_marginal_probabilities.iloc[-1].idxmax()
        # Ensure we access the transition matrix safely
        transition_matrix = res.regime_transition_matrix
        persistence = transition_matrix[state, state]
        
        # 6. Cleanup
        del model
        del res
        gc.collect()
        
        return {
            "regime": "Bullish" if state == 1 else "Bearish",
            "persistence": round(float(persistence), 2),
            "rsi": 0.0,
            "macd": 0.0
        }
        
    except Exception as e:
        print(f"MATH ENGINE ERROR: {str(e)}")
        gc.collect()
        return {"regime": "Error", "persistence": 0.0, "rsi": 0.0, "macd": 0.0}
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
