import gc
import pandas as pd
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
    """Memory-efficient math logic."""
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2

    # Markov
    returns = df['Close'].pct_change().dropna().tail(500)
    model = MarkovAutoregression(returns, k_regimes=2, order=1)
    res = model.fit(disp=False)
    
    state = res.smoothed_marginal_probabilities.iloc[-1].idxmax()
    persistence = res.regime_transition_matrix[state, state]
    
    return {
        "regime": "Bullish" if state == 1 else "Bearish",
        "persistence": round(float(persistence), 2),
        "rsi": round(float(rsi.iloc[-1]), 2),
        "macd": round(float(macd.iloc[-1]), 2)
    }

async def process_ticker(ticker, client):
    """Processes ONE ticker, clears memory, returns."""
    try:
        resp = await client.get(f"{WORKER_URL}?ticker={ticker}", timeout=30.0)
        data = resp.json()
        df = pd.DataFrame({'Close': data['closes']}).astype('float32')
        
        result = run_math(df)
        
        # Send to DreamHost
        await client.post(DREAMHOST_API, json={"ticker": ticker, "data": result}, headers=HEADERS)
        
        # Clear Memory
        del df, data, result
        gc.collect()
    except Exception as e:
        print(f"Error {ticker}: {e}")

@app.get("/run-refresh")
async def run_full_refresh():
    async with httpx.AsyncClient() as client:
        for ticker in TICKERS:
            await process_ticker(ticker, client)
    return {"status": "Complete"}
