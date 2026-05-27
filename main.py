import asyncio
import time
import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

# ==========================================
# CONFIGURATION & UNIVERSAL STATE
# ==========================================
global_cache = {"data": [], "last_updated": 0}
CACHE_INTERVAL_SECONDS = 7200  
CORE_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "LLY", "JPM", "V", "UNH", "WMT",
    "RY.TO", "TD.TO", "SHOP.TO", "CP.TO", "CNR.TO"
]
# This points to the Vercel-deployed instance
BASE_DATA_URL = "https://markov-screener-proxy.vercel.app/api/data"

# ==========================================
# MARKOV ENGINE
# ==========================================
def run_markov_math(data_list):
    df = pd.Series(data_list)
    returns = np.log(df / df.shift(1)).dropna()
    model = sm.tsa.MarkovAutoregression(returns, k_regimes=2, order=1, switching_variance=True)
    res = model.fit(disp=False)
    return float(res.smoothed_marginal_probabilities[1].iloc[-1])

def heavy_matrix_calculations():
    print(f"[{time.strftime('%X')}] Background Daemon: Initiating 15-stock Markov regressions...")
    results = []
    
    for ticker in CORE_UNIVERSE:
        url = f"{BASE_DATA_URL}/{ticker}"
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                raw_data = response.json()
                prob = run_markov_math(raw_data)
                results.append({"ticker": ticker, "regime_prob": round(prob, 4), "status": "active"})
                print(f"Successfully processed: {ticker}")
            else:
                print(f"Failed to fetch {ticker}: Status {response.status_code}")
        except Exception as e:
            print(f"CRITICAL ERROR processing {ticker}: {str(e)}")
    
    return results

# ==========================================
# BACKGROUND WORKER
# ==========================================
async def permanent_cache_worker():
    global global_cache
    while True:
        try:
            fresh_data = heavy_matrix_calculations()
            if fresh_data:
                global_cache["data"] = fresh_data
                global_cache["last_updated"] = time.time()
                print(f"[{time.strftime('%X')}] Cache updated.")
        except Exception as e:
            print(f"Daemon Error: {str(e)}")
        await asyncio.sleep(CACHE_INTERVAL_SECONDS)

# ==========================================
# FASTAPI LIFECYCLE & ROUTES
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(permanent_cache_worker())
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# PROXY ROUTE: Handles the fetch logic for both local and Vercel-deployed environments
@app.get("/api/data/{ticker}")
async def get_ticker_data(ticker: str):
    try:
        df = yf.download(ticker, period="1y", interval="1d")
        if df.empty:
            raise HTTPException(status_code=404, detail="Ticker not found")
        # Return only the closing prices as a list
        return df['Close'].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "online", "cache_age": time.time() - global_cache["last_updated"]}

@app.get("/api/screener")
async def get_screener_data(token: str):
    if token != "ecf3ac57988156c7d0dd278042861445":
        raise HTTPException(status_code=401)
    return global_cache["data"]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
