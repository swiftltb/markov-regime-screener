import asyncio
import time
import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

# ==========================================
# CONFIGURATION
# ==========================================
global_cache = {"data": [], "last_updated": 0}
CACHE_INTERVAL_SECONDS = 7200  
CORE_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "LLY", "JPM", "V", "UNH", "WMT",
    "RY.TO", "TD.TO", "SHOP.TO", "CP.TO", "CNR.TO"
]
# Ensure this URL matches your deployed Vercel domain exactly
BASE_DATA_URL = "https://your-vercel-app-url.vercel.app/api/data"

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
            print(f"Attempting to fetch: {url}")
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                raw_data = response.json()
                prob = run_markov_math(raw_data)
                results.append({"ticker": ticker, "regime_prob": round(prob, 4), "status": "active"})
                print(f"Successfully processed: {ticker}")
            else:
                print(f"Failed to fetch {ticker}: Status {response.status_code} - URL: {url}")
                
        except Exception as e:
            print(f"CRITICAL ERROR processing {ticker}: {str(e)}")
    
    print(f"Loop finished. Total items in cache: {len(results)}")
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
                print(f"[{time.strftime('%X')}] Cache updated with {len(fresh_data)} items.")
            else:
                print(f"[{time.strftime('%X')}] Warning: Daemon finished but results list was empty.")
        except Exception as e:
            print(f"Daemon Error: {str(e)}")
        await asyncio.sleep(CACHE_INTERVAL_SECONDS)

# ==========================================
# LIFECYCLE & ROUTES
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Wait a few seconds to let the server bind before starting intensive work
    await asyncio.sleep(5)
    asyncio.create_task(permanent_cache_worker())
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health_check():
    return {
        "status": "online", 
        "engine": "active" if global_cache["last_updated"] > 0 else "initializing",
        "cache_age": time.time() - global_cache["last_updated"] if global_cache["last_updated"] > 0 else 0
    }

@app.get("/api/screener")
async def get_screener_data(token: str):
    if token != "ecf3ac57988156c7d0dd278042861445":
        raise HTTPException(status_code=401, detail="Invalid token")
    return global_cache["data"]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
