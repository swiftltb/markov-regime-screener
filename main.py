import asyncio
import time
import requests
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

# ==========================================
# GLOBAL STATE & CACHE CONFIGURATION
# ==========================================
global_cache = {
    "data": [],
    "last_updated": 0
}

CACHE_INTERVAL_SECONDS = 7200  
CORE_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "LLY", "JPM", "V", "UNH", "WMT",
    "RY.TO", "TD.TO", "SHOP.TO", "CP.TO", "CNR.TO"
]
# Replace this with your actual live Vercel base URL
BASE_DATA_URL = "https://your-vercel-app-url.vercel.app/api/data"

# ==========================================
# CORE QUANTITATIVE MATHEMATICAL ENGINES
# ==========================================
def heavy_matrix_calculations():
    print(f"[{time.strftime('%X')}] Background Daemon: Initiating 15-stock Markov regressions...")
    results = []
    
    for ticker in CORE_UNIVERSE:
        try:
            # Fetch data from your Vercel-hosted provider
            response = requests.get(f"{BASE_DATA_URL}/{ticker}", timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # --- INSERT YOUR MARKOV LOGIC HERE ---
                # Example:
                # processed_metric = perform_markov_math(data)
                # results.append({"ticker": ticker, "metrics": processed_metric})
                
                # Placeholder so the list populates:
                results.append({"ticker": ticker, "status": "active_processed"})
                print(f"Successfully processed: {ticker}")
            else:
                print(f"Failed to fetch {ticker}: Status {response.status_code}")
                
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            
    print(f"Loop finished. Total assets processed: {len(results)}")
    return results

# ==========================================
# PERSISTENT AUTOMATED DAEMON WORKER
# ==========================================
async def permanent_cache_worker():
    global global_cache
    while True:
        try:
            fresh_data = heavy_matrix_calculations()
            if fresh_data:
                global_cache["data"] = fresh_data
                global_cache["last_updated"] = time.time()
                print(f"[{time.strftime('%X')}] Cache refreshed successfully.")
            else:
                print(f"[{time.strftime('%X')}] Warning: Background run returned empty. Preserving state.")
        except Exception as e:
            print(f"[{time.strftime('%X')}] Critical Error during daemon refresh: {str(e)}")
        
        await asyncio.sleep(CACHE_INTERVAL_SECONDS)

# ==========================================
# FASTAPI LIFECYCLE MANAGEMENT
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(permanent_cache_worker())
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ENDPOINTS
# ==========================================

@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health_check():
    return {
        "status": "online", 
        "timestamp": time.time(),
        "cache_age_seconds": time.time() - global_cache["last_updated"] if global_cache["last_updated"] > 0 else 0
    }

@app.get("/api/screener")
async def get_screener_data(token: str):
    if token != "ecf3ac57988156c7d0dd278042861445":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        
    if not global_cache["data"]:
        print("System Warning: Cache empty. Triggering forced calculation.")
        global_cache["data"] = heavy_matrix_calculations()
        global_cache["last_updated"] = time.time()

    return global_cache["data"]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
