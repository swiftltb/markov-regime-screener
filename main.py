import asyncio
import time
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

# 2-Hour Precise Background Clock Allocation (7200 seconds)
CACHE_INTERVAL_SECONDS = 7200  

# Unified Expanded Target Portfolio (15 Institutional Assets)
CORE_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",        # Top 5 NASDAQ
    "LLY", "JPM", "V", "UNH", "WMT",               # Top 5 NYSE
    "RY.TO", "TD.TO", "SHOP.TO", "CP.TO", "CNR.TO" # Top 5 TSX
]

# ==========================================
# CORE QUANTITATIVE MATHEMATICAL ENGINES
# ==========================================
def heavy_matrix_calculations():
    """
    Your structural Markov switching engine loop that connects 
    to your Vercel data fetcher and runs math arrays.
    """
    print(f"[{time.strftime('%X')}] Background Daemon: Initiating 15-stock Markov regressions...")
    results = []
    
    try:
        # --- YOUR EXISTING LOGIC LOOP RUNS HERE ---
        # for ticker in CORE_UNIVERSE:
        #     1. Pull data via Vercel proxy
        #     2. Run statsmodels MarkovAutoregression
        #     3. Calculate probabilities & format metrics
        #     4. results.append(formatted_asset_dict)
        pass
    except Exception as e:
        print(f"[{time.strftime('%X')}] Mathematical matrix processing failure: {str(e)}")
        
    return results

# ==========================================
# PERSISTENT AUTOMATED DAEMON WORKER
# ==========================================
async def permanent_cache_worker():
    """
    Persistent background loop that wakes up exactly every 2 hours
    to calculate, update, and hold the market universe in server RAM.
    """
    global global_cache
    while True:
        try:
            fresh_data = heavy_matrix_calculations()
            
            if fresh_data:
                global_cache["data"] = fresh_data
                global_cache["last_updated"] = time.time()
                print(f"[{time.strftime('%X')}] Cache refreshed successfully. Next automated update in 2 hours.")
            else:
                print(f"[{time.strftime('%X')}] Warning: Background run returned empty dataset. Preserving existing cache state.")
        except Exception as e:
            print(f"[{time.strftime('%X')}] Critical Error during daemon cache refresh: {str(e)}")
        
        # Put the worker thread to sleep for exactly 2 hours
        await asyncio.sleep(CACHE_INTERVAL_SECONDS)

# ==========================================
# FASTAPI LIFECYCLE MANAGEMENT
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles automatic background worker registration at server boot"""
    # Spawns the daemon loop asynchronously the microsecond Render spins up
    asyncio.create_task(permanent_cache_worker())
    yield

# Initialize Application Instance
app = FastAPI(lifespan=lifespan)

# Standard Security Configuration for WordPress Layout Cross-Origin Requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ENDPOINTS / ROUTING MATRIX
# ==========================================

@app.get("/api/health")
async def health_check():
    """
    LIGHTWEIGHT KEEP-AWAKE ENDPOINT
    Bypasses token security and math layers for fast, clean uptime tracker pings.
    """
    return {
        "status": "online", 
        "timestamp": time.time(),
        "cache_age_seconds": time.time() - global_cache["last_updated"] if global_cache["last_updated"] > 0 else 0
    }

@app.get("/api/screener")
async def get_screener_data(token: str):
    """
    MAIN DASHBOARD DATA INGESTION PIPELINE
    Serves metrics directly out of RAM instantly (0-2ms response latency).
    """
    global global_cache
    
    # 1. High-Priority Token Security Check
    if token != "ecf3ac57988156c7d0dd278042861445":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token credential")
        
    # 2. Startup Guard: If a user hits the site before the first background thread loop complete
    if not global_cache["data"]:
        print("System Warning: Cache hit during startup execution phase. Running instant calculation block.")
        global_cache["data"] = heavy_matrix_calculations()
        global_cache["last_updated"] = time.time()

    # 3. Deliver populated data cache instantly
    return global_cache["data"]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
