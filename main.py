import requests
import threading
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# --- SAFETY LOGIC: CORS CONFIGURATION ---
# Allows your WordPress site to request data from the Brain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stockscreen.art"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
PROXY_URL = "https://raspy-recipe-da41.arthur-barabash.workers.dev/"
SELF_URL = "https://markov-screener-api.onrender.com" 

# --- HEARTBEAT PINGER (Render Keep-Alive) ---
def start_heartbeat():
    def ping():
        while True:
            try:
                requests.get(f"{SELF_URL}/health")
            except Exception as e:
                print(f"Engine Status Alert: Heartbeat failed: {e}")
            time.sleep(600)
    thread = threading.Thread(target=ping, daemon=True)
    thread.start()

@app.on_event("startup")
async def startup_event():
    start_heartbeat()

# --- BACKEND LOGIC: SHIELD INTEGRATION ---
def fetch_data_from_shield(ticker):
    try:
        response = requests.get(f"{PROXY_URL}?ticker={ticker}", timeout=10)
        return response.json() if response.status_code == 200 else {"error": "Shield Error"}
    except Exception as e:
        return {"error": str(e)}

# --- API ENDPOINTS ---
@app.get("/analyze/{ticker}")
async def analyze(ticker: str):
    # Safety Logic: Simple validation
    if not ticker or len(ticker) > 5:
        return {"error": "Invalid Ticker"}
    
    data = fetch_data_from_shield(ticker.upper())
    return {"ticker": ticker.upper(), "payload": data}

@app.get("/health")
def health_check():
    # Engine Status Alert Box
    return {"status": "Brain Online", "shield_status": "Active"}

# --- DISCLAIMER CALLOUT ---
# API provides raw data for UI consumption; ensure WordPress displays this disclaimer
# "Disclaimer: This data is for informational purposes and does not constitute financial advice."
