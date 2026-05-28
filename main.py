import requests
import threading
import time
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- CONFIGURATION ---
# Replace with your actual Cloudflare Worker URL
PROXY_URL = "https://raspy-recipe-da41.arthur-barabash.workers.dev/"
# Your Render URL
SELF_URL = "https://markov-screener-api.onrender.com."

# --- HEARTBEAT PINGER ---
def start_heartbeat():
    def ping():
        while True:
            try:
                requests.get(f"{SELF_URL}/health")
                print("Heartbeat sent to keep Render awake.")
            except Exception as e:
                print(f"Heartbeat failed: {e}")
            time.sleep(600)  # Pings every 10 minutes
            
    thread = threading.Thread(target=ping, daemon=True)
    thread.start()

@app.on_event("startup")
async def startup_event():
    start_heartbeat()

# --- SHIELD LOGIC ---
def fetch_data_from_shield(ticker):
    try:
        response = requests.get(f"{PROXY_URL}?ticker={ticker}", timeout=10)
        return response.json() if response.status_code == 200 else {"error": "Shield Error"}
    except Exception as e:
        return {"error": str(e)}

# --- WEB UI ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Corrected TemplateResponse syntax: request is the first argument
    return templates.TemplateResponse(request, "index.html", {"data": None})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_ticker(request: Request, ticker: str = Form(...)):
    raw_data = fetch_data_from_shield(ticker.upper())
    # Corrected TemplateResponse syntax
    return templates.TemplateResponse(request, "index.html", {
        "ticker": ticker.upper(), 
        "data": raw_data
    })

@app.get("/health")
def health_check():
    return {"status": "Brain Online", "shield_url": PROXY_URL}
