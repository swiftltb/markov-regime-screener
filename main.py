from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# CONFIGURATION
# Replace with your actual Cloudflare Worker URL
PROXY_URL = "https://your-proxy-name.your-subdomain.workers.dev/"

# --- SHIELD LOGIC ---
def fetch_data_from_shield(ticker):
    """
    Acts as the 'Brain' calling the 'Shield' (Cloudflare Worker).
    This keeps your Render IP clean and prevents blocking.
    """
    try:
        response = requests.get(f"{PROXY_URL}?ticker={ticker}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"error": "Shield returned status " + str(response.status_code)}
    except Exception as e:
        return {"error": str(e)}

# --- WEB UI ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "data": None})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_ticker(request: Request, ticker: str = Form(...)):
    # Fetch data via Shield
    raw_data = fetch_data_from_shield(ticker.upper())
    
    # Simple Safety/Error Check
    if "error" in raw_data:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": f"Failed to retrieve data for {ticker}. Check Shield status."
        })

    # Prepare data for Chart.js and UI
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "ticker": ticker.upper(),
        "data": raw_data
    })

# --- STATUS ENDPOINT ---
@app.get("/health")
def health_check():
    return {"status": "Brain Online", "shield_url": PROXY_URL}
