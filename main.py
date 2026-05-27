from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import uvicorn
import os
import requests
import time
import asyncio

app = FastAPI()

# --- CONFIGURATION ---
BASE_DATA_URL = "https://markov-screener-proxy.vercel.app/api/main"

# --- DEBUGGING LOGIC ---
def execute_debug_call():
    """Manually triggered to avoid startup lifecycle issues."""
    try:
        # Give the environment a moment to stabilize
        time.sleep(2) 
        response = requests.get(f"{BASE_DATA_URL}/debug-path/AAPL")
        print(f"--- SYSTEM DEBUG LOG: {response.status_code} | {response.text} ---")
    except Exception as e:
        print(f"--- SYSTEM DEBUG LOG ERROR: {e} ---")

@app.get("/trigger-debug")
async def trigger_debug(background_tasks: BackgroundTasks):
    background_tasks.add_task(execute_debug_call)
    return {"message": "Debug task queued. Check Render logs."}

# --- ROUTES ---
@app.get("/debug-path/{path:path}")
async def debug_path(path: str):
    return {"received_path": path}

@app.get("/data/{ticker}")
async def get_ticker_data(ticker: str):
    df = await asyncio.to_thread(yf.download, ticker, period="1y", interval="1d")
    return {"status": "success", "data": df.tail().to_dict()}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <body>
            <h1>Engine Operational</h1>
            <p>Trigger debug: <a href="/trigger-debug">Click here</a></p>
        </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
