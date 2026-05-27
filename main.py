from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import uvicorn
import os
import requests
import asyncio

app = FastAPI()

# --- CONFIGURATION ---
BASE_DATA_URL = "https://markov-screener-proxy.vercel.app/api/main"

# --- SAFETY LOGIC & ENGINE ---
async def fetch_ticker_data(ticker: str):
    try:
        # Loop for data retrieval
        df = await asyncio.to_thread(yf.download, ticker, period="1y", interval="1d")
        return {"status": "success", "data": df.tail().to_dict()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- DEBUGGING ROUTE ---
@app.get("/debug-path/{path:path}")
async def debug_path(path: str):
    return {"received_path": path}

# --- PRIMARY ROUTES ---
@app.get("/data/{ticker}")
async def get_ticker_data(ticker: str):
    return await fetch_ticker_data(ticker)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: sans-serif; padding: 20px; }
                .callout { padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .warning { background-color: #fff3cd; border: 1px solid #ffeeba; }
                .status { background-color: #d4edda; border: 1px solid #c3e6cb; }
            </style>
        </head>
        <body>
            <div class="callout status">Engine Status: Operational</div>
            <div class="callout warning">Disclaimer: Past performance is not indicative of future results.</div>
            
            <input type="text" id="tickerInput" placeholder="Enter Ticker (e.g. AAPL)">
            <button onclick="fetchData()">Analyze</button>
            
            <table id="screenerTable" border="1">
                <thead><tr><th>Ticker</th><th>Latest Close</th></tr></thead>
                <tbody></tbody>
            </table>
            <canvas id="myChart" width="400" height="200"></canvas>
            
            <script>
                async function fetchData() {
                    const ticker = document.getElementById('tickerInput').value;
                    const res = await fetch('/data/' + ticker);
                    const data = await res.json();
                    alert("Analysis Modal: Data received for " + ticker);
                }
            </script>
        </body>
    </html>
    """

# --- BACKGROUND WORKER (TEST) ---
async def run_debug_test():
    # Giving the server a moment to start before hitting itself
    await asyncio.sleep(5)
    try:
        response = requests.get(f"{BASE_DATA_URL}/debug-path/AAPL")
        print(f"DEBUG RESPONSE: {response.json()}")
    except Exception as e:
        print(f"DEBUG ERROR: {e}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(run_debug_test())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
