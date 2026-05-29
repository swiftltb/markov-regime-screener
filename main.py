import time
import random
import yfinance as yf
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

app = FastAPI()

# --- In-Memory Cache ---
data_cache = {}

def get_data_safely(ticker):
    # 1. Check Cache
    if ticker in data_cache:
        if datetime.now() < data_cache[ticker]['expiry']:
            return data_cache[ticker]['data']

    # 2. Hybrid Routing
    if ticker.endswith('.TO'):
        # Polite throttling for YFinance
        time.sleep(random.uniform(5, 10))
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1mo")
            if not df.empty:
                data_cache[ticker] = {'data': df, 'expiry': datetime.now() + timedelta(hours=4)}
                return df
        except Exception as e:
            print(f"YFinance Error: {e}")
            return None
    else:
        # FMP Logic placeholder
        return None 

@app.get("/analyze", response_class=HTMLResponse)
async def analyze(ticker: str):
    data = get_data_safely(ticker)
    
    # Check if engine is alive
    status = "Operational" if data is not None else "Error/Engine Heartbeat Fail"
    
    # HTML response structure (Responsive UI with Safety Logic)
    return f"""
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div id="safety-alert" style="background: { '#ffcccc' if data is None else '#ccffcc' }; padding: 10px;">
            Engine Status: {status}
        </div>
        
        <h2>{ticker} Analysis</h2>
        <div id="screener-table">...</div>
        <div id="analysis-modal">...</div>
        <canvas id="tickerChart"></canvas>
        
        <script>
            // Chart.js Data Mapping logic goes here
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
