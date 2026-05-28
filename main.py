from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn, threading, time, requests, functools
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

app = FastAPI()

# 1. CORS & Security (Pre-flight fixed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stockscreen.art", "https://www.stockscreen.art"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str):
    return JSONResponse(status_code=200, content={"status": "ok"})

# 2. Financial Metrics Engine (The Brain)
def calculate_metrics(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi = 100 - (100 / (1 + (gain / loss)))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    return rsi.iloc[-1], macd.iloc[-1]

def run_markov_analysis(ticker):
    try:
        # ARCHITECTURE ENFORCED: Render only speaks to the proxy
        worker_url = f"https://raspy-recipe-da41.arthur-barabash.workers.dev//?ticker={ticker}"
        response = requests.get(worker_url, timeout=20)
        
        if response.status_code != 200:
            return {"ticker": ticker, "error": "Proxy communication failure"}
        
        raw_data = response.json()
        df = pd.DataFrame({'Close': raw_data['closes']})
        df.index = pd.to_datetime(raw_data['timestamps'], unit='s')
        
        # Markov Analysis
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        model = MarkovAutoregression(df['Returns'].dropna(), k_regimes=2, order=1)
        res = model.fit(disp=False)
        regime = "Bull" if res.filtered_marginal_probabilities[0].iloc[-1] > 0.5 else "Bear"
        
        rsi, macd = calculate_metrics(df)
        
        return {
            "ticker": ticker, 
            "price": float(df['Close'].iloc[-1]), 
            "rsi": round(float(rsi), 2), 
            "macd": round(float(macd), 2), 
            "regime": regime
        }
    except Exception as e:
        return {"ticker": ticker, "error": f"Brain processing error: {str(e)}"}

# 3. Cached Data Layer (15 Tickers)
@functools.lru_cache(maxsize=1)
def get_cached_screener_data():
    tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "AMD", "GOOGL", "AMZN", "META", "NFLX", "INTC", "CSCO", "PEP", "ADBE", "QCOM", "AVGO"]
    return [run_markov_analysis(t) for t in tickers]

# 4. API Endpoints
@app.get("/screener-data")
async def get_screener_data():
    return {"status": "success", "data": get_cached_screener_data()}

@app.get("/health")
async def health_check():
    return {"status": "Brain Online"}

# 5. Infrastructure (Heartbeat)
def heartbeat():
    while True:
        try: requests.get("https://markov-screener-api.onrender.com/health")
        except: pass
        time.sleep(600)

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=heartbeat, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
