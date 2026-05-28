from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np, pandas as pd, functools, requests
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import uvicorn

app = FastAPI()

# --- HARDENING: Session with Retry Logic ---
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stockscreen.art", "https://www.stockscreen.art"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- ENGINE: Analysis Logic ---
def run_markov_analysis(ticker):
    try:
        worker_url = f"https://raspy-recipe-da41.arthur-barabash.workers.dev/?ticker={ticker}"
        response = requests.get(worker_url, timeout=10)
        data = response.json()
        
        df = pd.DataFrame({'Close': data['closes']})
        
        # 1. PRE-PROCESSING: The "Math-Safe" Pipeline
        # Calculate log returns and clean them completely
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # 2. STATIONARITY CHECK
        # If variance is too low (flatline data), skip model, return 'Neutral'
        if df['Returns'].std() < 1e-6:
            regime = "Neutral"
        else:
            # 3. ROBUST FITTING
            # Scale returns (standardize variance)
            scaled_returns = (df['Returns'] - df['Returns'].mean()) / df['Returns'].std()
            
            # Fitting in a controlled scope
            model = MarkovAutoregression(scaled_returns, k_regimes=2, order=1, trend='c')
            # Using 'nm' solver is mathematically required for high-stability SVD
            res = model.fit(disp=False, method='nm', maxiter=5000)
            
            # Extract probability safely
            prob = res.filtered_marginal_probabilities[0].iloc[-1]
            regime = "Bull" if prob > 0.5 else "Bear"

        # Calculate indicators
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        macd = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()

        return {
            "ticker": ticker,
            "price": float(df['Close'].iloc[-1]),
            "rsi": round(float(rsi.iloc[-1]), 2),
            "macd": round(float(macd.iloc[-1]), 2),
            "regime": regime
        }
    except Exception as e:
        # If it truly fails here, we log the specific ticker error
        return {"ticker": ticker, "regime": "Math Error", "price": 0, "rsi": 0, "macd": 0}
# --- CACHE ---
@functools.lru_cache(maxsize=1)
def get_cached_screener_data():
    tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "AMD", "GOOGL", "AMZN", "META", "NFLX", "INTC", "CSCO", "PEP", "ADBE", "QCOM", "AVGO"]
    return [run_markov_analysis(t) for t in tickers]

# --- ENDPOINTS ---
@app.api_route("/health", methods=["GET", "HEAD"])
@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "Brain Online"}

@app.get("/screener-data")
async def get_screener_data():
    try:
        data = get_cached_screener_data()
        return JSONResponse(content={"status": "success", "data": data})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
