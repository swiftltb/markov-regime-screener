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
        # REPLACE THIS URL WITH YOUR ACTUAL WORKER URL
        worker_url = f"https://raspy-recipe-da41.arthur-barabash.workers.dev/?ticker={ticker}"
        response = session.get(worker_url, timeout=20)
        response.raise_for_status()
        raw_data = response.json()
        
        # Validation
        if 'closes' not in raw_data or 'timestamps' not in raw_data:
            return {"ticker": ticker, "price": 0, "rsi": 0, "macd": 0, "regime": f"Error: Keys {list(raw_data.keys())}"}
            
        df = pd.DataFrame({'Close': raw_data['closes']})
        df.index = pd.to_datetime(raw_data['timestamps'], unit='s')
        
        # --- FITTING BLOCK (STABILIZED) ---
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        scaled_returns = df['Returns'] * 100
        
        try:
            # Using Nelder-Mead (nm) solver for better convergence stability
            model = MarkovAutoregression(scaled_returns, k_regimes=2, order=1, trend='c')
            res = model.fit(disp=False, method='nm', maxiter=2000)
            regime = "Bull" if res.filtered_marginal_probabilities[0].iloc[-1] > 0.5 else "Bear"
        except Exception:
            regime = "Neutral"
        # ----------------------------------
        
        # Indicators
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
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
        return {"ticker": ticker, "error": str(e)}

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
