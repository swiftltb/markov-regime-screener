from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from hmmlearn import hmm
import gc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stockscreen.art"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# 1. Health Check (Matches Dashboard Path /)
@app.get("/")
async def root():
    return {"status": "online", "message": "Compute Engine Ready"}

# 2. Computation Engine
def run_math(df):
    try:
        # Technical Indicators
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs.iloc[-1])))
        
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = float((exp1 - exp2).iloc[-1])

        # HMM Regime Detection
        returns = df['Close'].pct_change().dropna().values.reshape(-1, 1)
        data_subset = returns[-60:]
        model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
        model.fit(data_subset)
        
        current_state = model.predict(data_subset)[-1]
        persistence = float(model.transmat_[current_state, current_state])
        bullish_state = np.argmax(model.means_.flatten())
        regime = "Bullish" if current_state == bullish_state else "Bearish"

        return {
            "regime": regime, 
            "persistence": round(persistence, 2),
            "rsi": round(rsi, 2), 
            "macd": round(macd, 2),
            "price": float(df['Close'].iloc[-1]),
            "year_high": float(df['Close'].max()),
            "year_low": float(df['Close'].min())
        }
    except Exception as e:
        return {"regime": "Error", "details": str(e)}

@app.post("/analyze")
async def analyze(request: Request):
    payload = await request.json()
    
    # 1. Handle the nested structure from FMP's /stable/ endpoint
    # FMP returns {"symbol": "...", "historical": [...]}
    if isinstance(payload, dict):
        # Extract the list of records
        records = payload.get("historical", [])
        # Extract only the 'close' values
        data_to_process = [item["close"] for item in records if "close" in item]
    elif isinstance(payload, list):
        data_to_process = payload
    else:
        return {"regime": "Error", "details": "Unknown data format"}

    # 2. Safety check: Ensure we have data before creating the DataFrame
    if not data_to_process:
        return {"regime": "Error", "details": "No price data found in response"}

    df = pd.DataFrame(data_to_process, columns=["Close"])
    return run_math(df)
