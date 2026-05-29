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
    try:
        payload = await request.json()
        
        # LOGGING: See exactly what we got
        print(f"DEBUG: Payload Type: {type(payload)}")
        
        # Scenario 1: It's the {"data": [...]} dictionary
        if isinstance(payload, dict) and "data" in payload:
            data_to_process = payload["data"]
        # Scenario 2: It's a raw list
        elif isinstance(payload, list):
            data_to_process = payload
        else:
            return {"regime": "Error", "details": f"Unknown format: {str(payload)[:50]}"}

        if not data_to_process or len(data_to_process) == 0:
            return {"regime": "Error", "details": "Payload is empty"}

        df = pd.DataFrame(data_to_process, columns=["Close"])
        return run_math(df)
        
    except Exception as e:
        return {"regime": "Error", "details": f"Caught exception: {str(e)}"}
