from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from hmmlearn import hmm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stockscreen.art", "https://www.stockscreen.art"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

def get_recommendation(regime, rsi, macd, signal, prev_macd, prev_signal, volume_is_high):
    # Detect Crossovers
    bullish_crossover = (prev_macd <= prev_signal) and (macd > signal)
    bearish_crossover = (prev_macd >= prev_signal) and (macd < signal)

    if regime == "Bullish":
        if rsi < 30 and bullish_crossover and volume_is_high: return "Strong Buy"
        if rsi < 50 and bullish_crossover: return "Buy"
        return "Hold"
    else: # Bearish Regime
        if rsi > 70 and bearish_crossover and volume_is_high: return "Strong Sell"
        if rsi > 50 and bearish_crossover: return "Sell"
        return "Hold"

# 1. Health Check
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
        
        # MACD & Signal Line
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # Volume Confirmation
        avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
        volume_is_high = float(df['Volume'].iloc[-1]) > float(avg_vol)
        
        # Current and Previous values
        macd = float(macd_line.iloc[-1])
        signal = float(signal_line.iloc[-1])
        prev_macd = float(macd_line.iloc[-2])
        prev_signal = float(signal_line.iloc[-2])

        # HMM Regime Detection
        returns = df['Close'].pct_change().dropna().values.reshape(-1, 1)
        data_subset = returns[-60:]
        model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
        model.fit(data_subset)
        
        # Predict state and get probabilities
        states = model.predict(data_subset)
        probs = model.predict_proba(data_subset)
        
        current_state = states[-1]
        current_prob = float(probs[-1, current_state]) # Confidence in current state
        persistence = float(model.transmat_[current_state, current_state])
        
        bullish_state = np.argmax(model.means_.flatten())
        regime = "Bullish" if current_state == bullish_state else "Bearish"

        # Recommendation Logic
        rec = get_recommendation(regime, rsi, macd, signal, prev_macd, prev_signal, volume_is_high)

        return {
            "regime": regime, 
            "persistence": round(persistence, 2),
            "probability": round(current_prob, 4), # Added Probability Score
            "rsi": round(rsi, 2), 
            "macd": round(macd, 2),
            "volume_confirmed": volume_is_high,
            "price": float(df['Close'].iloc[-1]),
            "year_high": float(df['Close'].max()),
            "year_low": float(df['Close'].min()),
            "recommendation": rec
        }
    except Exception as e:
        return {"regime": "Error", "details": str(e)}

@app.post("/analyze")
async def analyze(request: Request):
    try:
        payload = await request.json()
        
        if isinstance(payload, dict) and "data" in payload:
            data_to_process = payload["data"]
        elif isinstance(payload, list):
            data_to_process = payload
        else:
            return {"regime": "Error", "details": "Invalid payload structure"}

        if not data_to_process or len(data_to_process) < 30:
            return {"regime": "Error", "details": "Insufficient data"}

        df = pd.DataFrame(data_to_process)
        df.rename(columns={'close': 'Close', 'volume': 'Volume'}, inplace=True)
        
        return run_math(df)
        
    except Exception as e:
        return {"regime": "Error", "details": f"Caught exception: {str(e)}"}
