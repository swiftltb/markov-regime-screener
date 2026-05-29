import time
import random
import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from hmmlearn import hmm
import os

app = FastAPI()

# --- Security ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stockscreen.art", "https://www.stockscreen.art"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Add the Health Check HERE ---
@app.get("/")
async def root():
    return {"status": "Engine Operational"}

# --- Then your existing functions follow ---
def calculate_rsi(data, window=14):
# --- Indicator Calculations ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# --- Analysis Core ---
def perform_analysis(df):
    close = df['Close']
    returns = close.pct_change().dropna().values.reshape(-1, 1)
    
    # HMM Regime Detection
    model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
    model.fit(returns)
    regimes = model.predict(returns)
    
    rsi = calculate_rsi(close).iloc[-1]
    macd, _ = calculate_macd(close)
    
    # Recommendation Logic
    is_bullish = regimes[-1] == 1
    prob = model.predict_proba(returns[-1:])
    
    if is_bullish and rsi < 70:
        rec, reason = "BUY", f"Bullish regime (Prob: {np.max(prob):.2f}) with RSI {rsi:.1f}. Favorable entry."
    elif rsi > 70:
        rec, reason = "SELL", f"RSI {rsi:.1f} indicates overbought conditions. Potential reversal."
    else:
        rec, reason = "HOLD", f"Market is {'Bullish' if is_bullish else 'Bearish'}. Indicators neutral."
    
    return {
        "regime": "Bullish" if is_bullish else "Bearish",
        "regime_probability": float(np.max(prob)),
        "rsi": float(rsi),
        "macd": float(macd.iloc[-1]),
        "recommendation": rec,
        "recommendation_reason": reason
    }

# --- API Route ---
@app.post("/analyze")
async def analyze(request: Request):
    payload = await request.json()
    ticker = payload.get("ticker")
    raw_data = payload.get("data")
    
    if not ticker:
        return {"error": "Ticker required", "status": "Error"}

    try:
        # FMP provided by Worker OR YFinance Fetch
        if raw_data:
            df = pd.DataFrame(raw_data)
        else:
            time.sleep(random.uniform(5, 10))
            stock = yf.Ticker(ticker)
            df = stock.history(period="3mo")
            if df.empty:
                return {"error": f"No data found for {ticker}", "status": "Error"}
        
        results = perform_analysis(df)
        
        return {
            "ticker": ticker,
            "status": "Operational",
            "analysis": results,
            "data": df[['Close', 'Volume']].tail(30).to_dict(orient='records')
        }
    except Exception as e:
        return {"error": str(e), "status": "Error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
