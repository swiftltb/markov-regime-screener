from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn import hmm
import gc

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def run_math(df):
    """Memory-efficient Data Engine: Returns only raw data for frontend processing."""
    try:
        # RSI & MACD logic
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = float(100 - (100 / (1 + rs.iloc[-1])))

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = float((exp1 - exp2).iloc[-1])

        # HMM Logic (Memory-efficient Gaussian HMM)
        returns = df['Close'].pct_change().dropna().values.reshape(-1, 1)
        data_subset = returns[-60:]
        
        gc.collect()
        model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
        model.fit(data_subset)
        
        current_state = model.predict(data_subset)[-1]
        persistence = float(model.transmat_[current_state, current_state])
        bullish_state = np.argmax(model.means_.flatten())
        regime = "Bullish" if current_state == bullish_state else "Bearish"
        
        del model
        gc.collect()

        return {
            "regime": regime,
            "persistence": round(persistence, 2),
            "rsi": round(rsi, 2),
            "macd": round(macd, 2),
            "price": float(df['Close'].iloc[-1])
        }
    except Exception as e:
        return {"regime": "Error", "persistence": 0.0, "rsi": 0.0, "macd": 0.0, "price": 0.0}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "disclaimer": "DISCLAIMER: For informational purposes only.",
        "engine_status": "Engine: ACTIVE (HMM Optimized)"
    })

@app.post("/analyze")
async def analyze(ticker: str = Form(...)):
    try:
        df = yf.download(ticker, period="1y", interval="1d")
        if df.empty: return {"error": "Invalid ticker"}
        results = run_math(df)
        return {"ticker": ticker, "results": results, "chart_data": df['Close'].tail(60).tolist()}
    except Exception as e:
        return {"error": str(e)}
