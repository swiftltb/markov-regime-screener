import requests
import threading
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from cachetools import cached, TTLCache

app = FastAPI()

# --- SAFETY LOGIC: CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stockscreen.art"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
PROXY_URL = "https://raspy-recipe-da41.arthur-barabash.workers.dev/"
TICKERS = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "INTC", "PYPL", "ADBE", "QCOM", "AVGO", "CSCO"]

# 2-hour TTL cache (7200 seconds)
cache = TTLCache(maxsize=1, ttl=7200)

# --- ENGINE LOGIC ---
def get_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return {"macd": macd.iloc[-1], "signal": signal.iloc[-1]}

def get_markov_metrics(prices):
    returns = np.log(prices / prices.shift(1)).dropna()
    model = sm.tsa.MarkovRegression(returns, k_states=2, trend='c', switching_variance=True)
    res = model.fit(disp=False)
    state = res.smoothed_marginal_probabilities[0].iloc[-1]
    regime = "Bullish" if state > 0.5 else "Bearish"
    persistence = res.regime_transition_matrix[0,0] if regime == "Bullish" else res.regime_transition_matrix[1,1]
    return regime, round(state, 4), round(persistence, 4)

def determine_action(regime, persistence, rsi, macd_vals):
    if regime == "Bullish" and persistence > 0.85 and macd_vals['macd'] > macd_vals['signal'] and rsi < 65:
        return "BUY"
    elif regime == "Bearish" and persistence > 0.85 and macd_vals['macd'] < macd_vals['signal'] and rsi > 35:
        return "SELL"
    return "HOLD"

@cached(cache)
def fetch_and_calculate_all():
    master_data = []
    for ticker in TICKERS:
        try:
            resp = requests.get(f"{PROXY_URL}?ticker={ticker}", timeout=10)
            if resp.status_code == 200:
                raw = resp.json()
                prices = pd.Series(raw.get('prices'))
                
                regime, state, persistence = get_markov_metrics(prices)
                macd_vals = get_macd(prices)
                action = determine_action(regime, persistence, raw.get('rsi', 50), macd_vals)
                
                master_data.append({
                    "ticker": ticker,
                    "price": prices.iloc[-1],
                    "rsi": raw.get('rsi'),
                    "regime": regime,
                    "state": state,
                    "persistence": persistence,
                    "entry": raw.get('entry'),
                    "exit": raw.get('exit'),
                    "stop_loss": raw.get('stop_loss'),
                    "action": action
                })
        except Exception:
            continue
    return {"data": master_data}

# --- HEARTBEAT & ENDPOINTS ---
def start_heartbeat():
    def ping():
        while True:
            try: requests.get("https://markov-screener-api.onrender.com/health")
            except: pass
            time.sleep(600)
    threading.Thread(target=ping, daemon=True).start()

@app.on_event("startup")
async def startup_event():
    start_heartbeat()

@app.get("/screener-data")
async def screener_data():
    return fetch_and_calculate_all()

@app.get("/health")
def health_check():
    return {"status": "Brain Online"}
