import logging
import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Markov Regime Unified Serverless & Backend Engine")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Points to itself when hosted on Vercel to fetch unblocked market data
VERCEL_FETCHER_URL = "https://markov-regime-screener.vercel.app/"


# ==========================================
# 1. VERCEL LAYER: DATA FETCHER ENDPOINT
# ==========================================
@app.get("/")
def fetch_data(ticker: str = Query(None)):
    """
    Vercel executes this root route. It fetches maximum historical daily data 
    from Yahoo Finance and returns a clean JSON-serializable array of records.
    """
    if not ticker:
        logger.warning("Vercel Proxy: Request received without a ticker parameter.")
        raise HTTPException(
            status_code=400, 
            detail="Ticker parameter is required. Please use: ?ticker=XYZ"
        )
    
    clean_ticker = str(ticker).strip().upper()
    logger.info(f"Vercel Proxy: Processing asset data fetch for {clean_ticker}")
    
    try:
        stock = yf.Ticker(clean_ticker)
        df = stock.history(period="max")
        
        if df.empty:
            logger.error(f"Vercel Proxy: No data returned from yfinance for {clean_ticker}")
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for ticker '{clean_ticker}'."
            )
        
        # Clean and format data for seamless transmission
        df = df.reset_index()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        data_dict = df.to_dict(orient="records")
        return {
            "ticker": clean_ticker,
            "record_count": len(data_dict),
            "data": data_dict
        }
        
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Vercel Proxy Errors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 2. RENDER LAYER: MARKOV CALCULATOR INTEGRATION
# ==========================================
def get_markov_calculations(ticker):
    """
    Helper function that calls the Vercel data fetcher endpoint, rehydrates 
    the data into a DataFrame, and runs the Markov Regression calculations.
    """
    clean_ticker = str(ticker).strip().upper()
    try:
        logger.info(f"Render Engine: Querying serverless proxy for {clean_ticker}")
        
        # Pull data from the live Vercel route we verified earlier
        response = requests.get(VERCEL_FETCHER_URL, params={"ticker": clean_ticker}, timeout=20)
        
        if response.status_code != 200:
            logger.error(f"Render Engine: Proxy communication failed with status {response.status_code}")
            return None
            
        payload = response.json()
        raw_records = payload.get("data", [])
        
        if not raw_records:
            return None
            
        # Rehydrate DataFrame
        df = pd.DataFrame(raw_records)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        close_series = df['Close'].dropna().astype(float)
        if len(close_series) < 150:
            logger.warning(f"Render Engine: Insufficient history for {clean_ticker}")
            return None

        # Calculate continuous log returns
        returns = np.log(close_series / close_series.shift(1)).dropna()
        
        # Fit 2-Regime Switching Auto-Regression Model
        model = MarkovRegression(returns, k_regimes=2, trend='c', switching_variance=True)
        model_fit = model.fit(disp=False)
        
        prob_regime_0 = float(model_fit.smoothed_marginal_probabilities[0].iloc[-1])
        prob_regime_1 = float(model_fit.smoothed_marginal_probabilities[1].iloc[-1])
        
        active_regime = 0 if prob_regime_0 > prob_regime_1 else 1
        regime_label = "Low Volatility / Steady State" if active_regime == 0 else "High Volatility / Correction Phase"
        regime_confidence = prob_regime_0 if active_regime == 0 else prob_regime_1

        current_price = float(close_series.iloc[-1])
        pct_change_5d = float(((close_series.iloc[-1] / close_series.iloc[-5]) - 1) * 100) if len(close_series) >= 5 else 0.0

        return {
            "ticker": clean_ticker,
            "status": "Success",
            "last_close": round(current_price, 2),
            "pct_change_5d": round(pct_change_5d, 2),
            "active_regime": active_regime,
            "regime_label": regime_label,
            "regime_probability": round(regime_confidence * 100, 2),
            "chart_data": {
                "labels": df.index.strftime('%Y-%m-%d').tolist()[-90:],
                "prices": close_series.tolist()[-90:],
                "regime_probabilities": model_fit.smoothed_marginal_probabilities[active_regime].tolist()[-90:]
            }
        }
    except Exception as e:
        logger.error(f"Render Engine: Matrix math allocation failure on {clean_ticker}: {str(e)}")
        return None


@app.get("/api/screener")
def screener(token: str = Query(None)):
    """
    Render executes this route. Loops through your asset universe, collects 
    unblocked data from Vercel, processes the statistics, and serves the dashboard.
    """
    if token != os.getenv("API_SECRET_KEY"): 
        logger.warning("Render Engine: Unauthenticated request signature blocked.")
        raise HTTPException(status_code=403, detail="Forbidden: Invalid Signature Token")
        
    ticker_universe = ["SPY", "QQQ", "BTC-USD"]
    processed_results = []
    
    for asset in ticker_universe:
        data = get_markov_calculations(asset)
        if data:
            processed_results.append(data)
            
    return processed_results
