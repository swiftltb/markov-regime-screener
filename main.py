import logging, os
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_markov_data(ticker):
    clean_ticker = str(ticker).strip().upper()
    try:
        # Fetch data
        df = yf.download(clean_ticker, period="5d", interval="1d", progress=False)
        
        # LOGGING EVERYTHING
        logger.info(f"DEBUG: Ticker {clean_ticker} result type: {type(df)}")
        if df is not None:
            logger.info(f"DEBUG: Ticker {clean_ticker} shape: {df.shape}")
        
        if df is None or df.empty:
            return None
            
        return {"ticker": clean_ticker, "status": "Data Received"}
    except Exception as e:
        logger.error(f"Error for {clean_ticker}: {str(e)}")
        return None

@app.get("/api/screener")
def screener(token: str = Query(None)):
    if token != os.getenv("API_SECRET_KEY"): raise HTTPException(403)
    results = [d for d in [get_markov_data("SPY")] if d]
    logger.info(f"Screener returning: {results}")
    return results
