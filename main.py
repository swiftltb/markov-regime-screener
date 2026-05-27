from fastapi import FastAPI, HTTPException, Query
import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Markov Regime Screener Data Fetcher Proxy")

@app.get("/")
def fetch_data(ticker: str = Query(None)):
    """
    Root level endpoint to accept direct query string parameters natively.
    """
    if not ticker:
        logger.warning("Request received without a ticker parameter.")
        raise HTTPException(
            status_code=400, 
            detail="Ticker parameter is required. Please use: ?ticker=XYZ"
        )
    
    clean_ticker = str(ticker).strip().upper()
    logger.info(f"Processing fetch request for ticker: {clean_ticker}")
    
    try:
        stock = yf.Ticker(clean_ticker)
        df = stock.history(period="max")
        
        if df.empty:
            logger.error(f"No historical market data found for ticker: {clean_ticker}")
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for ticker '{clean_ticker}'."
            )
        
        # Format dataset columns for smooth JSON parsing
        df = df.reset_index()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        data_dict = df.to_dict(orient="records")
        logger.info(f"Successfully retrieved {len(data_dict)} records for {clean_ticker}.")
        
        return {
            "ticker": clean_ticker,
            "record_count": len(data_dict),
            "data": data_dict
        }
        
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Critical failure during data collection: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal engine failure: {str(e)}"
        )
