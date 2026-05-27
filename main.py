from fastapi import FastAPI, HTTPException, Query
import yfinance as yf
import pandas as pd
import logging

# Set up logging for Vercel's build/runtime logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Markov Regime Screener Data Fetcher Proxy",
    description="Serverless proxy app to bypass yfinance IP blacklists on traditional cloud hosting platforms"
)

@app.get("/")
def fetch_data(ticker: str = Query(None, description="The stock ticker symbol to fetch data for (e.g., AAPL, SPY)")):
    """
    Fetches maximum historical daily data for a given ticker from Yahoo Finance 
    and returns a clean, JSON-serializable array of records.
    """
    if not ticker:
        logger.warning("Request received without a ticker parameter.")
        raise HTTPException(
            status_code=400, 
            detail="Ticker parameter is required. Please use the query format: ?ticker=XYZ"
        )
    
    clean_ticker = str(ticker).strip().upper()
    logger.info(f"Processing serverless fetch request for ticker: {clean_ticker}")
    
    try:
        # Initialize the Yahoo Finance Ticker engine
        stock = yf.Ticker(clean_ticker)
        
        # Fetching maximum historical daily data (required for stable statsmodels Markov calculations)
        df = stock.history(period="max")
        
        # Validation check: If yfinance returns an empty DataFrame, the ticker doesn't exist or is blocked
        if df.empty:
            logger.error(f"No historical market data found or returned for ticker: {clean_ticker}")
            raise HTTPException(
                status_code=404, 
                detail=f"No historical market data found for ticker '{clean_ticker}'. Please verify the symbol."
            )
        
        # Reset the index to turn the Date timestamp into an accessible column
        df = df.reset_index()
        
        # Convert Date objects to clean 'YYYY-MM-DD' strings so FastAPI can serialize it to JSON without errors
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Convert all numeric columns explicitly to float to prevent any legacy numpy serialization errors
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # Turn the DataFrame into an array of row-dictionaries
        data_dict = df.to_dict(orient="records")
        
        logger.info(f"Successfully retrieved and structured {len(data_dict)} historical records for {clean_ticker}.")
        return {
            "ticker": clean_ticker,
            "record_count": len(data_dict),
            "data": data_dict
        }
        
    except HTTPException as http_ex:
        # Re-raise explicit HTTP exceptions so they don't get swallowed by the generic catch-all block
        raise http_ex
    except Exception as e:
        logger.error(f"Critical execution failure during data collection for {clean_ticker}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal serverless engine failure while processing asset request: {str(e)}"
        )
