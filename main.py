from fastapi import FastAPI, HTTPException, Query
import yfinance as yf
import pandas as pd
import numpy as np
import traceback # Added this to capture exact error lines

app = FastAPI()

@app.get("/api/regime")
async def get_regime(ticker: str):
    try:
        # 1. Fetch
        clean_ticker = ticker.strip().upper().replace("NYSE:", "")
        df = yf.download(clean_ticker, period="1y", progress=False)
        
        # 2. Validation
        if df is None or df.empty:
            return {"error": f"No data for {clean_ticker}"}
        
        # 3. Forced Data Integrity
        df.columns = [str(c).lower() for c in df.columns]
        if 'close' not in df.columns:
            return {"error": f"Columns found: {list(df.columns)}. Missing 'close'."}
            
        return {"status": "success", "ticker": clean_ticker, "data_points": len(df)}

    except Exception as e:
        # RETURN THE ERROR TO THE BROWSER INSTEAD OF CRASHING
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
