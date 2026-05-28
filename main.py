from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

# 1. Mandatory CORS Middleware for WordPress Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stockscreen.art", "https://www.stockscreen.art"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# 2. Explicit Pre-flight Handler to prevent 405 Method Not Allowed
@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str):
    return JSONResponse(status_code=200, content={"status": "ok"})

# 3. Screener Endpoint (Matches your WordPress Snippet)
@app.get("/screener-data")
async def get_screener_data():
    # Placeholder: Your Markov Engine Logic will be called here
    return {
        "status": "success",
        "data": [
            {"ticker": "AAPL", "price": 190.25, "rsi": 55, "regime": "Bull", "state": "Trending", "persistence": "High", "action": "HOLD"},
            {"ticker": "NVDA", "price": 850.10, "rsi": 62, "regime": "Bull", "state": "Expansion", "persistence": "Very High", "action": "BUY"}
        ]
    }

# 4. Heartbeat/Health Check
@app.get("/health")
async def health_check():
    return {"status": "Brain Online"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
