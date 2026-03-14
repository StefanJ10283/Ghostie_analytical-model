import httpx
import uvicorn
import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from data_processor import analyse_business

DATA_RETRIEVAL_URL = os.environ.get("DATA_RETRIEVAL_URL", "https://8dwmeuc3b1.execute-api.ap-southeast-2.amazonaws.com/Prod")

app = FastAPI(
    title="Ghostie Analytical Model API",
    version="1.0.0",
    root_path="/Prod",
)

@app.get("/")
def root():
    return {
        "service": "Ghostie Analytical Model API",
        "version": "1.0.0",
        "status":  "running",
        "endpoints": {
            "GET /sentiment": "Fetch & analyse business data from the data retrieval server",
            "GET /health":    "Health check",
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/sentiment")
def retrieve(
    business_name: str = Query(...),
    location:      str = Query(...),
    category:      str = Query(...),
):
    """
    Calls the data retrieval server for the latest collected data,
    then runs sentiment analysis and returns an aggregated result.
    """
    # Fetch latest data (or hash if no new data)
    try:
        response = httpx.get(
            f"{DATA_RETRIEVAL_URL}/retrieve",
            params={"business_name": business_name, "location": location, "category": category},
            timeout=30.0,
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to data retrieval server at {DATA_RETRIEVAL_URL}. Is it running?"
        )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json().get("detail", "Data retrieval failed"))

    retrieval = response.json()
    current_hash = retrieval.get("hash_key")

    # If no new data, fetch by hash to get the actual items
    is_cached = retrieval.get("status") == "NO NEW DATA"
    if is_cached:
        try:
            retrieval = httpx.get(
                f"{DATA_RETRIEVAL_URL}/retrieve/{current_hash}",
                timeout=30.0,
            ).json()
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Could not connect to data retrieval server.")

    result = analyse_business(
        business_name=business_name,
        location=location,
        category=category,
        data=retrieval.get("data", []),
    )

    return JSONResponse(content=result)

from mangum import Mangum
handler = Mangum(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
