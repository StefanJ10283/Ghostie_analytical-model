import httpx
import uvicorn
import json
import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from data_processor import analyse_business

DATA_RETRIEVAL_URL = "http://localhost:8001"
CACHE_DIR = "analysis_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

app = FastAPI(
    title="Ghostie Analytical Model API",
    version="1.0.0"
)

def cache_key(business_name: str, location: str, category: str, hash_key: str) -> str:
    safe = lambda s: s.lower().strip().replace(" ", "_")
    short_hash = hash_key[:12]
    return os.path.join(CACHE_DIR, f"{safe(business_name)}_{safe(location)}_{safe(category)}_{short_hash}.json")

def save_cache(path: str, result: dict):
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

def load_cache(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

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
    Returns the cached result if data has not changed since last retrieval.
    """
    # Fetch from data retrieval server
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
    path = cache_key(business_name, location, category, current_hash)

    # No new data — return cached analysis, or fetch by hash if no cache exists yet
    if retrieval.get("status") == "NO NEW DATA":
        cached = load_cache(path)
        if cached:
            cached["status"] = "NO NEW DATA (cached)"
            return JSONResponse(content=cached)

        # No cache yet for this hash — fetch the data by hash key and analyse it
        try:
            hash_response = httpx.get(
                f"{DATA_RETRIEVAL_URL}/retrieve/{current_hash}",
                timeout=30.0,
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Could not connect to data retrieval server.")

        if hash_response.status_code != 200:
            raise HTTPException(status_code=404, detail="No cached analysis found and could not retrieve data by hash.")

        retrieval = hash_response.json()

    # Analyse and cache the result
    result = analyse_business(
        business_name=business_name,
        location=location,
        category=category,
        data=retrieval.get("data", []),
    )

    result["status"]   = "NEW DATA" if retrieval.get("status") != "NO NEW DATA" else "NO NEW DATA (cached)"
    result["hash_key"] = current_hash

    save_cache(path, result)

    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
