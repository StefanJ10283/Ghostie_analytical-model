import httpx
import uvicorn
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from data_processor import analyse_business

DATA_RETRIEVAL_URL = "http://localhost:8001"

app = FastAPI(
    title="Ghostie Analytical Model API",
    version="1.0.0"
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

    # If nothing changed, no need to re-analyse
    if retrieval.get("status") == "NO NEW DATA":
        return JSONResponse(content={
            "status":        "NO NEW DATA",
            "hash_key":      retrieval.get("hash_key"),
            "business_name": business_name,
            "location":      location,
            "category":      category,
            "message":       "Data unchanged since last retrieval. Use cached analytical outputs.",
        })

    # Analyse the data array
    result = analyse_business(
        business_name=business_name,
        location=location,
        category=category,
        data=retrieval.get("data", []),
    )

    result["status"]   = "NEW DATA"
    result["hash_key"] = retrieval.get("hash_key")

    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
