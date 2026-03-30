import httpx
import uvicorn
import os
import logging
from decimal import Decimal
from datetime import datetime, timezone
import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from data_processor import analyse_business

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("analytical_model")

# ── DynamoDB setup ──────────────────────────────────────────────────────────────
dynamodb = boto3.resource(
    "dynamodb",
    region_name=os.getenv("DYNAMODB_REGION", "ap-southeast-2"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
)
analytical_results_table = dynamodb.Table("analytical_results")

def floats_to_decimals(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: floats_to_decimals(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [floats_to_decimals(i) for i in obj]
    return obj

def floats_to_ints_and_floats(obj):
    """Convert Decimals back to int/float when reading from DynamoDB."""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, dict):
        return {k: floats_to_ints_and_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [floats_to_ints_and_floats(i) for i in obj]
    return obj

def save_to_dynamodb(business_key: str, result: dict):
    try:
        analytical_results_table.put_item(Item=floats_to_decimals({
            "business_key": business_key,
            "date_time":    datetime.now(timezone.utc).isoformat(),
            **result,
        }))
    except ClientError as e:
        print(f"DynamoDB write failed: {e.response['Error']['Message']}")

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

@app.get("/analyse")
def analyse_text(text: str = Query(..., description="Text to analyse")):
    """Analyse a raw piece of text and return its sentiment score."""
    from analyser import analyse
    _, label, _, score, _, rating = analyse(text)
    return {
        "text":      text,
        "sentiment": label,
        "score":     round((score + 1) / 2 * 100, 1),
        "rating":    rating,
    }

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

    business_key = f"{business_name.lower().strip()}_{location.lower().strip()}_{category.lower().strip()}"

    # Convert scores to 0-100 before saving and returning
    result["overall_score"] = round((result["overall_score"] + 1) / 2 * 100, 1)
    for item in result.get("breakdown", []):
        item["score"] = round((item["score"] + 1) / 2 * 100, 1)

    logger.info("sentiment_analysed business=%s location=%s category=%s items=%d score=%.1f sentiment=%s cached=%s",
                business_name, location, category,
                result["items_analysed"], result["overall_score"],
                result["overall_sentiment"], is_cached)

    save_to_dynamodb(business_key, result)

    return JSONResponse(content=result)

@app.get("/leaderboard")
def leaderboard():
    """Return the top 5 businesses by their most recent analytical score."""
    try:
        response = analytical_results_table.scan(
            ProjectionExpression="business_key, date_time, overall_score, overall_sentiment, overall_rating, business_name, #loc, category",
            ExpressionAttributeNames={"#loc": "location"},
        )
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"DynamoDB error: {e.response['Error']['Message']}")

    items = response.get("Items", [])
    while "LastEvaluatedKey" in response:
        response = analytical_results_table.scan(
            ProjectionExpression="business_key, date_time, overall_score, overall_sentiment, overall_rating, business_name, #loc, category",
            ExpressionAttributeNames={"#loc": "location"},
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        items.extend(response.get("Items", []))

    # Keep only the latest entry per business_key
    latest: dict = {}
    for item in items:
        item = floats_to_ints_and_floats(item)
        key = item["business_key"]
        if key not in latest or item["date_time"] > latest[key]["date_time"]:
            latest[key] = item

    # Sort by overall_score descending, take top 5
    top = sorted(latest.values(), key=lambda x: x.get("overall_score", 0), reverse=True)[:5]

    results = []
    for item in top:
        results.append({
            "business_name":     item.get("business_name"),
            "location":          item.get("location"),
            "category":          item.get("category"),
            "overall_sentiment": item.get("overall_sentiment"),
            "overall_rating":    item.get("overall_rating"),
            "overall_score":     item.get("overall_score"),
            "as_of":             item.get("date_time"),
        })

    logger.info("leaderboard_queried total_businesses=%d", len(latest))
    return {"leaderboard": results}

@app.get("/history")
def history(
    business_name: str = Query(...),
    location:      str = Query(...),
    category:      str = Query(...),
):
    """Return all past analysis results for a business, sorted newest first."""
    from boto3.dynamodb.conditions import Key
    business_key = f"{business_name.lower().strip()}_{location.lower().strip()}_{category.lower().strip()}"
    try:
        response = analytical_results_table.query(
            KeyConditionExpression=Key("business_key").eq(business_key),
            ScanIndexForward=False,  # newest first
        )
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"DynamoDB error: {e.response['Error']['Message']}")

    logger.info("history_queried business_key=%s", business_key)

    if not response.get("Items"):
        raise HTTPException(status_code=404, detail=f"No history found for '{business_name}' in '{location}'.")

    results = []
    for item in response["Items"]:
        item = floats_to_ints_and_floats(item)
        results.append({
            "date_time":         item.get("date_time"),
            "overall_sentiment": item.get("overall_sentiment"),
            "overall_rating":    item.get("overall_rating"),
            "overall_score":     item.get("overall_score"),
        })

    return {"business_key": business_key, "count": len(results), "results": results}

from mangum import Mangum
handler = Mangum(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
