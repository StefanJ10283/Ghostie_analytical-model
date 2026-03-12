# Ghostie Analytical Model

A sentiment analysis service that scores hospitality businesses based on news articles and customer reviews. Part of the Ghostie microservices pipeline.

## Architecture

```
Data Collection (port 8000)
        |
        v
Data Retrieval  (port 8001)
        |
        v
Analytical Model (port 8002)  <-- this service
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI server - exposes the `/sentiment` endpoint |
| `data_processor.py` | Processes raw news/review data and aggregates scores |
| `analyser.py` | ML models - FinBERT + custom analyser |
| `custom_analyser.py` | VADER (general NLP) + financial lexicon + rule-based phrases |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the server

```bash
python main.py
```

Server starts at `http://localhost:8002`. Models download automatically on first run (~1.5 GB).

## API Endpoints

### GET /sentiment

Fetches the latest collected data from the data retrieval server, runs sentiment analysis, and returns an aggregated result.

**Query params:**
- `business_name` - e.g. `The Rocks Brewery`
- `location` - e.g. `Sydney`
- `category` - e.g. `restaurant`

**Example:**
```
GET http://localhost:8002/sentiment?business_name=The+Rocks+Brewery&location=Sydney&category=restaurant
```

**Response - new data:**
```json
{
  "status": "NEW DATA",
  "business_name": "The Rocks Brewery",
  "location": "Sydney",
  "category": "restaurant",
  "overall_sentiment": "positive",
  "overall_rating": 5,
  "overall_score": 0.775,
  "items_analysed": 8,
  "hash_key": "2a1adea5...",
  "breakdown": [
    {
      "id": "gmaps_...",
      "source": "google_maps_reviews",
      "body": "Amazing beers and great service!",
      "rating": 5.0,
      "sentiment": "positive",
      "score": 0.978
    }
  ]
}
```

**Response - no new data:**
```json
{
  "status": "NO NEW DATA",
  "hash_key": "2a1adea5...",
  "message": "Data unchanged since last retrieval. Use cached analytical outputs."
}
```

## Rating Scale

| Score | Rating | Sentiment |
|-------|--------|-----------|
| >= +0.50 | 5 | Strongly positive |
| +0.15 to +0.50 | 4 | Mildly positive |
| -0.15 to +0.15 | 3 | Neutral |
| -0.50 to -0.15 | 2 | Mildly negative |
| < -0.50 | 1 | Strongly negative |

## How Scoring Works

Each item is scored differently based on its source:

**Google Maps reviews**
- Has text - ML analysis (60%) blended with star rating (40%)
- No text - star rating only (5 stars = +1.0, 1 star = -1.0)

**News articles**
- Title + body combined, then ML analysis

**ML analysis (per item)**
```
FinBERT score  x 0.3
Custom score   x 0.7
= combined score (-1 to +1)
```

The custom analyser combines VADER (NLP) with a financial lexicon and rule-based phrases, using tanh normalisation to bound the score to (-1, +1).

All item scores are averaged into a single `overall_score` which maps to the 1-5 rating.

## Testing

Call the endpoint twice with the same params to verify the hash check works:

- First call returns `NEW DATA` with full analysis
- Second call returns `NO NEW DATA` (data unchanged)
