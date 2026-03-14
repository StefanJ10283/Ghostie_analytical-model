import re
from collections import Counter
from datetime import datetime, timedelta, timezone
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from analyser import analyse, combined_rating, combined_label

_STOP_WORDS = set(stopwords.words('english'))

def extract_keywords(data: list, top_n: int = 5, days: int = 7) -> list[str]:
    """Return the top N most frequent non-stop words from the last `days` days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    counts = Counter()

    for item in data:
        ts = item.get("timestamp")
        if ts:
            try:
                item_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if item_time < cutoff:
                    continue
            except ValueError:
                pass

        if item.get("source") == "newsapi":
            text = f"{item.get('title', '')} {item.get('body', '')}"
        else:
            text = item.get("body", "") or ""

        if not text.strip():
            continue

        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        counts.update(w for w in words if w not in _STOP_WORDS)

    return [word for word, _ in counts.most_common(top_n)]

def _star_to_score(rating) -> float:
    """Convert a 1-5 star rating to a -1 to +1 score."""
    try:
        r = float(rating)
        return (r - 3) / 2  # 1→-1.0, 2→-0.5, 3→0.0, 4→+0.5, 5→+1.0
    except (TypeError, ValueError):
        return None

def _analyse_item(item: dict) -> float | None:
    """
    Return a -1 to +1 score for a single news or review item.

    - Reviews: if body text exists, run ML analysis and blend with star rating.
               if body is empty, use star rating only.
    - News:    combine title + body and run ML analysis.
    """
    source = item.get("source", "")
    title  = item.get("title", "") or ""
    body   = item.get("body", "") or ""

    if source == "google_maps_reviews":
        star_score = _star_to_score(item.get("metadata", {}).get("rating"))

        if body.strip():
            # Blend ML score (60%) with star rating (40%)
            _, _, _, ml_score, _, _ = analyse(body)
            if star_score is not None:
                return ml_score * 0.6 + star_score * 0.4
            return ml_score
        elif star_score is not None:
            # No text — rely purely on star rating
            return star_score
        else:
            return None

    else:
        # News article — combine title and body
        text = f"{title}. {body}".strip(". ")
        if len(text.split()) < 4:
            return None
        _, _, _, ml_score, _, _ = analyse(text)
        return ml_score

def analyse_business(business_name: str, location: str, category: str, data: list) -> dict:
    """
    Run sentiment analysis on each item in the data array and aggregate
    into an overall rating.

    Args:
        business_name: Name of the business.
        location:      Location of the business.
        category:      Business category or industry.
        data:          List of news/review items from the data retrieval server.

    Returns:
        dict with overall sentiment, rating, score, and per-item breakdown.
    """
    results = []

    for item in data:
        score = _analyse_item(item)
        if score is None:
            continue

        results.append({
            "id":        item.get("id", ""),
            "source":    item.get("source", ""),
            "title":     item.get("title", ""),
            "body":      (item.get("body") or "")[:200],
            "rating":    item.get("metadata", {}).get("rating"),
            "sentiment": combined_label(score),
            "score":     round(score, 3),
        })

    if not results:
        return {
            "business_name":     business_name,
            "location":          location,
            "category":          category,
            "overall_sentiment": "neutral",
            "overall_rating":    3,
            "overall_score":     0.0,
            "items_analysed":    0,
            "keywords":          extract_keywords(data),
            "breakdown":         [],
        }

    overall_score = sum(r["score"] for r in results) / len(results)

    return {
        "business_name":     business_name,
        "location":          location,
        "category":          category,
        "overall_sentiment": combined_label(overall_score),
        "overall_rating":    combined_rating(overall_score),
        "overall_score":     round(overall_score, 3),
        "items_analysed":    len(results),
        "keywords":          extract_keywords(data),
        "breakdown":         results,
    }
