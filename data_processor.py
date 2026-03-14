import re
import json
import os
from collections import Counter, defaultdict
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from analyser import analyse, combined_rating, combined_label

_LEARNED_LEXICON_PATH = os.path.join(os.path.dirname(__file__), "learned_lexicon.json")

def _update_learned_lexicon(keywords: list[str], data: list) -> None:
    """Infer polarity of each keyword from all collected items.

    Scores every item (news via ML blend, reviews via star rating + VADER)
    then averages scores for items containing each keyword:
        avg > +0.2  → positive entry
        avg < -0.2  → negative entry
        otherwise   → skipped (ambiguous)

    Requires at least 2 items per keyword.
    Hard-coded lexicon words are never overwritten.
    """
    from custom_analyser import POSITIVE_WORDS, NEGATIVE_WORDS

    hardcoded = set(POSITIVE_WORDS) | set(NEGATIVE_WORDS)

    # Score every item using _analyse_item so reviews (star rating + VADER)
    # and news (ML blend) are both included
    word_scores: dict[str, list[float]] = defaultdict(list)
    for item in data:
        score = _analyse_item(item)
        if score is None:
            continue
        text = f"{item.get('title', '')} {item.get('body', '')}".lower()
        for kw in keywords:
            if kw in text:
                word_scores[kw].append(score)

    if os.path.exists(_LEARNED_LEXICON_PATH):
        with open(_LEARNED_LEXICON_PATH) as f:
            learned = json.load(f)
    else:
        learned = {"positive": {}, "negative": {}}

    updated = []
    for kw in keywords:
        if kw in hardcoded:
            continue
        scores = word_scores.get(kw, [])
        if len(scores) < 2:
            continue
        avg = sum(scores) / len(scores)
        if avg > 0.2:
            learned["positive"][kw] = round(min(avg, 0.8), 2)
            learned["negative"].pop(kw, None)
            updated.append(f"+{kw}")
        elif avg < -0.2:
            learned["negative"][kw] = round(min(abs(avg), 0.8), 2)
            learned["positive"].pop(kw, None)
            updated.append(f"-{kw}")

    with open(_LEARNED_LEXICON_PATH, "w") as f:
        json.dump(learned, f, indent=2)
    if updated:
        print(f"Learned lexicon updated: {', '.join(updated)}")

_STOP_WORDS = set(stopwords.words('english')) | {
    "card", "cards", "gift", "swap", "brand", "brands", "option", "options",
    "also", "one", "two", "three", "may", "get", "use", "via", "per",
    "including", "available", "new", "like", "says", "said", "will", "year",
    "years", "month", "months", "week", "ago", "day", "days", "time",
    "company", "companies", "market", "report", "business", "product", "products",
}

def extract_keywords(data: list, top_n: int = 5,
                     business_name: str = "", location: str = "", category: str = "") -> list[str]:
    """Return the top N keywords by document frequency across all collected items.

    Uses document frequency (max 1 count per article) so a single repetitive
    article cannot dominate the results.
    """
    # Build query-specific stop words so the business name, location, and
    # category don't dominate the keyword list (e.g. "sydney", "commbank")
    query_words = set()
    for phrase in (business_name, location, category):
        query_words.update(re.findall(r'\b[a-z]{3,}\b', phrase.lower()))
    stop = _STOP_WORDS | query_words

    counts = Counter()

    for item in data:
        if item.get("source") == "newsapi":
            text = f"{item.get('title', '')} {item.get('body', '')}"
        else:
            text = item.get("body", "") or ""

        if not text.strip():
            continue

        # Use a set so each word counts once per article (document frequency)
        words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
        counts.update(w for w in words if w not in stop)

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
            # Reviews use VADER+lexicon only (financial model trained on news, not reviews)
            _, _, _, ml_score, _, _ = analyse(body, use_ml=False)
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
            "keywords":          extract_keywords(data, business_name=business_name, location=location, category=category),
            "breakdown":         [],
        }

    overall_score = sum(r["score"] for r in results) / len(results)

    keywords = extract_keywords(data, business_name=business_name, location=location, category=category)
    _update_learned_lexicon(keywords, data)

    return {
        "business_name":     business_name,
        "location":          location,
        "category":          category,
        "overall_sentiment": combined_label(overall_score),
        "overall_rating":    combined_rating(overall_score),
        "overall_score":     round(overall_score, 3),
        "items_analysed":    len(results),
        "keywords":          keywords,
        "breakdown":         results,
    }
