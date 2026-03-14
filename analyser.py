import os
import re
import joblib
from custom_analyser import custom_score

# Load trained model once at startup
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")
_model = joblib.load(_MODEL_PATH) if os.path.exists(_MODEL_PATH) else None

if _model:
    print("Loaded trained sentiment model.")
else:
    print("sentiment_model.pkl not found — falling back to VADER+lexicon.")

def combined_rating(avg):
    """Map a -1 to +1 average score to a 1-5 rating."""
    if avg < -0.5:
        return 1
    if avg < -0.15:
        return 2
    if avg < 0.15:
        return 3
    if avg < 0.5:
        return 4
    return 5

def combined_label(avg):
    """Map a -1 to +1 average score to a sentiment label."""
    if avg < -0.15:
        return "negative"
    if avg > 0.15:
        return "positive"
    return "neutral"

def _ml_score(text: str) -> float:
    """Score text using the trained scikit-learn model.

    Returns a -1 to +1 score derived from class probabilities:
        score = P(positive) - P(negative)
    """
    if _model is None:
        return custom_score(text)

    words = text.split()
    if len(words) <= 40:
        # Short enough — score directly as trained
        probs = _model.predict_proba([text])[0]
        return float(probs[2] - probs[0])

    # Long article: score sentence by sentence
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.split()) >= 4]
    if not sentences:
        probs = _model.predict_proba([text])[0]
        return float(probs[2] - probs[0])

    scores = []
    for sent in sentences:
        probs = _model.predict_proba([sent])[0]
        scores.append(float(probs[2] - probs[0]))

    # Use only sentences with meaningful signal to avoid neutral filler diluting the result
    strong = [s for s in scores if abs(s) > 0.15]
    if strong:
        return sum(strong) / len(strong)
    # All sentences are neutral — return the overall mean
    return sum(scores) / len(scores)

def analyse(text: str, use_ml: bool = True):
    """Run sentiment analysis and return a result tuple.

    For news (use_ml=True): uses trained model blended with VADER+lexicon.
    For reviews (use_ml=False): uses VADER+lexicon only.

    Returns: (text, label, lex_score, final_score, label, rating)
    The 4th element (final_score) is what data_processor.py uses.
    """
    from custom_analyser import _lexicon_score
    lex_score = custom_score(text)
    raw_lex   = _lexicon_score(text)   # un-normalised lexicon signal

    if use_ml and _model is not None and abs(raw_lex) > 0.1:
        # Financial text: lexicon has a signal → blend ML + VADER+lexicon
        ml = _ml_score(text)
        score = ml * 0.3 + lex_score * 0.7
    else:
        # General/non-financial text: no lexicon signal → VADER only
        score = lex_score

    label  = combined_label(score)
    rating = combined_rating(score)
    return text, label, lex_score, score, label, rating
