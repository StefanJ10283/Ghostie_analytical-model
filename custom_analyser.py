import re
import json
import os
import math
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already present
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

_vader = SentimentIntensityAnalyzer()

# --- Financial Lexicon ---

POSITIVE_WORDS = {
    "profit": 1.0, "profits": 1.0, "revenue": 0.7, "revenues": 0.7,
    "growth": 0.9, "earnings": 0.7, "beat": 1.0, "exceed": 0.9, "exceeds": 0.9,
    "surge": 1.0, "surged": 1.0, "rally": 0.9, "gain": 0.8, "gains": 0.8,
    "rise": 0.7, "rose": 0.7, "increase": 0.6, "increased": 0.6,
    "improve": 0.8, "improved": 0.8, "improvement": 0.8,
    "strong": 0.7, "robust": 0.9, "outperform": 1.0, "outperformed": 1.0,
    "dividend": 0.8, "dividends": 0.8, "recovery": 0.9,
    "expansion": 0.8, "expanding": 0.8, "upgrade": 0.9, "upgraded": 0.9,
    "confidence": 0.7, "momentum": 0.8, "efficient": 0.7, "efficiency": 0.7,
    "innovation": 0.7, "innovative": 0.7, "invest": 0.6, "investment": 0.6,
    "opportunity": 0.7, "opportunities": 0.7, "favorable": 0.8,
    "positive": 0.7, "stable": 0.5, "stability": 0.6,
    "competitive": 0.5, "successful": 0.8, "success": 0.8, "record": 0.9,
    "breakthrough": 1.0, "booming": 1.0, "thriving": 1.0, "lucrative": 0.9,
    "accelerate": 0.7, "accelerating": 0.7, "advance": 0.6,
    "benefit": 0.7, "benefits": 0.7, "progress": 0.7, "ahead": 0.5,
    "diversified": 0.5, "safe": 0.5, "safer": 0.5,
}

NEGATIVE_WORDS = {
    "loss": 1.0, "losses": 1.0, "decline": 0.9, "declined": 0.9,
    "fall": 0.7, "fell": 0.7, "drop": 0.8, "dropped": 0.8,
    "recession": 1.0, "crisis": 1.0, "risk": 0.6, "risks": 0.6,
    "debt": 0.7, "default": 1.0, "bankrupt": 1.0, "bankruptcy": 1.0,
    "miss": 0.8, "missed": 0.8, "misses": 0.8, "missing": 0.7,
    "weak": 0.8, "weakness": 0.8,
    "poor": 0.8, "concern": 0.6, "concerns": 0.6, "uncertainty": 0.7,
    "volatile": 0.7, "volatility": 0.7, "downgrade": 1.0, "downgraded": 1.0,
    "penalty": 0.9, "penalties": 0.9, "fine": 0.7,
    "lawsuit": 0.9, "lawsuits": 0.9, "litigation": 0.9,
    "court": 0.6, "courts": 0.6, "injunction": 0.9, "ruling": 0.6,
    "fraud": 1.0, "layoff": 1.0, "layoffs": 1.0, "cut": 0.6, "cuts": 0.6,
    "reduce": 0.5, "reduced": 0.5, "deficit": 1.0, "burden": 0.8,
    "constraint": 0.7, "constraints": 0.7, "concentrated": 0.5,
    "concentration": 0.6, "entrenched": 0.8, "dominance": 0.7, "dominant": 0.6,
    "barrier": 0.8, "barriers": 0.8, "restrict": 0.7, "restriction": 0.7,
    "restrictions": 0.7, "limited": 0.5, "limit": 0.5, "limits": 0.5,
    "compress": 0.7, "compression": 0.7, "squeeze": 0.7, "strain": 0.7,
    "struggle": 0.8, "difficult": 0.6, "difficulty": 0.6, "difficulties": 0.6,
    "deteriorate": 1.0, "deteriorating": 1.0, "worsen": 0.9, "worsening": 0.9,
    "damage": 0.9, "harm": 0.8, "harmful": 0.8, "negative": 0.7,
    "pressure": 0.6, "pressures": 0.6, "challenge": 0.5, "challenges": 0.5,
    "headwind": 0.8, "headwinds": 0.8, "underperform": 1.0, "underperforms": 1.0,
    "slump": 0.9, "slumped": 0.9, "shrink": 0.8, "shrinking": 0.8,
    "contract": 0.7, "contracting": 0.7, "fragile": 0.8,
    "vulnerability": 0.8, "vulnerable": 0.8, "costly": 0.7, "overvalued": 0.9,
    "dive": 0.9, "dived": 0.9, "dives": 0.9,
    "misleading": 1.0, "illusory": 1.0, "deceptive": 1.0, "accused": 0.8,
    "antitrust": 0.9, "accc": 0.7, "regulator": 0.6, "probe": 0.7,
    "outage": 0.9, "outages": 0.9, "incident": 0.6, "incidents": 0.6,
    "settle": 0.6, "settlement": 0.7, "penalty": 0.9, "sue": 0.9, "sued": 0.9,
}

# --- Learned Lexicon (auto-updated from real data) ---
_LEARNED_PATH = os.path.join(os.path.dirname(__file__), "learned_lexicon.json")
if os.path.exists(_LEARNED_PATH):
    with open(_LEARNED_PATH) as _f:
        _learned = json.load(_f)
    # Learned words do not override hard-coded ones
    for _w, _v in _learned.get("positive", {}).items():
        POSITIVE_WORDS.setdefault(_w, _v)
    for _w, _v in _learned.get("negative", {}).items():
        NEGATIVE_WORDS.setdefault(_w, _v)

# --- Rule-based Phrases ---

POSITIVE_PHRASES = [
    ("beat expectations", 1.2),
    ("record earnings", 1.2),
    ("record revenue", 1.2),
    ("strong growth", 1.1),
    ("better than expected", 1.2),
    ("above forecast", 1.2),
    ("raised guidance", 1.1),
    ("market share gains", 1.0),
    ("ahead of schedule", 1.0),
    ("profit growth", 1.0),
    ("exceeds forecast", 1.2),
    ("better than anticipated", 1.1),
    ("raised dividend", 1.1),
    ("share buyback", 0.8),
    ("cost savings", 0.8),
    ("improved margins", 1.0),
    ("revenue growth", 0.9),
    ("expanding margins", 1.0),
    ("surpassing analyst", 1.1),
    ("ahead of analyst", 1.0),
]

NEGATIVE_PHRASES = [
    ("at the price of", 1.0),
    ("came at the price", 1.1),
    ("barriers to entry", 1.2),
    ("more entrenched", 1.0),
    ("became more entrenched", 1.2),
    ("raised barriers", 1.1),
    ("missed targets", 1.2),
    ("below expectations", 1.2),
    ("profit warning", 1.2),
    ("under pressure", 0.9),
    ("reinforced the dominance", 1.2),
    ("reinforced dominance", 1.1),
    ("limit their ability", 1.0),
    ("barriers to competition", 1.2),
    ("reduced competition", 1.1),
    ("bite much harder", 1.0),
    ("not weakened", 0.9),
    ("highly concentrated", 0.8),
    ("market concentration", 0.8),
    ("below forecast", 1.2),
    ("missed analyst", 1.1),
    ("worse than expected", 1.2),
    ("job cuts", 1.0),
    ("mass layoffs", 1.1),
]

def _lexicon_score(text):
    """Financial lexicon + phrase scoring. Returns raw (unnormalised) score."""
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    raw = 0.0

    for phrase, weight in POSITIVE_PHRASES:
        if phrase in text_lower:
            raw += weight
    for phrase, weight in NEGATIVE_PHRASES:
        if phrase in text_lower:
            raw -= weight

    for word in words:
        if word in POSITIVE_WORDS:
            raw += POSITIVE_WORDS[word]
        elif word in NEGATIVE_WORDS:
            raw -= NEGATIVE_WORDS[word]

    # tanh normalises to (-1, 1)
    return math.tanh(raw / 3.0)

def custom_score(text):
    """Combine VADER (general NLP) and financial lexicon scores.

    - VADER handles everyday language, negations, intensifiers.
    - Financial lexicon handles domain-specific implied sentiment.
    - When the lexicon has no strong signal, rely on VADER alone so that
      everyday positive/negative language is not diluted by a zero lexicon score.
    Returns a value between -1 (very negative) and +1 (very positive).
    """
    vader_score = _vader.polarity_scores(text)["compound"]  # already -1 to +1
    lex_score = _lexicon_score(text)

    if abs(lex_score) > 0.1:
        # Financial text: blend both signals equally
        return (vader_score + lex_score) / 2
    else:
        # General text: trust VADER fully
        return vader_score
