"""
Unit tests for data_processor.py

Tests:
- _star_to_score:   converts star rating to -1 to +1 score
- extract_keywords: extracts top N keywords from data items
- _analyse_item:    scores individual review/news items
- analyse_business: aggregates scores across all items
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import _star_to_score, extract_keywords, _analyse_item, analyse_business

# ── _star_to_score ────────────────────────────────────────────────────────────

class TestStarToScore:
    def test_5_star_returns_1(self):
        assert _star_to_score(5) == 1.0

    def test_1_star_returns_minus_1(self):
        assert _star_to_score(1) == -1.0

    def test_3_star_returns_0(self):
        assert _star_to_score(3) == 0.0

    def test_4_star_returns_0_5(self):
        assert _star_to_score(4) == 0.5

    def test_2_star_returns_minus_0_5(self):
        assert _star_to_score(2) == -0.5

    def test_invalid_returns_none(self):
        assert _star_to_score(None) is None

    def test_string_number_converts(self):
        assert _star_to_score("4") == 0.5

    def test_non_numeric_string_returns_none(self):
        assert _star_to_score("abc") is None

# ── extract_keywords ──────────────────────────────────────────────────────────

class TestExtractKeywords:
    def test_returns_list(self):
        data = [{"body": "The food was great and the service was excellent"}]
        result = extract_keywords(data)
        assert isinstance(result, list)

    def test_returns_at_most_top_n(self):
        data = [{"body": "great food service staff clean fast friendly good nice"}]
        result = extract_keywords(data, top_n=3)
        assert len(result) <= 3

    def test_excludes_business_name_words(self):
        data = [{"body": "subway has great sandwiches and subway is fast"}]
        result = extract_keywords(data, business_name="Subway")
        assert "subway" not in result

    def test_excludes_stop_words(self):
        data = [{"body": "the food was really very good and the service was also great"}]
        result = extract_keywords(data)
        assert "the" not in result
        assert "and" not in result

    def test_empty_data_returns_empty_list(self):
        result = extract_keywords([])
        assert result == []

    def test_items_with_no_body_skipped(self):
        data = [{"body": ""}, {"body": None}]
        result = extract_keywords(data)
        assert result == []

# ── _analyse_item ─────────────────────────────────────────────────────────────

class TestAnalyseItem:
    def test_positive_review_returns_positive_score(self):
        item = {"source": "google_maps_reviews", "body": "Amazing food and excellent service!", "rating": 5}
        score = _analyse_item(item)
        assert score is not None
        assert score > 0

    def test_negative_review_returns_negative_score(self):
        item = {"source": "google_maps_reviews", "body": "Terrible experience, disgusting food.", "rating": 1}
        score = _analyse_item(item)
        assert score is not None
        assert score < 0

    def test_rating_only_no_body(self):
        item = {"source": "google_maps_reviews", "body": "", "rating": 5}
        score = _analyse_item(item)
        assert score == 1.0

    def test_no_body_no_rating_returns_none(self):
        item = {"source": "google_maps_reviews", "body": "", "rating": None}
        score = _analyse_item(item)
        assert score is None

    def test_news_item_returns_score(self):
        item = {"source": "newsapi", "title": "Company reports record profits", "body": "The firm posted record earnings beating analyst expectations."}
        score = _analyse_item(item)
        assert score is not None
        assert -1.0 <= score <= 1.0

    def test_news_too_short_returns_none(self):
        item = {"source": "newsapi", "title": "News", "body": "Ok"}
        score = _analyse_item(item)
        assert score is None

    def test_score_within_bounds(self):
        item = {"source": "google_maps_reviews", "body": "Good place overall", "rating": 4}
        score = _analyse_item(item)
        assert -1.0 <= score <= 1.0

    def test_review_field_used_as_body(self):
        item = {"source": "google_maps_reviews", "review": "Fantastic experience!", "rating": 5}
        score = _analyse_item(item)
        assert score is not None

# ── analyse_business ──────────────────────────────────────────────────────────

class TestAnalyseBusiness:
    def test_returns_required_keys(self):
        data = [{"source": "google_maps_reviews", "body": "Great food!", "rating": 5}]
        result = analyse_business("TestCafe", "Sydney", "cafe", data)
        for key in ["business_name", "location", "category", "overall_sentiment",
                    "overall_rating", "overall_score", "items_analysed", "keywords", "breakdown"]:
            assert key in result

    def test_empty_data_returns_defaults(self):
        result = analyse_business("TestCafe", "Sydney", "cafe", [])
        assert result["items_analysed"] == 0
        assert result["overall_sentiment"] == "neutral"
        assert result["overall_rating"] == 3
        assert result["overall_score"] == 0.0

    def test_positive_reviews_give_positive_sentiment(self):
        data = [
            {"source": "google_maps_reviews", "body": "Amazing! Best place ever!", "rating": 5},
            {"source": "google_maps_reviews", "body": "Incredible food and service!", "rating": 5},
        ]
        result = analyse_business("TestCafe", "Sydney", "cafe", data)
        assert result["overall_sentiment"] == "positive"
        assert result["overall_score"] > 0

    def test_negative_reviews_give_negative_sentiment(self):
        data = [
            {"source": "google_maps_reviews", "body": "Awful, terrible, disgusting place.", "rating": 1},
            {"source": "google_maps_reviews", "body": "Worst experience ever, horrible service.", "rating": 1},
        ]
        result = analyse_business("TestCafe", "Sydney", "cafe", data)
        assert result["overall_sentiment"] == "negative"
        assert result["overall_score"] < 0

    def test_items_analysed_count_is_correct(self):
        data = [
            {"source": "google_maps_reviews", "body": "Great!", "rating": 5},
            {"source": "google_maps_reviews", "body": "Good!", "rating": 4},
        ]
        result = analyse_business("TestCafe", "Sydney", "cafe", data)
        assert result["items_analysed"] == 2

    def test_business_metadata_preserved(self):
        result = analyse_business("Subway", "Melbourne", "restaurant", [])
        assert result["business_name"] == "Subway"
        assert result["location"] == "Melbourne"
        assert result["category"] == "restaurant"

    def test_breakdown_contains_per_item_scores(self):
        data = [{"source": "google_maps_reviews", "body": "Great food!", "rating": 5}]
        result = analyse_business("TestCafe", "Sydney", "cafe", data)
        assert len(result["breakdown"]) == 1
        item = result["breakdown"][0]
        assert "sentiment" in item
        assert "score" in item
