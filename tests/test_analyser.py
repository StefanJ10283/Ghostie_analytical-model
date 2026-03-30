"""
Unit tests for analyser.py

Tests the core sentiment scoring functions:
- combined_rating: maps -1 to +1 score to 1-5 star rating
- combined_label:  maps -1 to +1 score to sentiment label
- analyse:         full sentiment analysis pipeline
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyser import combined_rating, combined_label, analyse

# ── combined_rating ──────────────────────────────────────────────────────────

class TestCombinedRating:
    def test_very_negative_returns_1(self):
        assert combined_rating(-1.0) == 1

    def test_negative_returns_2(self):
        assert combined_rating(-0.3) == 2

    def test_neutral_returns_3(self):
        assert combined_rating(0.0) == 3

    def test_positive_returns_4(self):
        assert combined_rating(0.3) == 4

    def test_very_positive_returns_5(self):
        assert combined_rating(1.0) == 5

    def test_boundary_negative_low(self):
        # exactly -0.5 → rating 2
        assert combined_rating(-0.5) == 2

    def test_boundary_positive_high(self):
        # exactly 0.5 → rating 5 (not < 0.5, so falls through to return 5)
        assert combined_rating(0.5) == 5

    def test_boundary_neutral_low(self):
        # just below -0.15 → rating 2
        assert combined_rating(-0.16) == 2

    def test_boundary_neutral_high(self):
        # just below 0.15 → rating 3
        assert combined_rating(0.14) == 3

# ── combined_label ───────────────────────────────────────────────────────────

class TestCombinedLabel:
    def test_negative_label(self):
        assert combined_label(-0.5) == "negative"

    def test_positive_label(self):
        assert combined_label(0.5) == "positive"

    def test_neutral_label(self):
        assert combined_label(0.0) == "neutral"

    def test_boundary_negative(self):
        assert combined_label(-0.15) == "neutral"

    def test_boundary_positive(self):
        assert combined_label(0.15) == "neutral"

    def test_just_negative(self):
        assert combined_label(-0.16) == "negative"

    def test_just_positive(self):
        assert combined_label(0.16) == "positive"

# ── analyse ──────────────────────────────────────────────────────────────────

class TestAnalyse:
    def test_returns_six_element_tuple(self):
        result = analyse("The company reported strong profits and record earnings.")
        assert len(result) == 6

    def test_positive_text_returns_positive_label(self):
        _, label, _, score, _, _ = analyse("Excellent profits, record revenue, strong growth this quarter.")
        assert label == "positive"
        assert score > 0

    def test_negative_text_returns_negative_label(self):
        _, label, _, score, _, _ = analyse("The company reported massive losses and filed for bankruptcy.")
        assert label == "negative"
        assert score < 0

    def test_neutral_text_returns_neutral_label(self):
        _, label, _, score, _, _ = analyse("The company held its annual general meeting today.")
        assert label == "neutral"

    def test_score_within_bounds(self):
        _, _, _, score, _, _ = analyse("Profits surged to record levels beating all analyst expectations.")
        assert -1.0 <= score <= 1.0

    def test_rating_within_bounds(self):
        _, _, _, _, _, rating = analyse("Strong growth and record revenue this quarter.")
        assert 1 <= rating <= 5

    def test_use_ml_false_still_returns_tuple(self):
        result = analyse("Great food and excellent service!", use_ml=False)
        assert len(result) == 6

    def test_empty_text_does_not_crash(self):
        result = analyse("")
        assert len(result) == 6

    def test_long_article_scores_within_bounds(self):
        long_text = ("The company reported strong revenue growth. " * 20)
        _, _, _, score, _, _ = analyse(long_text)
        assert -1.0 <= score <= 1.0
