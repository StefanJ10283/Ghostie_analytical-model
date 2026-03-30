"""
Component tests for the Analytical Model API endpoints.

Tests all endpoints using FastAPI TestClient with mocked external dependencies
(Data Retrieval API and DynamoDB) so tests run without AWS credentials.

Level of abstraction: Component testing
- Tests the full request/response cycle for each endpoint
- Mocks external services (Data Retrieval API, DynamoDB)
- Does NOT test individual functions (that's covered by unit tests)
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from main import app

client = TestClient(app, raise_server_exceptions=False)

# ── Sample test data ──────────────────────────────────────────────────────────

SAMPLE_DATA = [
    {"id": "1", "source": "google_maps_reviews", "body": "Great food and excellent service!", "rating": 5},
    {"id": "2", "source": "google_maps_reviews", "body": "Decent place, nothing special.", "rating": 3},
    {"id": "3", "source": "google_maps_reviews", "body": "Terrible experience, awful food.", "rating": 1},
]

RETRIEVAL_NEW_DATA = {
    "status": "NEW DATA",
    "hash_key": "abc123",
    "data": SAMPLE_DATA,
}

RETRIEVAL_NO_NEW_DATA = {
    "status": "NO NEW DATA",
    "hash_key": "abc123",
}

RETRIEVAL_CACHED = {
    "status": "FOUND",
    "hash_key": "abc123",
    "data": SAMPLE_DATA,
}

DYNAMO_HISTORY_ITEMS = [
    {
        "business_key": "subway_sydney_restaurant",
        "date_time": "2026-03-15T10:00:00+00:00",
        "overall_sentiment": "positive",
        "overall_rating": 4,
        "overall_score": 64.2,
    },
    {
        "business_key": "subway_sydney_restaurant",
        "date_time": "2026-03-14T10:00:00+00:00",
        "overall_sentiment": "neutral",
        "overall_rating": 3,
        "overall_score": 52.0,
    },
]

# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_healthy_status(self):
        response = client.get("/health")
        assert response.json()["status"] == "healthy"

# ── / (root) ──────────────────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_returns_service_info(self):
        response = client.get("/")
        body = response.json()
        assert "service" in body
        assert "version" in body
        assert "endpoints" in body

# ── /sentiment ────────────────────────────────────────────────────────────────

class TestSentimentEndpoint:
    def test_missing_params_returns_422(self):
        response = client.get("/sentiment")
        assert response.status_code == 422

    def test_missing_location_returns_422(self):
        response = client.get("/sentiment?business_name=Subway&category=restaurant")
        assert response.status_code == 422

    @patch("main.httpx.get")
    @patch("main.save_to_dynamodb")
    def test_new_data_returns_200(self, mock_save, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = RETRIEVAL_NEW_DATA
        mock_get.return_value = mock_response

        response = client.get("/sentiment?business_name=Subway&location=Sydney&category=restaurant")
        assert response.status_code == 200

    @patch("main.httpx.get")
    @patch("main.save_to_dynamodb")
    def test_response_contains_required_fields(self, mock_save, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = RETRIEVAL_NEW_DATA
        mock_get.return_value = mock_response

        response = client.get("/sentiment?business_name=Subway&location=Sydney&category=restaurant")
        body = response.json()
        for key in ["business_name", "location", "category", "overall_sentiment",
                    "overall_rating", "overall_score", "items_analysed", "keywords", "breakdown"]:
            assert key in body, f"Missing key: {key}"

    @patch("main.httpx.get")
    @patch("main.save_to_dynamodb")
    def test_score_is_0_to_100(self, mock_save, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = RETRIEVAL_NEW_DATA
        mock_get.return_value = mock_response

        response = client.get("/sentiment?business_name=Subway&location=Sydney&category=restaurant")
        body = response.json()
        assert 0 <= body["overall_score"] <= 100

    @patch("main.httpx.get")
    @patch("main.save_to_dynamodb")
    def test_no_new_data_fetches_by_hash(self, mock_save, mock_get):
        mock_no_new = MagicMock()
        mock_no_new.status_code = 200
        mock_no_new.json.return_value = RETRIEVAL_NO_NEW_DATA

        mock_cached = MagicMock()
        mock_cached.status_code = 200
        mock_cached.json.return_value = RETRIEVAL_CACHED

        mock_get.side_effect = [mock_no_new, mock_cached]

        response = client.get("/sentiment?business_name=Subway&location=Sydney&category=restaurant")
        assert response.status_code == 200
        assert mock_get.call_count == 2

    @patch("main.httpx.get")
    @patch("main.save_to_dynamodb")
    def test_result_saved_to_dynamodb(self, mock_save, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = RETRIEVAL_NEW_DATA
        mock_get.return_value = mock_response

        client.get("/sentiment?business_name=Subway&location=Sydney&category=restaurant")
        mock_save.assert_called_once()

    @patch("main.httpx.get")
    @patch("main.save_to_dynamodb")
    def test_retrieval_error_returns_error_status(self, mock_save, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"detail": "Internal error"}
        mock_get.return_value = mock_response

        response = client.get("/sentiment?business_name=Subway&location=Sydney&category=restaurant")
        assert response.status_code == 500

# ── /analyse ──────────────────────────────────────────────────────────────────

class TestAnalyseEndpoint:
    def test_missing_text_returns_422(self):
        response = client.get("/analyse")
        assert response.status_code == 422

    def test_positive_text_returns_positive(self):
        response = client.get("/analyse?text=Amazing+food+and+excellent+service!")
        assert response.status_code == 200
        body = response.json()
        assert body["sentiment"] == "positive"

    def test_negative_text_returns_negative(self):
        response = client.get("/analyse?text=Terrible+awful+disgusting+horrible+place")
        assert response.status_code == 200
        body = response.json()
        assert body["sentiment"] == "negative"

    def test_response_contains_required_fields(self):
        response = client.get("/analyse?text=The+food+was+good")
        body = response.json()
        for key in ["text", "sentiment", "score", "rating"]:
            assert key in body

    def test_score_is_0_to_100(self):
        response = client.get("/analyse?text=Great+experience+overall")
        body = response.json()
        assert 0 <= body["score"] <= 100

    def test_rating_is_1_to_5(self):
        response = client.get("/analyse?text=Great+experience+overall")
        body = response.json()
        assert 1 <= body["rating"] <= 5

    def test_text_echoed_in_response(self):
        response = client.get("/analyse?text=Hello+world")
        body = response.json()
        assert body["text"] == "Hello world"

# ── /history ──────────────────────────────────────────────────────────────────

class TestHistoryEndpoint:
    def test_missing_params_returns_422(self):
        response = client.get("/history")
        assert response.status_code == 422

    @patch.object(main.analytical_results_table, "query")
    def test_returns_200_with_data(self, mock_query):
        mock_query.return_value = {"Items": DYNAMO_HISTORY_ITEMS}
        response = client.get("/history?business_name=Subway&location=Sydney&category=restaurant")
        assert response.status_code == 200

    @patch.object(main.analytical_results_table, "query")
    def test_returns_404_when_no_history(self, mock_query):
        mock_query.return_value = {"Items": []}
        response = client.get("/history?business_name=Unknown&location=Sydney&category=restaurant")
        assert response.status_code == 404

    @patch.object(main.analytical_results_table, "query")
    def test_response_contains_results(self, mock_query):
        mock_query.return_value = {"Items": DYNAMO_HISTORY_ITEMS}
        response = client.get("/history?business_name=Subway&location=Sydney&category=restaurant")
        body = response.json()
        assert "results" in body
        assert "count" in body
        assert body["count"] == len(DYNAMO_HISTORY_ITEMS)

    @patch.object(main.analytical_results_table, "query")
    def test_each_result_has_required_fields(self, mock_query):
        mock_query.return_value = {"Items": DYNAMO_HISTORY_ITEMS}
        response = client.get("/history?business_name=Subway&location=Sydney&category=restaurant")
        for item in response.json()["results"]:
            for key in ["date_time", "overall_sentiment", "overall_rating", "overall_score"]:
                assert key in item

# ── /leaderboard ──────────────────────────────────────────────────────────────

class TestLeaderboardEndpoint:
    SCAN_ITEMS = [
        {"business_key": "subway_sydney_restaurant", "date_time": "2026-03-15T10:00:00",
         "overall_score": 72.0, "overall_sentiment": "positive", "overall_rating": 4,
         "business_name": "Subway", "location": "Sydney", "category": "restaurant"},
        {"business_key": "mcdonalds_sydney_restaurant", "date_time": "2026-03-15T09:00:00",
         "overall_score": 55.0, "overall_sentiment": "neutral", "overall_rating": 3,
         "business_name": "McDonald's", "location": "Sydney", "category": "restaurant"},
    ]

    @patch.object(main.analytical_results_table, "scan")
    def test_returns_200(self, mock_scan):
        mock_scan.return_value = {"Items": self.SCAN_ITEMS}
        response = client.get("/leaderboard")
        assert response.status_code == 200

    @patch.object(main.analytical_results_table, "scan")
    def test_returns_leaderboard_key(self, mock_scan):
        mock_scan.return_value = {"Items": self.SCAN_ITEMS}
        response = client.get("/leaderboard")
        assert "leaderboard" in response.json()

    @patch.object(main.analytical_results_table, "scan")
    def test_max_5_results(self, mock_scan):
        items = [
            {"business_key": f"biz{i}_sydney_cafe", "date_time": "2026-03-15T10:00:00",
             "overall_score": float(i * 10), "overall_sentiment": "positive", "overall_rating": 4,
             "business_name": f"Biz{i}", "location": "Sydney", "category": "cafe"}
            for i in range(10)
        ]
        mock_scan.return_value = {"Items": items}
        response = client.get("/leaderboard")
        assert len(response.json()["leaderboard"]) <= 5

    @patch.object(main.analytical_results_table, "scan")
    def test_sorted_by_score_descending(self, mock_scan):
        mock_scan.return_value = {"Items": self.SCAN_ITEMS}
        response = client.get("/leaderboard")
        scores = [item["overall_score"] for item in response.json()["leaderboard"]]
        assert scores == sorted(scores, reverse=True)

    @patch.object(main.analytical_results_table, "scan")
    def test_takes_latest_per_business(self, mock_scan):
        # Two entries for same business — should only return the latest
        items = [
            {"business_key": "subway_sydney_restaurant", "date_time": "2026-03-15T10:00:00",
             "overall_score": 72.0, "overall_sentiment": "positive", "overall_rating": 4,
             "business_name": "Subway", "location": "Sydney", "category": "restaurant"},
            {"business_key": "subway_sydney_restaurant", "date_time": "2026-03-14T10:00:00",
             "overall_score": 30.0, "overall_sentiment": "negative", "overall_rating": 2,
             "business_name": "Subway", "location": "Sydney", "category": "restaurant"},
        ]
        mock_scan.return_value = {"Items": items}
        response = client.get("/leaderboard")
        results = response.json()["leaderboard"]
        assert len(results) == 1
        assert results[0]["overall_score"] == 72.0
