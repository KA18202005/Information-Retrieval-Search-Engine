"""Unit tests for Flask Backend Web Application and REST Endpoints"""

import pytest
from frontend.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_homepage_route(client):
    response = client.get("/")
    assert response.status_code == 200
    html_text = response.data.decode("utf-8")
    assert "Retriev" in html_text
    assert "IR" in html_text
    assert "Academic Edition" in html_text


def test_search_route(client):
    response = client.get("/search?q=information+retrieval")
    assert response.status_code == 200
    html_text = response.data.decode("utf-8")
    assert "result" in html_text


def test_api_search_endpoint(client):
    response = client.get("/api/search?q=machine+learning")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "data" in data
    assert data["data"]["total_results"] > 0


def test_api_suggest_endpoint(client):
    response = client.get("/api/suggest?q=infromation+retrival")
    assert response.status_code == 200
    data = response.get_json()
    assert "suggestion" in data


def test_api_stats_endpoint(client):
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["stats"]["total_documents"] == 14


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["indexed_documents"] == 14
