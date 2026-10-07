import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_api_status():
    response = client.get("/api/status")
    # Assert it returns a 200 status code
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "online"
    # Assert latest_timestamp is returned
    assert "latest_timestamp" in data
