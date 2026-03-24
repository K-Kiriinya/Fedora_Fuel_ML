import pytest
import os
from pathlib import Path

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_home_page(client):
    """Test home page accessibility."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Fedora Fuel ML" in response.text

def test_metrics_endpoint(client):
    """Test metrics fetch (may say error if no model exists)."""
    response = client.get("/metrics/")
    assert response.status_code == 200
    # response.json() will be either metrics or error dict

@pytest.mark.asyncio
async def test_api_login_invalid(client):
    """Test login with wrong credentials."""
    response = client.post(
        "/api/login",
        data={"username": "wrong", "password": "wrong"}
    )
    assert response.status_code == 401

def test_admin_no_auth(client):
    """Test admin access without HTTP Basic Auth."""
    response = client.get("/admin/")
    assert response.status_code == 401
