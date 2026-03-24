import pytest
from fastapi.testclient import TestClient
from fuel_pricing.api.main import app

@pytest.fixture
def client():
    return TestClient(app)
