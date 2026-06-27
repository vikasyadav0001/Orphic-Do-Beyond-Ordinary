from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_read_main():
    """Verify that the base API route is active and returns the welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.text.strip('"') == "Welcome to the project Orphic."

