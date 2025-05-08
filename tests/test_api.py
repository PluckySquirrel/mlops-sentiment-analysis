# tests/test_api.py
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Sentiment Analysis API"}

def test_predict_endpoint_positive():
    response = client.post("/predict", json={"review": "I loved this movie!"})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "positive"}

def test_predict_endpoint_negative():
    response = client.post("/predict", json={"review": "This film was terrible."})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "negative"}