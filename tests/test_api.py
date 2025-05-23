from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Sentiment Analysis API"}


def test_model_info_endpoint():
    response = client.get("/model")
    assert response.status_code == 200
    assert "model_path" in response.json()
    assert "loaded_at" in response.json()


def test_predict_endpoint_positive():
    response = client.post("/predict", json={"review": "Great movie!"})
    assert response.status_code == 200
    assert isinstance(response.json()["sentiment"], float)
    assert response.json()["sentiment"] > 0.5  # Expect positive score > 0.5


def test_predict_endpoint_negative():
    response = client.post("/predict", json={"review": "Terrible film."})
    assert response.status_code == 200
    assert isinstance(response.json()["sentiment"], float)
    assert response.json()["sentiment"] < 0.5  # Expect negative score < 0.5


def test_predict_endpoint_empty():
    response = client.post("/predict", json={"review": ""})
    assert response.status_code == 400
