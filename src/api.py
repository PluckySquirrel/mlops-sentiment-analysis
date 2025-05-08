# src/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model từ file
model = joblib.load("models/sentiment_model.pkl")

# Tạo app FastAPI
app = FastAPI(title="Sentiment Analysis API")

# Schema cho request
class ReviewRequest(BaseModel):
    review: str

# Schema cho response
class SentimentResponse(BaseModel):
    sentiment: str  # "positive" hoặc "negative"

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API"}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(data: ReviewRequest):
    pred = model.predict([data.review])[0]
    sentiment = "positive" if pred == 1 else "negative"
    return {"sentiment": sentiment}
