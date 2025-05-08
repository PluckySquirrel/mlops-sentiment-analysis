# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn

# Load model
model = joblib.load("models/sentiment_model.pkl")

# Create app
app = FastAPI(title="Sentiment Analysis API")

# Schemas
class ReviewRequest(BaseModel):
    review: str

class SentimentResponse(BaseModel):
    sentiment: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API"}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(data: ReviewRequest):
    pred = model.predict([data.review])[0]
    sentiment = "positive" if pred == 1 else "negative"
    return {"sentiment": sentiment}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)