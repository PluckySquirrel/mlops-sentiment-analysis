import os
import joblib
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.preprocessing import clean_text
from typing import Dict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(title="Sentiment Analysis API")

# Load latest model
MODEL_DIR = "models"


def get_latest_model() -> str:
    """Return path to the latest model file."""
    try:
        models = [f for f in os.listdir(MODEL_DIR) if f.startswith("sentiment_model_") and f.endswith(".pkl")]
        if not models:
            raise FileNotFoundError("No model files found")
        return os.path.join(MODEL_DIR, max(models, key=lambda x: x.split("_")[-1]))
    except Exception as e:
        logger.error(f"Error finding latest model: {e}")
        raise


try:
    model_path = get_latest_model()
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


# Schemas
class ReviewRequest(BaseModel):
    review: str


class SentimentResponse(BaseModel):
    sentiment: str


class ModelInfo(BaseModel):
    model_path: str
    loaded_at: str


@app.get("/", response_model=Dict[str, str])
async def read_root():
    """Welcome endpoint."""
    logger.info("Received request to root endpoint")
    return {"message": "Welcome to the Sentiment Analysis API"}


@app.get("/model", response_model=ModelInfo)
async def get_model_info():
    """Return information about the loaded model."""
    logger.info("Received request for model info")
    return {"model_path": model_path, "loaded_at": datetime.now().isoformat()}


@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(data: ReviewRequest):
    """Predict sentiment for a given review."""
    logger.info(f"Processing review: {data.review[:50]}...")
    try:
        if not data.review.strip():
            logger.warning("Empty review received")
            raise HTTPException(status_code=400, detail="Review cannot be empty")

        clean_review = clean_text(data.review)
        pred = model.predict([clean_review])[0]
        sentiment = "positive" if pred == 1 else "negative"
        logger.info(f"Predicted sentiment: {sentiment}")
        return {"sentiment": sentiment}

    except HTTPException as e:
        raise e  # Re-raise HTTPException directly
    except Exception as e:
        logger.error(f"Error predicting sentiment: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    