# predict_review.py
import joblib
from src.preprocessing import clean_text


def predict_review(review, model_path):
    # Clean review
    clean_review = clean_text(review)

    # Load model
    model = joblib.load(model_path)

    # Predict
    pred = model.predict([clean_review])[0]
    sentiment = "positive" if pred == 1 else "negative"

    return sentiment


if __name__ == "__main__":
    review = "This movie was up to par, but I think it has some good features."
    sentiment = predict_review(review, "../models/sentiment_model.pkl")
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}")