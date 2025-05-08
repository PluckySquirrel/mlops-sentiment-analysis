# evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from src.preprocessing import clean_text


def evaluate_model(test_csv_path, model_path):
    # Load test data
    df = pd.read_csv(test_csv_path)
    df.dropna(inplace=True)
    df['clean_review'] = df['review'].apply(clean_text)
    x_test = df['clean_review']
    y_test = df['label']

    # Load model
    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(x_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["negative", "positive"])

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)


if __name__ == "__main__":
    evaluate_model("../data/test.csv", "../models/sentiment_model.pkl")