# src/train_model.py

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src.preprocessing import preprocess_dataset

def train_and_evaluate(csv_path: str, model_path: str = "models/sentiment_model.pkl"):
    # Load và chia dữ liệu
    try:
        x_train, x_test, y_train, y_test = preprocess_dataset(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    except Exception as e:
        raise Exception(f"Error in preprocessing: {str(e)}")

    # Tạo pipeline gồm vectorizer + model
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression())
    ])

    # Huấn luyện
    pipeline.fit(x_train, y_train)

    # Dự đoán
    y_pred = pipeline.predict(x_test)

    # Đánh giá
    print("Classification report:\n")
    print(classification_report(y_test, y_pred))

    # Lưu model
    joblib.dump(pipeline, model_path)
    print(f"\n✅ Model saved to {model_path}")
