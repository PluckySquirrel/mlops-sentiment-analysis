# tests/test_train_model.py
import os
import joblib
import pandas as pd
from src.train_model import train_and_evaluate


def test_train_and_evaluate(tmp_path):
    # Create a small sample dataset
    data = {
        "review": ["Great movie!", "Terrible film."],
        "label": [1, 0]
    }
    df = pd.DataFrame(data)
    csv_path = str(tmp_path / "sample_test.csv")
    df.to_csv(csv_path, index=False)

    # Define model path
    model_path = str(tmp_path / "test_model.pkl")

    # Run training
    train_and_evaluate(csv_path, model_path)

    # Check if model file exists
    assert os.path.exists(model_path), "Model file should be created"

    # Load and test model
    model = joblib.load(model_path)
    prediction = model.predict(["Amazing experience!"])
    assert prediction[0] in [0, 1], "Prediction should be 0 or 1"