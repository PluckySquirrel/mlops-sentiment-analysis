import os
import pandas as pd
from src.train_model import train_and_evaluate


def test_train_and_evaluate(tmp_path):
    # Create a small sample dataset for train and test
    data = {
        "review": ["Great movie!", "Terrible film."],
        "label": [1, 0]
    }
    df = pd.DataFrame(data)
    train_csv_path = str(tmp_path / "sample_train.csv")
    test_csv_path = str(tmp_path / "sample_test.csv")
    df.to_csv(train_csv_path, index=False)
    df.to_csv(test_csv_path, index=False)

    # Define model directory
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)

    # Run training
    train_and_evaluate(train_csv_path, test_csv_path, model_dir)

    # Check if model and metrics files exist
    model_files = [f for f in os.listdir(model_dir) if f.startswith("sentiment_model_") and f.endswith(".pkl")]
    metrics_files = [f for f in os.listdir(model_dir) if f.startswith("metrics_") and f.endswith(".txt")]
    assert len(model_files) == 1, "Expected one model file"
    assert len(metrics_files) == 1, "Expected one metrics file"
