# tests/test_preprocessing.py
import pandas as pd
from src.preprocessing import clean_text, preprocess_dataset


def test_clean_text():
    # Test text cleaning
    text = "This is a TEST with https://example.com, <b>HTML</b>, 123, and punctuation!!!"
    cleaned = clean_text(text)
    expected = "test html punctuation"
    assert cleaned == expected, f"Expected '{expected}', got '{cleaned}'"


def test_preprocess_dataset():
    # Create a small sample dataset
    data = {
        "review": ["I love this!", "Bad movie."],
        "label": [1, 0]
    }
    df = pd.DataFrame(data)
    csv_path = "tests/sample_test.csv"
    df.to_csv(csv_path, index=False)

    # Test preprocessing
    x_train, x_test, y_train, y_test = preprocess_dataset(csv_path)

    # Check output types and shapes
    assert isinstance(x_train, pd.Series), "x_train should be a pandas Series"
    assert isinstance(x_test, pd.Series), "x_test should be a pandas Series"
    assert isinstance(y_train, pd.Series), "y_train should be a pandas Series"
    assert isinstance(y_test, pd.Series), "y_test should be a pandas Series"
    assert len(x_train) > 0, "x_train should not be empty"
    assert len(x_test) > 0, "x_test should not be empty"
    assert len(y_train) == len(x_train), "y_train length should match x_train"
    assert len(y_test) == len(x_test), "y_test length should match x_test"