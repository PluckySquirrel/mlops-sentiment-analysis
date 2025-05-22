import pandas as pd
from src.preprocessing import clean_text, preprocess_dataset


def test_clean_text():
    # Test text cleaning
    text = "This is a TEST with https://example.com, <b>HTML</b>, 123, and punctuation!!!"
    cleaned = clean_text(text)
    expected = "test html punctuation"
    assert cleaned == expected, f"Expected '{expected}', got '{cleaned}'"


def test_preprocess_dataset(tmp_path):
    # Create a small sample dataset for train and test
    data = {
        "review": ["I love this!", "Bad movie."],
        "label": [1, 0]
    }
    df = pd.DataFrame(data)
    train_csv_path = str(tmp_path / "sample_train.csv")
    test_csv_path = str(tmp_path / "sample_test.csv")
    df.to_csv(train_csv_path, index=False)
    df.to_csv(test_csv_path, index=False)

    # Test preprocessing
    x_train, x_test, y_train, y_test = preprocess_dataset(train_csv_path, test_csv_path)
    assert len(x_train) == 2, "Expected 2 training samples"
    assert len(x_test) == 2, "Expected 2 test samples"
    assert len(y_train) == 2, "Expected 2 training labels"
    assert len(y_test) == 2, "Expected 2 test labels"
    assert x_train.iloc[0] == "love", "Expected cleaned text 'love'"
    assert y_train.iloc[0] == 1, "Expected label 1"
