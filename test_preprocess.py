# test_preprocess.py
from src.preprocessing import preprocess_dataset


def test_preprocessing():
    x_train, x_test, y_train, y_test = preprocess_dataset("data/train.csv", "data/test.csv")
    print(f"Training set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    print(f"Sample x_train: {x_train.iloc[0][:50]}...")
    print(f"Sample y_train: {y_train.iloc[0]}")


if __name__ == "__main__":
    test_preprocessing()
