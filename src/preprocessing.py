import re
import logging
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Compile regex patterns
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
PUNCT_PATTERN = re.compile(r"[^\w\s]|[\d]")  # Remove punctuation and numbers


def clean_text(text: str) -> str:
    logger.debug(f"Cleaning text: {text[:50]}...")
    if not isinstance(text, str):
        logger.error("Input to clean_text must be a string")
        raise TypeError("Input must be a string")

    try:
        text = text.lower()
        text = URL_PATTERN.sub("", text)
        text = HTML_PATTERN.sub("", text)
        text = PUNCT_PATTERN.sub("", text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        raise


def preprocess_dataset(train_csv_path: str, test_csv_path: str) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    required_columns = ["review", "label"]

    logger.debug(f"Loading training data from {train_csv_path}")
    try:
        train_df = pd.read_csv(train_csv_path)
        logger.debug(f"Training data columns: {train_df.columns.tolist()}")
        if not all(col in train_df.columns for col in required_columns):
            logger.error(f"Training CSV missing required columns: {required_columns}")
            raise KeyError(f"Training CSV must contain {required_columns}")

        train_df.dropna(inplace=True)
        logger.debug("Applying clean_text to training data")
        train_df["clean_review"] = train_df["review"].apply(clean_text)
        x_train = train_df["clean_review"]
        y_train = train_df["label"]

        logger.debug(f"Loading test data from {test_csv_path}")
        test_df = pd.read_csv(test_csv_path)
        logger.debug(f"Test data columns: {test_df.columns.tolist()}")
        if not all(col in test_df.columns for col in required_columns):
            logger.error(f"Test CSV missing required columns: {required_columns}")
            raise KeyError(f"Test CSV must contain {required_columns}")

        test_df.dropna(inplace=True)
        logger.debug("Applying clean_text to test data")
        test_df["clean_review"] = test_df["review"].apply(clean_text)
        x_test = test_df["clean_review"]
        y_test = test_df["label"]

        logger.info(f"Training set size: {len(x_train)}, Test set size: {len(x_test)}")
        return x_train, x_test, y_train, y_test

    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        raise