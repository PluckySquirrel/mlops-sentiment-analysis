import os
import sys
import joblib
import logging
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.preprocessing import preprocess_dataset

# Configure logging to console
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_and_evaluate(train_csv_path: str, test_csv_path: str, model_dir: str) -> None:
    """
    Train and evaluate a sentiment analysis model, saving the model and metrics.

    Args:
        train_csv_path (str): Path to training CSV.
        test_csv_path (str): Path to testing CSV.
        model_dir (str): Directory to save model and metrics.

    Raises:
        FileNotFoundError: If CSV files or model directory not found.
        Exception: For other training errors.
    """
    logger.debug(
        f"Starting train_and_evaluate with train={train_csv_path}, test={test_csv_path}, model_dir={model_dir}")
    try:
        logger.debug("Creating model directory")
        os.makedirs(model_dir, exist_ok=True)

        logger.debug("Preprocessing data")
        x_train, x_test, y_train, y_test = preprocess_dataset(train_csv_path, test_csv_path)
        logger.debug(f"Preprocessed data: train_size={len(x_train)}, test_size={len(x_test)}")

        logger.debug("Training model")
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("classifier", LogisticRegression(max_iter=1000))
        ])
        pipeline.fit(x_train, y_train)

        logger.debug("Evaluating model")
        y_pred = pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["negative", "positive"])
        cm = confusion_matrix(y_test, y_pred)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(model_dir, f"metrics_{timestamp}.txt")
        logger.debug(f"Saving metrics to {metrics_path}")
        with open(metrics_path, "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))

        model_path = os.path.join(model_dir, f"sentiment_model_{timestamp}.pkl")
        logger.debug(f"Saving model to {model_path}")
        joblib.dump(pipeline, model_path)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metrics saved to {metrics_path}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:\n" + report)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in training/evaluation: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python -m src.train_model <train_csv> <test_csv> <model_dir>")
        sys.exit(1)
    train_and_evaluate(sys.argv[1], sys.argv[2], sys.argv[3])
