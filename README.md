# Sentiment Analysis API with CI/CD Pipeline

This project demonstrates a robust CI/CD pipeline for machine learning, using a sentiment analysis model as a case study. The system trains a Logistic Regression model on 50,000 movie reviews (25,000 train, 25,000 test) and deploys a FastAPI application to predict positive or negative sentiments.

## CI/CD Pipeline
- **Linting**: Ensures code quality using `flake8`.
- **Testing**: Runs unit tests for preprocessing, training, and API endpoints.
- **Dataset Validation**: Verifies `train.csv` and `test.csv` presence and structure.
- **Training**: Trains the model and saves it to `models/`.
- **Model Validation**: Checks model file integrity.
- **Deployment**: Deploys to Render via GitHub Actions.
- **Notifications**: Sends Slack alerts for deployment status.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m nltk.downloader stopwords punkt
2. Train the model:
   ```bash
   python -m src.train_model data/train.csv data/test.csv models/
3. Run the API:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000