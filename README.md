# Sentiment Analysis with CI/CD

This project implements a sentiment analysis model using scikit-learn and serves predictions via a FastAPI application. A CI/CD pipeline is configured using GitHub Actions.

## Setup
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare dataset: Place `reviews.csv` in `data/`

## Training
Run `python train.py` to train the model and save it to `models/sentiment_model.pkl`.

## API
Start the API with `uvicorn src.api:app --reload`. Access at `http://localhost:8000`.

## CI/CD
The GitHub Actions pipeline (`ci.yml`) lints code, runs tests, trains the model, and deploys to Heroku.