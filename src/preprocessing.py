# src/preprocessing.py

import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

def preprocess_dataset(csv_path: str):
    """
    Load, clean, and split the dataset.
    :param csv_path: Path to the CSV file with 'review' and 'label' columns.
    :return: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(csv_path)

    # Drop nulls
    df.dropna(inplace=True)

    # Clean text
    df['clean_review'] = df['review'].apply(clean_text)

    # Split
    x = df['clean_review']
    y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test
