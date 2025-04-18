import sys
import os

# Add the project root to the Python path so that 'app' is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import joblib
import warnings
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Suppress UserWarnings
warnings.simplefilter(action='ignore', category=UserWarning)

# Import the custom text cleaner from our utility module in the app folder
from app.utils.clean_text import CleanTextTransformer

def train_model(X, y):
    """
    Trains a model using an NLP pipeline that includes:
      - Custom cleaning (via CleanTextTransformer)
      - TF-IDF vectorization (with unigrams and bigrams, max_features=5000)
      - Logistic Regression
    """
    pipeline = Pipeline([
        ('cleaner', CleanTextTransformer()),
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)),
        ('clf', LogisticRegression(max_iter=300))
    ])
    pipeline.fit(X, y)
    return pipeline

def train_email_model():
    print("Training Email Model...")
    # Path to the cleaned email dataset
    email_path = os.path.join(os.path.dirname(__file__), '../datasets/email_dataset_cleaned.csv')
    email_df = pd.read_csv(email_path, encoding="latin1")
    
    # Ensure 'url' column exists
    if 'url' not in email_df.columns:
        email_df['url'] = '0'
    else:
        email_df['url'] = email_df['url'].fillna('0')
    
    # Fill missing values for 'subject' and 'body'
    email_df['subject'] = email_df['subject'].fillna('')
    email_df['body'] = email_df['body'].fillna('')
    email_df['label'] = email_df['label'].astype(str)
    
    # Combine text: "subject body hasurl" if url == "1", else "subject body nourl"
    email_df['combined_text'] = email_df['subject'] + " " + email_df['body'] + " " + \
        email_df['url'].apply(lambda x: "hasurl" if str(x).strip() == "1" else "nourl")
    
    # Filter valid labels ("0" or "1")
    valid_labels = {"0", "1", 0, 1}
    email_df = email_df[email_df['label'].isin(valid_labels)]
    email_df['label'] = email_df['label'].astype(str)
    
    # Train the pipeline using the combined text
    pipeline = train_model(email_df['combined_text'], email_df['label'])
    
    # Save the model pipeline
    model_dir = os.path.join(os.path.dirname(__file__), '../app/models')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(model_dir, 'email_pipeline.pkl'))
    print("Email model saved.")

def train_message_model():
    print("Training Message Model...")
    message_path = os.path.join(os.path.dirname(__file__), '../datasets/message_dataset.csv')
    message_df = pd.read_csv(message_path)
    
    # Check required columns exist
    if 'message_text' not in message_df.columns or 'label' not in message_df.columns:
        raise ValueError("message_dataset.csv must contain 'message_text' and 'label' columns.")
    
    # Drop rows with missing required fields
    message_df = message_df.dropna(subset=['message_text', 'label'])
    message_df['label'] = message_df['label'].astype(str)
    
    pipeline = train_model(message_df['message_text'], message_df['label'])
    
    model_dir = os.path.join(os.path.dirname(__file__), '../app/models')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(model_dir, 'message_pipeline.pkl'))
    print("Message model saved.")

def train_url_model():
    print("Training URL Model...")
    url_path = os.path.join(os.path.dirname(__file__), '../datasets/url_dataset.csv')
    url_df = pd.read_csv(url_path)
    
    # Check required columns for URL dataset
    if 'url_text' not in url_df.columns or 'label' not in url_df.columns:
        raise ValueError("url_dataset.csv must contain 'url_text' and 'label' columns.")
    
    url_df['url_text'] = url_df['url_text'].fillna('')
    url_df['label'] = url_df['label'].astype(str)
    
    pipeline = train_model(url_df['url_text'], url_df['label'])
    
    model_dir = os.path.join(os.path.dirname(__file__), '../app/models')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(model_dir, 'url_pipeline.pkl'))
    print("URL model saved.")

def main():
    train_email_model()
    train_message_model()
    train_url_model()

if __name__ == '__main__':
    main()





