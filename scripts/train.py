import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import joblib
import warnings
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Suppress non‑critical warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# Custom transformers
from app.utils.clean_text       import CleanTextTransformer
from app.utils.domain_features  import DomainFeatureExtractor

# Load your domain whitelist and VirusTotal API key
WHITELIST  = os.getenv("DOMAIN_WHITELIST", "paypal.com,amazon.com,google.com,facebook.com").split(",")
VT_API_KEY = os.getenv("VT_API_KEY", "YOUR_VT_API_KEY_HERE")

def train_text_model(X, y):
    """Pipeline: clean text → TF-IDF → LogisticRegression."""
    pipeline = Pipeline([
        ('cleaner', CleanTextTransformer()),
        ('tfidf',   TfidfVectorizer(stop_words='english',
                                    ngram_range=(1,2),
                                    max_features=5000)),
        ('clf',     LogisticRegression(max_iter=300))
    ])
    pipeline.fit(X, y)
    return pipeline

def train_email_model():
    print("🔹 Training Email Model…")
    path = os.path.join(os.path.dirname(__file__), '../datasets/email_dataset_cleaned.csv')
    df = pd.read_csv(path, encoding='latin1')

    # Fix for missing or null 'url' column (no .get().fillna on a str!)
    if 'url' in df.columns:
        df['url'] = df['url'].fillna('0')
    else:
        df['url'] = '0'

    # Fill missing text fields
    df['subject'] = df['subject'].fillna('')
    df['body']    = df['body'].fillna('')
    df['label']   = df['label'].astype(str)

    # Combine into single feature: "subject body hasurl"/"nourl"
    df['combined_text'] = (
        df['subject'] + ' ' +
        df['body']    + ' ' +
        df['url'].apply(lambda x: 'hasurl' if str(x).strip() == '1' else 'nourl')
    )

    # Keep only valid labels
    df = df[df['label'].isin({'0','1',0,1})]
    df['label'] = df['label'].astype(str)

    # Train & save
    pipeline = train_text_model(df['combined_text'], df['label'])
    out = os.path.join(os.path.dirname(__file__), '../app/models/email_pipeline.pkl')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(pipeline, out)
    print("   → Email model saved.")

def train_message_model():
    print("🔹 Training Message Model…")
    path = os.path.join(os.path.dirname(__file__), '../datasets/message_dataset.csv')
    df = pd.read_csv(path)

    if not {'message_text','label'}.issubset(df.columns):
        raise ValueError("message_dataset.csv must contain 'message_text' and 'label' columns.")
    df = df.dropna(subset=['message_text','label'])
    df['label'] = df['label'].astype(str)

    pipeline = train_text_model(df['message_text'], df['label'])
    out = os.path.join(os.path.dirname(__file__), '../app/models/message_pipeline.pkl')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(pipeline, out)
    print("   → Message model saved.")

def train_url_model():
    print("🔹 Training Advanced URL Model…")
    path = os.path.join(os.path.dirname(__file__), '../datasets/url_dataset.csv')
    df = pd.read_csv(path)

    if not {'url_text','label'}.issubset(df.columns):
        raise ValueError("url_dataset.csv must contain 'url_text' and 'label' columns.")
    df['url_text'] = df['url_text'].fillna('')
    df['label']    = df['label'].astype(str)

    # Text pipeline for URL strings
    text_pipe = Pipeline([
        ('cleaner', CleanTextTransformer()),
        ('tfidf',   TfidfVectorizer(stop_words='english',
                                    ngram_range=(1,2),
                                    max_features=5000))
    ])

    # Domain‐based feature extractor
    domain_pipe = DomainFeatureExtractor(
        whitelist=WHITELIST,
        vt_api_key=VT_API_KEY
    )

    # Combine text + domain features
    full_pipe = Pipeline([
        ('features', FeatureUnion([
            ('text',   text_pipe),
            ('domain', domain_pipe)
        ])),
        ('clf', LogisticRegression(max_iter=300))
    ])

    full_pipe.fit(df['url_text'], df['label'])
    out = os.path.join(os.path.dirname(__file__), '../app/models/url_adv_pipeline.pkl')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(full_pipe, out)
    print("   → Advanced URL model saved.")

def main():
    train_email_model()
    train_message_model()
    train_url_model()

if __name__ == '__main__':
    main()






