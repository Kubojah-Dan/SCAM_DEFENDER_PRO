import re, pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('stopwords'); nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

class CleanTextTransformer:
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        if not hasattr(X, 'apply'): X = pd.Series(X)
        return X.apply(clean_text)


