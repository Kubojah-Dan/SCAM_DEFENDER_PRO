import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources (if not already downloaded)
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans input text by:
      - Removing URLs,
      - Converting to lowercase,
      - Removing punctuation and special characters,
      - Removing stopwords,
      - Lemmatizing tokens.
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    # Lowercase and remove non-alphabetical characters
    text = re.sub(r'[^A-Za-z\s]', '', text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

class CleanTextTransformer:
    """A custom transformer for scikit-learn pipelines that cleans text."""
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # If X is not already a pandas Series, convert it
        if not hasattr(X, 'apply'):
            X = pd.Series(X)
        return X.apply(clean_text)

