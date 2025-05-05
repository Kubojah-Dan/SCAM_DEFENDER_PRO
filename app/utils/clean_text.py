import re, pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.base import TransformerMixin

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

class CleanTextTransformer(TransformerMixin):
    # existing code…
    def transform(self, X, **kwargs):
        return [self._clean(x) for x in X]
    def fit(self, X, y=None, **kwargs):
        return self
    def _clean(self, text):
        # your existing cleaning…
        return text

def clean_url(url: str) -> str:
    """
    Normalize URL text:
     - lowercase
     - strip scheme (http://, https://) and www.
     - replace non-alphanumeric with spaces
    """
    url = url.lower()  # unify case :contentReference[oaicite:0]{index=0}
    url = re.sub(r"https?://(www\.)?", "", url)      # remove scheme & www. :contentReference[oaicite:1]{index=1}
    url = re.sub(r"[^a-z0-9]", " ", url)             # non-alphanumeric → space :contentReference[oaicite:2]{index=2}
    return url.strip()

# export both names
__all__ = ["CleanTextTransformer", "clean_url"]
