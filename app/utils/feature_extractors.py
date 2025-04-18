import re
from urllib.parse import urlparse
from sklearn.base import BaseEstimator, TransformerMixin

class URLFeaturizer(BaseEstimator, TransformerMixin):
    """
    Extract lexical and structural features from URLs.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features_list = []
        for url in X:
            parsed = urlparse(url)
            domain = parsed.netloc or ""
            path = parsed.path or ""
            length = len(url)
            num_dots = url.count('.')
            has_ip = 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url) else 0
            features_list.append({
                'domain': domain,
                'path': path,
                'length': length,
                'num_dots': num_dots,
                'has_ip': has_ip
            })
        return features_list
