import sys, os, logging, warnings
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import joblib
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix

# suppress sklearn warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# custom utilities
from app.utils.clean_text      import CleanTextTransformer, clean_url
from app.utils.domain_features import DomainFeatureExtractor

# configuration
WHITELIST  = os.getenv("DOMAIN_WHITELIST",
                       "paypal.com,amazon.com,google.com,facebook.com,youtube.com"
                      ).split(",")
VT_API_KEY = os.getenv("VT_API_KEY", "YOUR_VT_API_KEY_HERE")

DATASETS = {
    "email": os.path.join(os.path.dirname(__file__), '../datasets/email_dataset_cleaned.csv'),
    "msg":   os.path.join(os.path.dirname(__file__), '../datasets/message_dataset.csv'),
    "url":   os.path.join(os.path.dirname(__file__), '../datasets/url_dataset.csv'),
}
OUTPUTS = {
    "email": os.path.join(os.path.dirname(__file__), '../app/models/email_pipeline.pkl'),
    "msg":   os.path.join(os.path.dirname(__file__), '../app/models/message_pipeline.pkl'),
    "url":   os.path.join(os.path.dirname(__file__), '../app/models/url_adv_pipeline_fast.pkl'),
}


def train_text_model(X, y):
    """Clean → TF-IDF → LogisticRegression"""
    pipe = Pipeline([
        ('cleaner', CleanTextTransformer()),
        ('tfidf',   TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)),
        ('clf',     LogisticRegression(solver='saga', max_iter=300, n_jobs=-1, random_state=42))
    ])
    pipe.fit(X, y)
    return pipe


def train_email_model():
    logger.info("🔹 Training Email Model…")
    df = pd.read_csv(DATASETS["email"], encoding='latin1')
    # ensure columns
    df['url']     = df['url'].fillna('0') if 'url' in df.columns else '0'
    df['subject'] = df['subject'].fillna('')
    df['body']    = df['body'].fillna('')
    df['label']   = df['label'].astype(str)
    # combine text
    df['combined_text'] = (
        df['subject'] + ' ' + df['body'] + ' ' +
        df['url'].apply(lambda x: 'hasurl' if str(x).strip()=='1' else 'nourl')
    )
    df = df[df['label'].isin({'0','1'})]
    # train & evaluate
    pipe = train_text_model(df['combined_text'], df['label'])
    acc  = accuracy_score(df['label'], pipe.predict(df['combined_text']))
    logger.info(f"   → Email accuracy: {acc*100:.2f}%")
    # save
    os.makedirs(os.path.dirname(OUTPUTS["email"]), exist_ok=True)
    joblib.dump(pipe, OUTPUTS["email"])
    logger.info("   → Email model saved.\n")


def train_message_model():
    logger.info("🔹 Training Message Model…")
    df = pd.read_csv(DATASETS["msg"])
    if not {'message_text','label'}.issubset(df.columns):
        raise ValueError("message_dataset.csv needs 'message_text' and 'label'")
    df = df.dropna(subset=['message_text','label'])
    df['label'] = df['label'].astype(str)
    pipe = train_text_model(df['message_text'], df['label'])
    acc  = accuracy_score(df['label'], pipe.predict(df['message_text']))
    logger.info(f"   → Message accuracy: {acc*100:.2f}%")
    os.makedirs(os.path.dirname(OUTPUTS["msg"]), exist_ok=True)
    joblib.dump(pipe, OUTPUTS["msg"])
    logger.info("   → Message model saved.\n")


def train_url_model_fast():
    """
    Fast URL model: no network calls, just lexical hashing + optional static domain features.
    This will finish in seconds on CPU.
    """
    logger.info("🔹 Training Fast URL Model…")
    df = pd.read_csv(DATASETS["url"])
    logger.info(f"   → Loaded {len(df)} URL rows; beginning feature extraction")
    if not {'url_text','label'}.issubset(df.columns):
        raise ValueError("url_dataset.csv needs 'url_text' and 'label'")
    df = df.dropna(subset=['url_text','label'])
    df['label'] = df['label'].astype(str)
    # clean & placeholder
    df['cleaned'] = (
        df['url_text'].astype(str)
          .apply(clean_url)
          .apply(lambda x: x if x.strip() else "empty")
    )

    # 1) lexical features: HashingVectorizer
    hasher = HashingVectorizer(n_features=2**10, alternate_sign=False, norm='l2')
    X_hash = hasher.transform(df['cleaned'])

    # 2) (optional) static domain features: no network calls
    #    e.g. length, dot-count, presence of IP
    #    You could implement a simple featurizer here instead of WHOIS/VT.
    dom_pipe = DomainFeatureExtractor(whitelist=WHITELIST, vt_api_key=None)
    # disable expensive steps by passing vt_api_key=None
    dom_feats = dom_pipe.transform(df['url_text'].tolist())
    X_dom = csr_matrix(dom_feats)

    # 3) combine
    X = hstack([X_hash, X_dom], format='csr')

    # 4) train a fast linear classifier
    clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3,
                        n_jobs=-1, random_state=42)
    clf.fit(X, df['label'])
    acc = accuracy_score(df['label'], clf.predict(X))
    logger.info(f"   → URL accuracy: {acc*100:.2f}%")

    # save all four components so your API can unpack them
    os.makedirs(os.path.dirname(OUTPUTS["url"]), exist_ok=True)
    joblib.dump((clean_url, hasher, dom_pipe, clf), OUTPUTS["url"])
    logger.info("   → Fast URL model saved.\n")


def main():
    train_email_model()
    train_message_model()
    train_url_model_fast()

if __name__ == '__main__':
    main()














