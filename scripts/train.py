import os, sys, logging, warnings
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import accuracy_score
from app.utils.feature_eng import URLLexicalFeatures
from app.utils.clean_text      import CleanTextTransformer, clean_url
from app.utils.feature_eng     import ScamKeywordCounter, URLLexicalFeatures, FileFeatureExtractor
from app.utils.domain_features import DomainFeatureExtractor
from sklearn.preprocessing import FunctionTransformer

warnings.simplefilter("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

WHITELIST = [
    "google.com",
    "youtube.com",
    "facebook.com",
    "amazon.com",
    "microsoft.com",
    "apple.com",
    "paypal.com",
    "netflix.com",
    "linkedin.com",
    "twitter.com",
    "instagram.com",
    "whatsapp.com",
    "github.com",
    "dropbox.com",
    "bankofamerica.com",
    "chase.com",
    "wellsfargo.com"
]

# ── Paths ───────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATHS = {
    "email_in":  os.path.join(ROOT, "app", "datasets", "email_dataset.csv"),
    "message_in":    os.path.join(ROOT, "app", "datasets", "message_dataset.csv"),
    "url_in":    os.path.join(ROOT, "app", "datasets", "url_dataset.csv"),
    "ben_in":    os.path.join(ROOT, "app", "datasets", "benign.csv"),
    "mal_in":    os.path.join(ROOT, "app", "datasets", "malware.csv"),
    "out_email": os.path.join(ROOT, "app", "models", "email_pipeline.pkl"),
    "message_out": os.path.join(ROOT, "app" , "models", "message_label_encoder.pkl"),
    "out_url":   os.path.join(ROOT, "app", "models", "url_adv_pipeline_fast.pkl"),
    "out_file":  os.path.join(ROOT, "app", "models", "file_malware_pipeline.pkl"),
}

# ── Helper to save and log metrics ─────────────────────────────────────────────
def save_and_log(pipe, X_train, y_train, X_test, y_test, outpath, name):
    tr = accuracy_score(y_train, pipe.predict(X_train))*100
    te = accuracy_score(y_test,  pipe.predict(X_test)) *100
    log.info(f"   → {name} train: {tr:.1f}%   test: {te:.1f}%")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    joblib.dump(pipe, outpath)

# ── 1) EMAIL with XGB, LGB, LR voting ──────────────────────────────────────────
def train_email():
    log.info("🔹 Training Email Model…")
    df = pd.read_csv(PATHS["email_in"], encoding="latin1")
    # new schema: 'label' ∈ {'ham','spam'}, 'message' column
    df = df.dropna(subset=["message","label"])
    # map to 0/1
    df["y"] = df["label"].map({"ham":0,"spam":1})
    X = df["message"].astype(str)
    y = df["y"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=42)

    # features: clean+tfidf  + scam keyword counts
    text_pipe = Pipeline([
        ("clean", CleanTextTransformer()),
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000))
    ])
    key_pipe = ScamKeywordCounter()
    feats = FeatureUnion([("text", text_pipe), ("keys", key_pipe)])

    # classifiers
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=-1, random_state=42)
    lgb = LGBMClassifier(n_jobs=-1, random_state=42)
    lr  = LogisticRegression(solver="saga", max_iter=300, n_jobs=-1, random_state=42)

    vote = VotingClassifier([("xgb",xgb),("lgb",lgb),("lr",lr)], voting="soft", n_jobs=-1)

    pipe = Pipeline([("feats", feats), ("clf", vote)])
    pipe.fit(Xtr, ytr)
    save_and_log(pipe, Xtr, ytr, Xte, yte, PATHS["out_email"], "Email")

# ── 2) MESSAGE model ──────────────────────────────────────────────────────────
def train_message_transformer():
    log.info("🔹 Training Message Transformer Model…")
    df = pd.read_csv(PATHS["message_in"]).dropna()
    df['label'] = LabelEncoder().fit_transform(df['label'])
    # split into train/test
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    # tokenize text
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_enc = tokenizer(
        train_df["message"].tolist(), truncation=True, padding="max_length", max_length=64, return_tensors="pt"
    )
    test_enc = tokenizer(
        test_df["message"].tolist(), truncation=True, padding="max_length", max_length=64, return_tensors="pt"
    )
    # define PyTorch Dataset returning dict with labels
    class SMSDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item
    train_ds = SMSDataset(train_enc, train_df['label'].tolist())
    test_ds = SMSDataset(test_enc, test_df['label'].tolist())
    # create model and training arguments
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    args = TrainingArguments(
        output_dir="app/models/msg_trf",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        logging_dir="logs",
        logging_steps=10,
        eval_strategy="epoch"
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer
    )
    # train and save
    trainer.train()
    model.save_pretrained("app/models/msg_trf_model")
    tokenizer.save_pretrained("app/models/msg_trf_tokenizer")
    log.info("   → Transformer message model saved.")

# ── 3) FAST URL model ─────────────────────────────────────────────────────────

def train_url():
    log.info("🔹 Training Fast URL Model…")
    df = pd.read_csv(PATHS["url_in"])
    df = df.dropna(subset=["url_text","label"])
    df = df[df["label"].isin(["bad","good"])]
    df["y"] = df["label"].map({"bad":1,"good":0})

    X = df["url_text"].astype(str)
    y = df["y"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=42)

    # featurization pipelines
    text_pipe = Pipeline([
        ("clean", CleanTextTransformer()),
        ("hash",  HashingVectorizer(n_features=2**10, alternate_sign=False))
    ])
    lex_pipe = URLLexicalFeatures()
    dom_pipe = DomainFeatureExtractor(whitelist=WHITELIST, vt_api_key=None)

    featurizer = FeatureUnion([
        ("text", text_pipe),
        ("lex",  lex_pipe),
        ("dom",  dom_pipe)
    ])

    # classifier (just logistic here; you can add XGB/LGBM back if you fix base_score)
    vote = VotingClassifier([("lr", LogisticRegression())], voting="soft", n_jobs=-1)

    # single pipeline
    url_pipeline = Pipeline([
        ("feats", featurizer),
        ("clf",   vote)
    ])

    url_pipeline.fit(Xtr, ytr)
    log.info(f"   → URL train: {url_pipeline.score(Xtr,ytr)*100:.1f}%   "
             f"test: {url_pipeline.score(Xte,yte)*100:.1f}%")

    os.makedirs(os.path.dirname(PATHS["out_url"]), exist_ok=True)
    joblib.dump(url_pipeline, PATHS["out_url"])
    log.info("   → Fast URL model saved.\n")

# ── 4) FILE malware model ──────────────────────────────────────────────────────
def train_file():
    log.info("🔹 Training File Malware Model…")
    ben = pd.read_csv(PATHS["ben_in"])
    mal = pd.read_csv(PATHS["mal_in"])
    ben["y"], mal["y"] = 0, 1
    df = pd.concat([ben, mal], ignore_index=True).dropna(subset=["y"])
    X_df = df.drop(columns=["y","hash"])   # drop non-numeric
    y     = df["y"]

    fe = FileFeatureExtractor()
    X_feats = fe.transform(X_df)

    Xtr, Xte, ytr, yte = train_test_split(X_feats, y, test_size=0.2,
                                          stratify=y, random_state=42)
    clf = LGBMClassifier(n_jobs=-1, random_state=42)
    clf.fit(Xtr, ytr)
    save_and_log(clf, Xtr, ytr, Xte, yte, PATHS["out_file"], "File")
    fe  = FileFeatureExtractor()
    X_feats = fe.transform(X_df)
    Xtr, Xte, ytr, yte = train_test_split(
        X_feats, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = LGBMClassifier(n_jobs=-1, random_state=42)
    clf.fit(Xtr, ytr)
   # save both the featurizer and the classifier as a tuple:
    os.makedirs(os.path.dirname(PATHS["out_file"]), exist_ok=True)
    joblib.dump((fe, clf), PATHS["out_file"])
    log.info("   → File malware model saved.\n")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_email()
    train_message_transformer()
    train_url()
    train_file()























