import datetime, os, sys, pandas as pd, joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required

# ensure root and app bundle are on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from app.utils.clean_text      import clean_url
from app.utils.feature_eng     import FileFeatureExtractor
from app.utils.domain_features import DomainFeatureExtractor

app = Flask(__name__)
CORS(app)
app.config["JWT_SECRET_KEY"] = "super-secret-change-me"
jwt = JWTManager(app)

# ── load models ───────────────────────────────────────────────────────────────
# build absolute paths so that working directory doesn't matter
def load_joblib(path_parts):
    path = os.path.join(ROOT, *path_parts)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)

EMAIL_PIPE = load_joblib(["app", "models", "email_pipeline.pkl"])
MSG_PIPE   = load_joblib(["app", "models", "message_model.pkl"])
URL_PIPE   = load_joblib(["app", "models", "url_adv_pipeline_fast.pkl"])
FE_PIPE, FILE_PIPE = load_joblib(["app", "models", "file_malware_pipeline.pkl"])

# ── auth endpoints (unchanged) ─────────────────────────────────────────────────
USERS = {}
@app.route("/auth/register", methods=["POST"])
def register():
    u,p = request.json.get("user"), request.json.get("pass")
    if u in USERS: return jsonify(msg="User exists"),400
    USERS[u]=p
    return jsonify(msg="Registered"),200

@app.route("/auth/register", methods=["POST"])
def register():
     data = request.json or {}
     email         = data.get("email")
     password      = data.get("password")
     confirm_pass  = data.get("confirm_password")
     if not email or not password or not confirm_pass:
         return jsonify(msg="email, password and confirm_password are required"), 400
     if password != confirm_pass:
         return jsonify(msg="password and confirm_password must match"), 400
     if email in USERS:
         return jsonify(msg="User exists"), 400
     USERS[email] = password
     return jsonify(msg="Registered"), 200

@app.route("/auth/login", methods=["POST"])
def login():
    u,p = request.json.get("user"), request.json.get("pass")
    if USERS.get(u)!=p: return jsonify(msg="Bad credentials"),401
    token = create_access_token(identity=u)
    return jsonify(access_token=token),200

@app.route("/auth/login", methods=["POST"])
def login():
     data = request.json or {}
     email    = data.get("email")
     password = data.get("password")
     if not email or not password:
         return jsonify(msg="email and password required"), 400
     if USERS.get(email) != password:
         return jsonify(msg="Bad credentials"), 401
     token = create_access_token(identity=email)
     return jsonify(access_token=token), 200

# ── prediction endpoints ───────────────────────────────────────────────────────
@app.route("/predict/email", methods=["POST"])
@jwt_required() # 192.168.101.102:8080/predit/email
def predict_email():
    # allow both subject and message, but only require at least one
    subj = request.json.get("subject", "") or ""
    body = request.json.get("message", "") or ""
    if not isinstance(subj, str) or not isinstance(body, str):
        return jsonify(msg="subject and message must be strings"),422
    # combine for classification
    txt = (subj + " \n" + body).strip()
    pred = EMAIL_PIPE.predict([txt])[0]
    conf = EMAIL_PIPE.predict_proba([txt])[0].max()*100
    return jsonify(prediction=("Spam" if pred==1 else "Ham"),
                   confidence=f"{conf:.1f}%", timestamp=datetime.datetime.utcnow().isoformat())

@app.route("/predict/message", methods=["POST"])
@jwt_required()
def predict_message():
    txt = request.json.get("message_text", "")
    if not isinstance(txt, str):
        return jsonify(msg="message_text must be a string"),422
    pred = MSG_PIPE.predict([txt])[0]
    conf = MSG_PIPE.predict_proba([txt])[0].max()*100
    return jsonify(prediction=("Scam" if pred==1 else "Safe"),
                   confidence=f"{conf:.1f}%", timestamp=datetime.datetime.utcnow().isoformat())

@app.route("/predict/url", methods=["POST"])
@jwt_required()
def predict_url():
    url = request.json.get("url", "")
    if not isinstance(url, str):
        return jsonify(msg="url must be a string"),422
    # pipeline handles featurization + classification
    pred = URL_PIPE.predict([url])[0]
    conf = URL_PIPE.predict_proba([url])[0].max()*100
    return jsonify(
      prediction=("Scam" if pred==1 else "Safe"),
      confidence=f"{conf:.1f}%",
      timestamp=datetime.datetime.utcnow().isoformat()
    )

@app.route("/predict/file", methods=["POST"])
@jwt_required()
def predict_file():
    # 1) Make sure a file was uploaded
    if 'file' not in request.files:
        return jsonify(msg="No file part in request"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(msg="No selected file"), 400

    # 2) Read the bytes and compute simple metadata features
    content = file.read()
    file_size = len(content)
    _, ext = os.path.splitext(file.filename)
    # compute byte‐entropy
    from collections import Counter
    import math
    counts = Counter(content)
    entropy = 0.0
    if file_size > 0:
        for cnt in counts.values():
            p = cnt / file_size
            entropy -= p * math.log2(p)

    # 3) Build a DataFrame for FE_PIPE
    feat_dict = {
        "file_size": file_size,
        "file_extension": ext.lower(),
        "byte_entropy": entropy
    }
    df = pd.DataFrame([feat_dict])

    # 4) Transform & predict
    X = FE_PIPE.transform(df)
    pred = FILE_PIPE.predict(X)[0]
    conf = FILE_PIPE.predict_proba(X)[0][int(pred)] * 100

    return jsonify(
        prediction=("Malware" if pred == 1 else "Benign"),
        confidence=f"{conf:.1f}%",
        timestamp=datetime.datetime.utcnow().isoformat()
    )
    
if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
















