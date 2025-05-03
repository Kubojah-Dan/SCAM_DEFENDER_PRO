import sqlite3
import datetime, os, sys, pandas as pd, joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

from app.utils.clean_text      import clean_url
from app.utils.feature_eng     import FileFeatureExtractor
from app.utils.domain_features import DomainFeatureExtractor
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# note: message and email pipelines already include their own transforms

app = Flask(__name__)
CORS(app)
app.config["JWT_SECRET_KEY"] = "The-vigilant-four"
jwt = JWTManager(app)

# ── load models ───────────────────────────────────────────────────────────────
EMAIL_PIPE = joblib.load("app/models/email_pipeline.pkl")
model_path = os.path.join(ROOT, "app", "models", "message_model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Expected model at {model_path}")
MSG_PIPE = joblib.load(model_path)

URL_PIPE = joblib.load("app/models/url_adv_pipeline_fast.pkl")
FE_PIPE, FILE_PIPE = joblib.load("app/models/file_malware_pipeline.pkl")  # if you dumped (fe,clf)

DB_PATH = os.path.join(ROOT, "app", "data", "predictions.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    return conn

def log_prediction(model, inp, pred, prob):
    c = get_db().cursor()
    c.execute(
      "INSERT INTO predictions(model,input_text,predicted,probability) VALUES (?,?,?,?)",
      (model, inp, pred, prob)
    )
    c.connection.commit()
    c.connection.close()


# ── auth endpoints (unchanged) ─────────────────────────────────────────────────
USERS = {}
@app.route("/auth/register", methods=["POST"])
def register():
    u,p = request.json.get("user"), request.json.get("pass")
    if u in USERS: return jsonify(msg="User exists"),400
    USERS[u]=p
    return jsonify(msg="Registered"),200

@app.route("/auth/login", methods=["POST"])
def login():
    u,p = request.json.get("user"), request.json.get("pass")
    if USERS.get(u)!=p: return jsonify(msg="Bad credentials"),401
    token = create_access_token(identity=u)
    return jsonify(access_token=token),200

@app.route("/admin/clear_predictions", methods=["POST"])
def clear_db():
    secret = request.headers.get("X-Admin-Secret","")
    if secret != app.config.get("ADMIN_SECRET","change-me"):
        return jsonify(msg="Forbidden"),403
    conn = get_db()
    conn.execute("DELETE FROM predictions;")
    conn.commit()
    conn.close()
    return jsonify(msg="Cleared"),200


# ── prediction endpoints ───────────────────────────────────────────────────────
@app.route("/predict/email", methods=["POST"])
#@jwt_required()
def predict_email():
    txt = request.json.get("message","")
    if not isinstance(txt,str):
        return jsonify(msg="message must be a string"),422
    pred = EMAIL_PIPE.predict([txt])[0]
    conf = EMAIL_PIPE.predict_proba([txt])[0].max()*100
    return jsonify(prediction=("Spam" if pred==1 else "Ham"),
                   confidence=f"{conf:.1f}%", timestamp=datetime.datetime.utcnow().isoformat())

@app.route("/predict/message", methods=["POST"])
#@jwt_required()
def predict_message():
    txt = request.json.get("message_text","")
    if not isinstance(txt,str):
        return jsonify(msg="message_text must be a string"),422
    pred = MSG_PIPE.predict([txt])[0]
    conf = MSG_PIPE.predict_proba([txt])[0].max()*100
    return jsonify(prediction=("Scam" if pred==1 else "Safe"),
                   confidence=f"{conf:.1f}%", timestamp=datetime.datetime.utcnow().isoformat())


@app.route("/predict/url", methods=["POST"])
#@jwt_required()
def predict_url():
    url = request.json.get("url","")
    if not isinstance(url,str):
        return jsonify(msg="url must be a string"),422

    # the pipeline does both featurization + classification
    pred = URL_PIPE.predict([url])[0]
    conf = URL_PIPE.predict_proba([url])[0].max()*100
    proba = URL_PIPE.predict_proba([url])[0][1]
    pred  = 1 if proba>0.3 else 0   # custom threshold
    conf  = proba*100
    return jsonify(
      prediction=("Scam" if pred==1 else "Safe"),
      confidence=f"{conf:.1f}%",
      timestamp=datetime.datetime.utcnow().isoformat()
    )

@app.route("/predict/file", methods=["POST"])
#@jwt_required()
def predict_file():
    feat = request.get_json() or {}
    df   = pd.DataFrame([feat])
    X    = FE_PIPE.transform(df)
    pred = FILE_PIPE.predict(X)[0]
    conf = FILE_PIPE.predict_proba(X)[0][int(pred)]*100
    return jsonify(prediction=("Malware" if pred==1 else "Benign"),
                   confidence=f"{conf:.1f}%", timestamp=datetime.datetime.utcnow().isoformat())

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)















