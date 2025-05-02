import sys, os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, datetime, pandas as pd

# make sure app/ is importable
sys.path.append(os.path.dirname(__file__))

from app.utils.clean_text      import clean_url
from app.utils.domain_features import DomainFeatureExtractor

# Flask setup
app = Flask(__name__)
CORS(app)

# load pipelines
EMAIL_PIPELINE    = joblib.load(os.path.join('app','models','email_pipeline.pkl'))
MESSAGE_PIPELINE  = joblib.load(os.path.join('app','models','message_pipeline.pkl'))
URL_PIPELINE_PATH = os.path.join('app','models','url_pipeline_fast.pkl')
char_vect, dom_pipe, URL_CLF = joblib.load(URL_PIPELINE_PATH)

@app.route('/')
def home():
    return jsonify({"message":"Scam Detection API is up"})

@app.route('/predict/email', methods=['POST'])
def predict_email():
    data = request.get_json() or {}
    subj = data.get('subject','')
    body = data.get('body','')
    urlf = data.get('url', '0')
    txt  = f"{subj} {body} {'hasurl' if str(urlf)=='1' else 'nourl'}"
    proba = EMAIL_PIPELINE.predict_proba([txt])[0]
    pred  = EMAIL_PIPELINE.predict([txt])[0]
    conf  = max(proba)*100
    return jsonify({
        "prediction": "Scam" if pred=="1" else "Safe",
        "confidence": f"{conf:.2f}%",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

@app.route('/predict/message', methods=['POST'])
def predict_message():
    data = request.get_json() or {}
    txt  = data.get('message_text','')
    proba = MESSAGE_PIPELINE.predict_proba([txt])[0]
    pred  = MESSAGE_PIPELINE.predict([txt])[0]
    conf  = max(proba)*100
    return jsonify({
        "prediction": "Scam" if pred=="1" else "Safe",
        "confidence": f"{conf:.2f}%",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

@app.route('/predict/url', methods=['POST'])
def predict_url():
    data = request.get_json() or {}
    url  = data.get('url','')
    clean = clean_url(url) or "empty"
    Xc    = char_vect.transform([clean])
    dom_f = dom_pipe.transform([url])
    X_all = pd.DataFrame((Xc.toarray()[0].tolist() + dom_f[0].tolist(),))
    pred  = URL_CLF.predict(X_all)[0]
    proba = URL_CLF.predict_proba(X_all)[0][int(pred)]*100
    return jsonify({
        "prediction": "Scam" if pred!="0" else "Safe",
        "confidence": f"{proba:.2f}%",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)









