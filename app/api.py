import sys
import os
from flask import Flask, request, jsonify
import joblib
import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict/*": {"origins": "*"}}, supports_credentials=True)  # Enable CORS for integration with the frontend

# Import the utility module (so the transformer is available for unpickling)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.utils.clean_text import CleanTextTransformer

# Load pipelines (these now include our custom transformer)
email_pipeline = joblib.load("models/email_pipeline.pkl")
message_pipeline = joblib.load("models/message_pipeline.pkl")
url_pipeline = joblib.load("models/url_pipeline.pkl")

@app.route('/')
def home():
    return jsonify({"message": "Welcome to Scam Detection API!"})

@app.route('/predict/email', methods=['POST'])
def predict_email():
    data = request.get_json() or {}
    subject = data.get("subject", "")
    body = data.get("body", "")
    url_flag = data.get("url", "0")
    combined_text = f"{subject} {body} {'hasurl' if str(url_flag).strip()=='1' else 'nourl'}"
    
    proba = email_pipeline.predict_proba([combined_text])[0]
    prediction = email_pipeline.predict([combined_text])[0]
    confidence = round(max(proba) * 100, 2)
    
    response = {
        "status": "Scam Detected" if prediction == "1" else "Safe Message",
        "prediction": "Scam" if prediction == "1" else "Safe",
        "confidence": f"{confidence}%",
        "message": ("⚠️ This email appears to be a scam. Please avoid clicking on suspicious links."
                    if prediction == "1" else "✅ This email appears safe. However, always be cautious."),
        "timestamp": datetime.datetime.now().isoformat(),
        "tips": "Never share your personal or financial information via email."
    }
    return jsonify(response)

@app.route('/predict/message', methods=['POST'])
def predict_message():
    data = request.get_json() or {}
    text = data.get("message_text", "")
    proba = message_pipeline.predict_proba([text])[0]
    prediction = message_pipeline.predict([text])[0]
    confidence = round(max(proba) * 100, 2)
    
    response = {
        "status": "Scam Detected" if prediction == "1" else "Safe Message",
        "prediction": "Scam" if prediction == "1" else "Safe",
        "confidence": f"{confidence}%",
        "message": ("⚠️ This message appears to be a scam. Avoid clicking on embedded links."
                    if prediction == "1" else "✅ This message appears safe."),
        "timestamp": datetime.datetime.now().isoformat(),
        "tips": "Be cautious when receiving unexpected messages."
    }
    return jsonify(response)

@app.route('/predict/url', methods=['POST'])
def predict_url():
    data = request.get_json() or {}
    text = data.get("url_text", "")
    proba = url_pipeline.predict_proba([text])[0]
    prediction = url_pipeline.predict([text])[0]
    confidence = round(max(proba) * 100, 2)
    
    response = {
        "status": "Scam Detected" if prediction == "1" else "Safe URL",
        "prediction": "Scam" if prediction == "1" else "Safe",
        "confidence": f"{confidence}%",
        "message": ("⚠️ This URL appears suspicious. Do not click on it."
                    if prediction == "1" else "✅ This URL appears safe."),
        "timestamp": datetime.datetime.now().isoformat(),
        "tips": "Always verify the legitimacy of URLs before clicking."
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)







