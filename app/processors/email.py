import numpy as np

def process_email(data, model):
    """
    Process email data to classify as scam or not scam.
    Expected input data (JSON) should include:
      - "subject"
      - "body"
      - "url" (optional; defaults to "0" if not provided)
    
    The function combines these fields into a single text string used for prediction.
    """
    subject = data.get("subject", "")
    body = data.get("body", "")
    url_flag = data.get("url", "0")  # Expect "1" if URL is present, otherwise "0"
    
    # Create a combined text feature matching the training pipeline
    combined_text = subject + " " + body + " " + ("hasurl" if str(url_flag).strip() == "1" else "nourl")
    processed_text = combined_text.lower().strip()
    
    prediction = model.predict([processed_text])[0]
    probability = np.max(model.predict_proba([processed_text]))
    
    details = {"probability": float(probability)}
    return int(prediction), details

