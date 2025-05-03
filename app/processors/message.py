import numpy as np

def process_message(text, model):
    """
    Process message text to classify as ham (good) or spam (malicious).
    Model is trained with labels: 'ham' for good, 'spam' for malicious.
    """
    # Preprocess text (lowercase etc.)
    processed_text = text.lower().strip()
    
    prediction = model.predict([processed_text])[0]
    probability = np.max(model.predict_proba([processed_text]))
    
    details = {"probability": float(probability)}
    return prediction, details
