def process_file(file_obj, model):
    """
    Process uploaded file. For demonstration, we read the file contents
    and check for known malicious patterns.
    """
    try:
        content = file_obj.read().decode('utf-8')
    except Exception as e:
        return "error", {"error": "Could not read file content", "details": str(e)}
    
    # You may add more advanced file scanning logic or even call an antivirus API here.
    # For now, we use a basic keyword-based approach.
    known_malicious_keywords = ["malware", "phishing", "scam"]
    score = sum(keyword in content.lower() for keyword in known_malicious_keywords)
    
    # If a trained model for files is provided, it can be used:
    if model is not None:
        prediction = model.predict([content])[0]
        probability = max(model.predict_proba([content])[0])
        details = {"score": score, "model_probability": float(probability)}
    else:
        # Fallback: if the content contains any malicious keywords
        prediction = 1 if score > 0 else 0  # 1 means malicious
        details = {"score": score}
    return int(prediction), details
