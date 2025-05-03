import re
import requests
import numpy as np
from app.utils.helpers import get_vpn_session

def normalize_url(url_text):
    """
    Normalizes the URL by adding http:// if no protocol is present.
    We assume that if a URL does not start with http or https but passes regex, it is a valid domain.
    """
    if not re.match(r'^https?://', url_text):
        url_text = "http://" + url_text
    return url_text

def call_virustotal(url, api_key, proxies=None):
    """
    Integrate with VirusTotal API to scan the url.
    For demonstration, we simulate the API call. In production, use VirusTotal’s official API.
    """
    vt_url = "https://www.virustotal.com/api/v3/urls"
    headers = {"x-apikey": api_key}
    # In practice, you need to encode the url and send a POST request.
    # Below is a dummy response.
    response = {"data": {"attributes": {"last_analysis_stats": {"malicious": 0, "harmless": 10}}}}
    return response

def process_url(url_text, model, config):
    """
    Processes the URL input and returns a prediction.
    It normalizes the URL, uses VPN-based session for external API calls, and checks with VirusTotal.
    """
    normalized = normalize_url(url_text)
    
    # Optional: scan the url using VirusTotal API
    vt_api_key = config.get("virustotal_api_key", "")
    proxies = config.get("vpn_proxies", None)
    session = get_vpn_session(proxies)
    vt_result = call_virustotal(normalized, vt_api_key, proxies)
    
    # Get model prediction (e.g., simple ML model for URL features)
    prediction = model.predict([normalized])[0]
    probability = np.max(model.predict_proba([normalized]))
    
    details = {
        "normalized_url": normalized,
        "vt_scan": vt_result,
        "probability": float(probability)
    }
    return int(prediction), details
