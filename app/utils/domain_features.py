import re
import ssl
import socket
import json
import whois
import requests
import pandas as pd
from urllib.parse import urlparse
import Levenshtein

# 1. Typosquat detection against a whitelist
def extract_domain(url):
    """Extract netloc (domain) from URL, stripping www."""
    parsed = urlparse(url if url.startswith('http') else 'http://' + url)
    domain = parsed.netloc.lower()
    return domain[4:] if domain.startswith('www.') else domain

def is_typosquat(domain, whitelist, max_dist=1):
    """Return True if domain is within max_dist of any whitelist entry."""
    return any(Levenshtein.distance(domain, legit) <= max_dist for legit in whitelist)

# 2. Domain age (WHOIS)
from datetime import datetime
def domain_age_days(domain):
    """Return domain age in days, or None on failure."""
    try:
        w = whois.whois(domain)
        created = w.creation_date
        if isinstance(created, list): created = created[0]
        if not created: return None
        return (datetime.now() - created).days
    except Exception:
        return None

# 3. VirusTotal reputation lookup
def vt_domain_reputation(domain, api_key):
    """Return reputation score and malicious vote count from VT."""
    url = f"https://www.virustotal.com/api/v3/domains/{domain}"
    headers = {"x-apikey": api_key}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json().get('data', {}).get('attributes', {})
        return {
            "reputation": data.get("reputation"),
            "malicious_votes": data.get("total_votes",{}).get("malicious", 0)
        }
    except Exception:
        return {"reputation": None, "malicious_votes": 0}

# 4. SSL certificate age
def cert_age_days(domain):
    """Return SSL cert age in days, or None on failure."""
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
            s.settimeout(5)
            s.connect((domain, 443))
            cert = s.getpeercert()
        not_before = datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z')
        return (datetime.now() - not_before).days
    except Exception:
        return None

class DomainFeatureExtractor:
    """
    Transformer for scikit-learn pipelines that computes:
      - is_typosquat
      - domain_age_days
      - vt_reputation
      - vt_malicious_votes
      - cert_age_days
    """
    def __init__(self, whitelist, vt_api_key):
        self.whitelist = whitelist
        self.vt_api_key = vt_api_key

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # X is array-like of URLs
        df = pd.DataFrame({'url': X})
        df['domain'] = df['url'].apply(extract_domain)
        df['is_typosquat'] = df['domain'].apply(lambda d: is_typosquat(d, self.whitelist))
        df['age_days'] = df['domain'].apply(domain_age_days)
        vt = df['domain'].apply(lambda d: vt_domain_reputation(d, self.vt_api_key))
        df['vt_reputation'] = vt.apply(lambda r: r['reputation'])
        df['vt_malicious_votes'] = vt.apply(lambda r: r['malicious_votes'])
        df['cert_age_days'] = df['domain'].apply(cert_age_days)
        # Fill NaNs and convert booleans to ints
        df = df.fillna(-1)
        df['is_typosquat'] = df['is_typosquat'].astype(int)
        return df[['is_typosquat','age_days','vt_reputation','vt_malicious_votes','cert_age_days']].values
