import ssl, socket
import whois, requests, tldextract
import pandas as pd
from datetime import datetime
from urllib.parse import urlparse
from sklearn.base import TransformerMixin
from joblib import Memory
import Levenshtein

# Cache setup: cache expensive transform calls
CACHE_DIR = './.cache_domain_feats'
memory = Memory(CACHE_DIR, verbose=0)

def extract_domain(url: str) -> str:
    # faster TLD+1 extraction
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

def is_typosquat(domain: str, whitelist: list[str], max_dist: int = 1) -> bool:
    return any(Levenshtein.distance(domain, legit) <= max_dist for legit in whitelist)

def domain_age_days(domain: str) -> int | None:
    try:
        w = whois.whois(domain)
        created = w.creation_date
        if isinstance(created, list): created = created[0]
        return (datetime.now() - created).days if created else None
    except:
        return None

_vt_session = requests.Session()
def vt_domain_reputation(domain: str, api_key: str) -> dict:
    if not api_key:
        return {"reputation": None, "malicious_votes": 0}
    try:
        r = _vt_session.get(
            f"https://www.virustotal.com/api/v3/domains/{domain}",
            headers={"x-apikey": api_key}, timeout=5
        ); r.raise_for_status()
        data = r.json().get("data", {}).get("attributes", {})
        return {"reputation": data.get("reputation"),
                "malicious_votes": data.get("total_votes",{}).get("malicious", 0)}
    except:
        return {"reputation": None, "malicious_votes": 0}

def cert_age_days(domain: str) -> int | None:
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
            s.settimeout(5); s.connect((domain, 443))
            cert = s.getpeercert()
        nb = datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z')
        return (datetime.now() - nb).days
    except:
        return None

class DomainFeatureExtractor(TransformerMixin):
    """Cached, vectorized domain feature extractor."""
    def __init__(self, whitelist=None, vt_api_key=None):
        self.whitelist = whitelist or []
        self.vt_api_key = vt_api_key

    def fit(self, X, y=None): return self

    @memory.cache  # cache results for identical URL lists
    def transform(self, X, y=None):
        # build DataFrame once, no printing
        df = pd.DataFrame({'url': X})
        df['domain'] = df['url'].map(extract_domain)
        df['is_typosquat'] = df['domain'].map(lambda d: int(is_typosquat(d, self.whitelist)))
        df['age_days']     = df['domain'].map(domain_age_days).fillna(-1)
        vt = df['domain'].map(lambda d: vt_domain_reputation(d, self.vt_api_key))
        df['vt_reputation']     = vt.map(lambda r: r['reputation']).fillna(-1)
        df['vt_malicious_votes'] = vt.map(lambda r: r['malicious_votes'])
        df['cert_age_days'] = df['domain'].map(cert_age_days).fillna(-1)
        return df[['is_typosquat','age_days','vt_reputation',
                   'vt_malicious_votes','cert_age_days']].to_numpy()




