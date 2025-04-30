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
