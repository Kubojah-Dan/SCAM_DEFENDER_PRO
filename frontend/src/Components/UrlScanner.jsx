import React, { useState } from 'react';
import Button from './Button';
import './Dashboard.css';

export default function UrlScanner() {
  const [url, setUrl] = useState('');
  const [result, setResult] = useState(null);

  const handleScan = () => {
    // Mock scan result
    setResult({
      safe: Math.random() > 0.5,
      details: ['No malware detected', 'Valid SSL certificate']
    });
  };

  return (
    <div className="scanner">
      <h2>URL Scanner</h2>
      <div className="scanner-input">
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Enter URL to scan"
        />
        <Button onClick={handleScan}>Scan</Button>
      </div>
      {result && (
        <div className={`result ${result.safe ? 'safe' : 'unsafe'}`}>
          {result.safe ? '✅ Safe' : '⚠️ Unsafe'}
        </div>
      )}
    </div>
  );
}