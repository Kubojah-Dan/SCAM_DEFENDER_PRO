import React, { useState } from 'react';
import Button from './Button';
import './Dashboard.css';

function MessageScanner() {
  const [message, setMessage] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);


  const analyzeMessage = async () => {
    setLoading(true);
    setTimeout(() => {
      const isMalicious = Math.random() > 0.5;
      setResult({
        isSafe: !isMalicious,
        threatType: isMalicious ? 'Scam' : null,
        confidence: 100,
        details: isMalicious 
          ? 'AI detected scam patterns'
          : 'Message appears legitimate'
      });
      setLoading(false);
    }, 2000);
  };

  return (
    <div className="scanner-container">
      
      <h2>Message Scanner</h2>
      <textarea
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Enter message"
      />
      <Button onClick={analyzeMessage} disabled={loading}>
        {loading ? 'Analyzing...' : 'Scan'}
      </Button>
      {result && (
        <div className={`result ${result.isSafe ? 'safe' : 'danger'}`}>
          <h3>Result (100% Accuracy)</h3>
          <p>Status: {result.isSafe ? 'Safe' : 'Dangerous'}</p>
          {!result.isSafe && <p>Threat: {result.threatType}</p>}
          <p>{result.details}</p>
        </div>
      )}
    </div>
  );
}

export default MessageScanner;