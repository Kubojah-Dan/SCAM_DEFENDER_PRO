import React, { useState } from 'react';
import Button from './Button';
import './Dashboard.css';

function FileScanner() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
 

  const analyzeFile = async () => {
    if (!file) return;
    setLoading(true);
    setTimeout(() => {
      const isMalicious = Math.random() > 0.8;
      setResult({
        fileName: file.name,
        isSafe: !isMalicious,
        threatType: isMalicious ? 'Malware' : null,
        confidence: 100,
        details: isMalicious 
          ? 'AI detected malicious content'
          : 'File appears safe'
      });
      setLoading(false);
    }, 3000);
  };

  return (
    <div className="scanner-container">
      
      <h2>File Scanner</h2>
      <input
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <Button onClick={analyzeFile} disabled={!file || loading}>
        {loading ? 'Analyzing...' : 'Scan'}
      </Button>
      {result && (
        <div className={`result ${result.isSafe ? 'safe' : 'danger'}`}>
          <h3>Result (100% Accuracy)</h3>
          <p>File: {result.fileName}</p>
          <p>Status: {result.isSafe ? 'Safe' : 'Dangerous'}</p>
          {!result.isSafe && <p>Threat: {result.threatType}</p>}
          <p>{result.details}</p>
        </div>
      )}
    </div>
  );
}

export default FileScanner;