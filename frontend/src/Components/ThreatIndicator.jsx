import React, { useState, useEffect } from 'react';
import { FiAlertTriangle, FiCheckCircle, FiX } from 'react-icons/fi';
import './Dashboard.css';

const ThreatIndicator = () => {
  const [threatLevel, setThreatLevel] = useState('green');
  const [detectedUrls, setDetectedUrls] = useState([]);
  const [notification, setNotification] = useState(null);

  // Simulate real-time threat detection
  useEffect(() => {
    const threatEvents = [
      { level: 'green', message: 'All systems secure' },
      { level: 'yellow', message: 'Suspicious domain detected: example.net', url: 'example.net' },
      { level: 'red', message: 'Phishing attempt blocked: scam-site.com', url: 'scam-site.com' },
      { level: 'red', message: 'Malware URL detected: bad-file-download.com', url: 'bad-file-download.com' }
    ];

    // Rotate through threat levels for demo purposes
    const interval = setInterval(() => {
      const randomEvent = threatEvents[Math.floor(Math.random() * threatEvents.length)];
      setThreatLevel(randomEvent.level);
      
      if (randomEvent.url) {
        setDetectedUrls(prev => [...prev.slice(-2), randomEvent]); // Keep last 2 threats
        setNotification(randomEvent);
      }
    }, 8000); // Change every 8 seconds for demo

    return () => clearInterval(interval);
  }, []);

  // In a real app, you would use browser APIs or extensions to monitor URLs
  // For example: chrome.webRequest.onBeforeRequest.addListener()

  const colors = {
    green: '#4CAF50',
    yellow: '#FFC107',
    red: '#F44336'
  };

  const dismissNotification = () => {
    setNotification(null);
  };

  return (
    <div className="threat-monitor">
      {/* Main threat indicator */}
      <div 
        className="threat-indicator" 
        style={{ background: colors[threatLevel] }}
      >
        <div className={`pulse-animation ${threatLevel}`} />
        <span>
          {threatLevel === 'green' ? 'All systems secure' : 
           threatLevel === 'yellow' ? 'Suspicious activity detected' : 'CRITICAL THREAT DETECTED!'}
        </span>
      </div>

      {/* Real-time notification popup */}
      {notification && (
        <div className={`threat-notification ${notification.level}`}>
          <div className="notification-header">
            {notification.level === 'red' ? (
              <FiAlertTriangle className="notification-icon" />
            ) : (
              <FiCheckCircle className="notification-icon" />
            )}
            <h4>
              {notification.level === 'red' ? 'Security Alert' : 'Security Notice'}
            </h4>
            <button onClick={dismissNotification} className="dismiss-btn">
              <FiX />
            </button>
          </div>
          <p>{notification.message}</p>
          {notification.url && (
            <div className="threat-details">
              <span>Blocked URL:</span>
              <a 
                href={`https://${notification.url}`} 
                target="_blank" 
                rel="noopener noreferrer"
                className="blocked-url"
              >
                {notification.url}
              </a>
            </div>
          )}
        </div>
      )}

      {/* Recent threats list */}
      {detectedUrls.length > 0 && (
        <div className="recent-threats">
          <h4>Recent Threats Blocked</h4>
          <ul>
            {detectedUrls.map((threat, index) => (
              <li key={index} className={threat.level}>
                <span className="threat-level-badge">{threat.level}</span>
                {threat.url}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default ThreatIndicator;