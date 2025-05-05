import React, { useEffect, useState } from 'react';
import { FiX, FiAlertTriangle, FiCheckCircle } from 'react-icons/fi';
import './Dashboard.css';
import Button from './Button';

const AlertNotification = ({ alerts, onActionClick }) => {
  const [activeAlerts, setActiveAlerts] = useState([]);

  useEffect(() => {
    if (alerts.length > 0) {
      // Display new alert
      setActiveAlerts(prev => [alerts[0], ...prev]);
      
      // Auto-remove after 5 seconds
      const timer = setTimeout(() => {
        setActiveAlerts(prev => prev.slice(0, -1));
      }, 5000);

      return () => clearTimeout(timer);
    }
  }, [alerts]);

  const handleDismiss = (id) => {
    setActiveAlerts(prev => prev.filter(alert => alert.id !== id));
  };

  const handleActionClick = (alert) => {
    if (onActionClick) {
      onActionClick(alert);
    }
    // Optionally dismiss after action
    handleDismiss(alert.id);
  };

  return (
    <div className="alert-notification-container">
      {activeAlerts.map(alert => (
        <div 
          key={alert.id}
          className={`alert-notification ${alert.type}`}
        >
          <div className="alert-icon">
            {alert.type === 'phishing' ? <FiAlertTriangle /> : <FiCheckCircle />}
          </div>
          <div className="alert-content">
            <h4>{alert.type === 'phishing' ? 'Scam Detected' : 'Safe'}</h4>
            <p>{alert.text}</p>
            <span className="alert-time">{alert.time}</span>
            
            {/* Added action button for phishing alerts */}
            {alert.type === 'phishing' && (
              <Button 
                className="alert-action-btn"
                onClick={() => handleActionClick(alert)}
              >
                Secure Account
              </Button>
            )}
          </div>
          <button 
            className="dismiss-btn"
            onClick={() => handleDismiss(alert.id)}
          >
            <FiX />
          </button>
        </div>
      ))}
    </div>
  );
};

export default AlertNotification;