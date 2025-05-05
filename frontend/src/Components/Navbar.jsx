import React from 'react';
import './Dashboard.css';

export default function Navbar({ activeTab, setActiveTab }) {
  return (
    <nav className="scanner-nav">
      <button 
        className={activeTab === 'url' ? 'active' : ''}
        onClick={() => setActiveTab('url')}
      >
        URL Scanner
      </button>
      <button
        className={activeTab === 'file' ? 'active' : ''}
        onClick={() => setActiveTab('file')}
      >
        File Scanner
      </button>
      <button
        className={activeTab === 'email' ? 'active' : ''}
        onClick={() => setActiveTab('email')}
      >
        Email Scam detect 
      </button>
      <button
        className={activeTab === 'message' ? 'active' : ''}
        onClick={() => setActiveTab('message')}
       >
        Message Scanner
       </button>
    </nav>
  );
}