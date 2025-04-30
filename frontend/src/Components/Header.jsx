import React from 'react';
import './Dashboard';

export default function Header() {
  return (
    <header className="dashboard-header">
      <div className="header-content">
        <h1>SHIELDX</h1>
        <div className="user-profile">
          <span></span>
          <div className="avatar"> X </div>
        </div>
      </div>
    </header>
  );
}