import React from 'react';
import './theme.css'; // Create this CSS file for header-specific styles

export default function Header({ toggleSidebar, user }) {
  return (
    <header className="dashboard-header">
      <div className="header-content">
        <div className="header-left">
          <button className="menu-toggle" onClick={toggleSidebar}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M3 12H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M3 6H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M3 18H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <h1 className="logo">SHIELDX</h1>
        </div>
        
        <div className="user-profile" onClick={toggleSidebar}>
          <span className="username">{user?.name || 'User'}</span>
          <div className="avatar">
            {user?.avatar ? (
              <img src={user.avatar} alt="User avatar" />
            ) : (
              <span>{user?.name?.charAt(0) || 'X'}</span>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}