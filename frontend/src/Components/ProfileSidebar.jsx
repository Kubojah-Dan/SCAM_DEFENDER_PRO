import React, { useState } from 'react';
import { FiMenu, FiX, FiUser, FiLogOut, FiSettings, FiHelpCircle, FiChevronRight, FiChevronLeft } from 'react-icons/fi';
import Button from './Button';
import './Dashboard.css';

const ProfileSidebar = ({ user, onLogout, onNavigate }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeView, setActiveView] = useState('menu'); // 'menu', 'account', 'settings'
  const [currentSetting, setCurrentSetting] = useState(null);

  const handleMenuItemClick = (item) => {
    if (item === 'help') {
      // Use onNavigate for help page if provided, otherwise default behavior
      if (onNavigate) {
        onNavigate('/help-center');
      } else {
        window.location.href = '/help-center';
      }
      return;
    }
    
    setActiveView(item);
    
    // Call onNavigate with the item if provided
    if (onNavigate) {
      onNavigate(item);
    }
  };

  const handleBackToMenu = () => {
    setActiveView('menu');
    setCurrentSetting(null);
  };

  const handleLogout = () => {
    if (onLogout) {
      onLogout();
    }
    setIsOpen(false);
  };

  const settingsOptions = [
    { id: 'notifications', label: 'Notification Preferences' },
    { id: 'privacy', label: 'Privacy Settings' },
    { id: 'security', label: 'Security Options' },
    { id: 'theme', label: 'Theme Preferences' }
  ];

  return (
    <>
      {/* Overlay when sidebar is open */}
      {isOpen && (
        <div 
          className="sidebar-overlay"
          onClick={() => {
            setIsOpen(false);
            setActiveView('menu');
            setCurrentSetting(null);
          }}
        />
      )}

      <div className={`profile-sidebar ${isOpen ? 'open' : ''}`}>
        <button 
          className="menu-toggle"
          onClick={() => {
            setIsOpen(!isOpen);
            if (!isOpen) {
              setActiveView('menu');
              setCurrentSetting(null);
            }
          }}
          aria-label={isOpen ? "Close menu" : "Open menu"}
        >
          {isOpen ? <FiX size={24} /> : <FiMenu size={24} />}
        </button>

        <div className="sidebar-content">
          {activeView === 'menu' ? (
            <>
              <div className="user-info">
                <div className="user-details">
                  <h4>{user.name || 'User'}</h4>
                  <p>{user.email}</p>
                  {user.createdAt && (
                    <small>Member since: {new Date(user.createdAt).toLocaleDateString()}</small>
                  )}
                </div>
              </div>
              
              <nav className="sidebar-nav">
                <button
                  className="nav-item"
                  onClick={() => handleMenuItemClick('account')}
                >
                  <FiUser className="nav-icon" />
                  <span>My Account</span>
                  <FiChevronRight className="nav-arrow" />
                </button>
                
                <button
                  className="nav-item"
                  onClick={() => handleMenuItemClick('settings')}
                >
                  <FiSettings className="nav-icon" />
                  <span>Settings</span>
                  <FiChevronRight className="nav-arrow" />
                </button>
                
                <button
                  className="nav-item"
                  onClick={() => handleMenuItemClick('help')}
                >
                  <FiHelpCircle className="nav-icon" />
                  <span>Help Center</span>
                  <FiChevronRight className="nav-arrow" />
                </button>
                
                <button 
                  className="nav-item logout"
                  onClick={handleLogout}
                >
                  <FiLogOut className="nav-icon" />
                  <span>Logout</span>
                </button>
              </nav>
            </>
          ) : activeView === 'account' ? (
            <div className="account-details-view">
              <button className="back-button" onClick={handleBackToMenu}>
                <FiChevronLeft /> Back to Menu
              </button>
              
              <h3>Account Details</h3>
              
              <div className="account-avatar-container">
                <img 
                  src={user.photoURL || '/default-avatar.png'} 
                  alt={user.name || 'User'} 
                  className="account-avatar"
                />
              </div>
              
              <div className="account-detail">
                <label>Full Name:</label>
                <p>{user.name || 'Not provided'}</p>
              </div>
              
              <div className="account-detail">
                <label>Email:</label>
                <p>{user.email}</p>
              </div>
              
              {user.role && (
                <div className="account-detail">
                  <label>Account Type:</label>
                  <p>{user.role}</p>
                </div>
              )}
              
              {user.createdAt && (
                <div className="account-detail">
                  <label>Member Since:</label>
                  <p>{new Date(user.createdAt).toLocaleDateString()}</p>
                </div>
              )}
              
              <Button className="edit-profile-btn">
                Edit Profile Information
              </Button>
            </div>
          ) : activeView === 'settings' && !currentSetting ? (
            <div className="settings-view">
              <button className="back-button" onClick={handleBackToMenu}>
                <FiChevronLeft /> Back to Menu
              </button>
              
              <h3>Settings</h3>
              
              <div className="settings-options">
                {settingsOptions.map(setting => (
                  <button
                    key={setting.id}
                    className="settings-option"
                    onClick={() => setCurrentSetting(setting.id)}
                  >
                    <span>{setting.label}</span>
                    <FiChevronRight />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="specific-setting-view">
              <button className="back-button" onClick={() => setCurrentSetting(null)}>
                <FiChevronLeft /> Back to Settings
              </button>
              
              <h3>{settingsOptions.find(s => s.id === currentSetting)?.label}</h3>
              
              <div className="setting-content">
                {/* Settings content remains the same */}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default ProfileSidebar;