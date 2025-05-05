import React, { useState } from 'react';
import Header from './Header';
import Navbar from './Navbar';
import UrlScanner from './UrlScanner';
import FileScanner from './FileScanner';
import EmailScanner from './EmailScanner';
import MessageScanner from './MessageScanner';
import ThreatIndicator from './ThreatIndicator';
import AlertNotification from './AlertNotification';
import ProfileSidebar from './ProfileSidebar';
import './Dashboard.css';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('url');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [user] = useState({
    name: "Lina",
    email: "sowmyalina@gmail.com",
    role: "Security Analyst"
  });

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    // Close sidebar when switching to a non-profile tab
    if (tab !== 'profile') {
      setSidebarOpen(false);
    }
  };

  return (
    <div className={`dashboard ${sidebarOpen ? 'sidebar-open' : ''}`}>
      <ProfileSidebar 
        user={user} 
        isOpen={sidebarOpen} 
        toggleSidebar={toggleSidebar} 
      />
      
      <div className="main-content">
        <Header 
          toggleSidebar={toggleSidebar} 
          user={user} 
          activeTab={activeTab}
        />
        
        <Navbar 
          activeTab={activeTab} 
          setActiveTab={handleTabChange} 
        />
        
        <div className="content-area">
          {activeTab === 'url' && <UrlScanner />}
          {activeTab === 'file' && <FileScanner />}
          {activeTab === 'email' && <EmailScanner />}
          {activeTab === 'message' && <MessageScanner />}
          {activeTab === 'threatindicator' && <ThreatIndicator />}
          {activeTab === 'alert' && <AlertNotification />}
          
          
          {/* Profile content shows in the sidebar instead */}
        </div>
      </div>
    </div>
  );
}