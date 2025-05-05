import React from 'react';
import { FiMenu, FiX } from 'react-icons/fi';
import './theme.css';

const Button = ({ children, isOpen, onToggle, ...props }) => {
  // If it's a menu toggle button
  if (props.className === 'menu-toggle') {
    return (
      <button 
        className="scamdefender-btn menu-toggle"
        onClick={onToggle}
        style={{ zIndex: 1001 }}
        {...props}
      >
        {isOpen ? <FiX size={24} /> : <FiMenu size={24} />}
      </button>
    );
  }

  // Regular button
  return (
    <button className="scamdefender-btn" {...props}>
      {children}
    </button>
  );
};

export default Button;
