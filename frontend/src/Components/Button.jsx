import React from 'react';
import './theme.css';

export default function Button({ children, ...props }) {
  return (
    <button className="scamdefender-btn" {...props}>
      {children}
    </button>
  );
}