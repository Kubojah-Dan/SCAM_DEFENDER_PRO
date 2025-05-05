import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import SocialLogin from './SocialLogin';
import InputField from './InputField';
import Button from './Button';
import './theme.css';
import logo from './shieldx-logo.png';

export default function Signup() {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [errors, setErrors] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const navigate = useNavigate();

  const validateUsername = (username) => {
    return username.length >= 3;
  };

  const validateEmail = (email) => {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
  };

  const validatePassword = (password) => {
    return password.length >= 8;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({...prev, [name]: value}));
    
    if (errors[name]) {
      setErrors(prev => ({...prev, [name]: ''}));
    }
  };

  const handleBlur = (e) => {
    const { name, value } = e.target;
    
    if (name === 'username' && value && !validateUsername(value)) {
      setErrors(prev => ({...prev, username: 'Username must be at least 3 characters'}));
    }
    
    if (name === 'email' && value && !validateEmail(value)) {
      setErrors(prev => ({...prev, email: 'Invalid email format'}));
    }
    
    if (name === 'password' && value && !validatePassword(value)) {
      setErrors(prev => ({...prev, password: 'Password must be at least 8 characters'}));
    }
    
    if (name === 'confirmPassword' && value && value !== formData.password) {
      setErrors(prev => ({...prev, confirmPassword: 'Passwords do not match'}));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    const validationErrors = {
      username: !formData.username ? 'Username is required' : !validateUsername(formData.username) ? 'Username must be at least 3 characters' : '',
      email: !formData.email ? 'Email is required' : !validateEmail(formData.email) ? 'Invalid email format' : '',
      password: !formData.password ? 'Password is required' : !validatePassword(formData.password) ? 'Password must be at least 8 characters' : '',
      confirmPassword: !formData.confirmPassword ? 'Please confirm password' : formData.confirmPassword !== formData.password ? 'Passwords do not match' : ''
    };
    
    setErrors(validationErrors);
    
    if (!validationErrors.username && !validationErrors.email && 
        !validationErrors.password && !validationErrors.confirmPassword) {
      navigate('/dashboard');
    }
  };

  return (
    <div className="signup-container" style={styles.container}>
      <div style={styles.header}>
        <img src={logo} alt="ShieldX Logo" style={styles.logo} />
        <h1 style={styles.title}>ShieldX</h1>
        <p style={styles.subtitle}>Secure Your Digital Life</p>
      </div>
      
      <form onSubmit={handleSubmit} style={styles.form}>
        <h2 style={{color: '#333', textAlign: 'center', marginBottom: '20px'}}>
          Create Account
        </h2>

        <div style={styles.inputGroup}>
          <label htmlFor="username" style={styles.label}>Username</label>
          <input
            id="username"
            name="username"
            type="text"
            value={formData.username}
            onChange={handleChange}
            onBlur={handleBlur}
            style={{
              ...styles.input,
              borderColor: errors.username ? '#ff4444' : '#ddd'
            }}
          />
          {errors.username && <span style={styles.error}>{errors.username}</span>}
        </div>

        <div style={styles.inputGroup}>
          <label htmlFor="email" style={styles.label}>Email</label>
          <input
            id="email"
            name="email"
            type="email"
            value={formData.email}
            onChange={handleChange}
            onBlur={handleBlur}
            style={{
              ...styles.input,
              borderColor: errors.email ? '#ff4444' : '#ddd'
            }}
          />
          {errors.email && <span style={styles.error}>{errors.email}</span>}
        </div>

        <div style={styles.inputGroup}>
          <label htmlFor="password" style={styles.label}>Password</label>
          <input
            id="password"
            name="password"
            type="password"
            value={formData.password}
            onChange={handleChange}
            onBlur={handleBlur}
            style={{
              ...styles.input,
              borderColor: errors.password ? '#ff4444' : '#ddd'
            }}
          />
          {errors.password && <span style={styles.error}>{errors.password}</span>}
        </div>

        <div style={styles.inputGroup}>
          <label htmlFor="confirmPassword" style={styles.label}>Confirm Password</label>
          <input
            id="confirmPassword"
            name="confirmPassword"
            type="password"
            value={formData.confirmPassword}
            onChange={handleChange}
            onBlur={handleBlur}
            style={{
              ...styles.input,
              borderColor: errors.confirmPassword ? '#ff4444' : '#ddd'
            }}
          />
          {errors.confirmPassword && <span style={styles.error}>{errors.confirmPassword}</span>}
        </div>

        <Button 
          type="submit" 
          style={{
            ...styles.button,
            opacity: (!formData.username || !formData.email || !formData.password || !formData.confirmPassword || 
                     errors.username || errors.email || errors.password || errors.confirmPassword) ? 0.7 : 1,
            cursor: (!formData.username || !formData.email || !formData.password || !formData.confirmPassword || 
                    errors.username || errors.email || errors.password || errors.confirmPassword) ? 'not-allowed' : 'pointer'
          }}
        >
          Sign Up
        </Button>

        <div style={{textAlign: 'center', margin: '15px 0', color: '#666'}}>
          or sign up with
        </div>

        <SocialLogin />

        <p style={{textAlign: 'center', marginTop: '15px', color: '#666'}}>
          Already have an account?
          <button 
            type="button"
            onClick={() => navigate('/login')}
            style={{
              background: 'none',
              border: 'none',
              color: '#1a73e8',
              cursor: 'pointer',
              marginLeft: '5px'
            }}
          >
            Login
          </button>
        </p>
      </form>
    </div>
  );
}

// Reuse the same styles object from Login.jsx
const styles = {
  container: {
    backgroundColor: '#1a73e8',
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '20px',
    color: 'white',
  },
  header: {
    textAlign: 'center',
    marginBottom: '30px',
  },
  logo: {
    width: '80px',
    height: '80px',
    marginBottom: '10px',
  },
  title: {
    fontSize: '2.5rem',
    fontWeight: 'bold',
    margin: '0',
  },
  subtitle: {
    fontSize: '1rem',
    opacity: '0.8',
    margin: '5px 0 0 0',
  },
  form: {
    backgroundColor: 'white',
    padding: '30px',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
    width: '100%',
    maxWidth: '400px',
  },
  inputGroup: {
    marginBottom: '20px',
  },
  label: {
    display: 'block',
    marginBottom: '8px',
    color: '#333',
    fontWeight: '500',
  },
  input: {
    width: '100%',
    padding: '12px',
    borderRadius: '4px',
    border: '1px solid #ddd',
    fontSize: '1rem',
    boxSizing: 'border-box',
  },
  error: {
    color: '#ff4444',
    fontSize: '0.8rem',
    marginTop: '4px',
    display: 'block',
  },
  button: {
    backgroundColor: '#1a73e8',
    color: 'white',
    width: '100%',
    padding: '12px',
    borderRadius: '4px',
    border: 'none',
    fontSize: '1rem',
    marginTop: '10px',
  }  
};