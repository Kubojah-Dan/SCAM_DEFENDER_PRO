import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import InputField from './InputField';
import Button from './Button';
import './theme.css';
import logo from './shieldx-logo.png';

export default function Login() {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [errors, setErrors] = useState({
    email: '',
    password: ''
  });
  const [isLogin, setIsLogin] = useState(true); // Toggle between login/signup
  const navigate = useNavigate();

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
    
    if (name === 'email' && value && !validateEmail(value)) {
      setErrors(prev => ({...prev, email: 'Invalid email format'}));
    }
    
    if (name === 'password' && value && !validatePassword(value)) {
      setErrors(prev => ({...prev, password: 'Password must be at least 8 characters'}));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    const validationErrors = {
      email: !formData.email ? 'Email is required' : !validateEmail(formData.email) ? 'Invalid email format' : '',
      password: !formData.password ? 'Password is required' : !validatePassword(formData.password) ? 'Password must be at least 8 characters' : ''
    };
    
    setErrors(validationErrors);
    
    if (!validationErrors.email && !validationErrors.password) {
      navigate('/dashboard');
    }
  };

  return (
    <div className="login-container" style={styles.container}>
      <div style={styles.header}>
        <img src={logo} alt="ShieldX Logo" style={styles.logo} />
        <h1 style={styles.title}>ShieldX</h1>
        <p style={styles.subtitle}>Secure Your Digital Life</p>
      </div>
      
      <form onSubmit={handleSubmit} style={styles.form}>
        <h2 style={{color: '#333', textAlign: 'center', marginBottom: '20px'}}>
          {isLogin ? 'Login' : 'Sign Up'}
        </h2>

        {!isLogin && (
          <div style={styles.inputGroup}>
            <label htmlFor="username" style={styles.label}>Username</label>
            <input
              id="username"
              name="username"
              type="text"
              value={formData.username || ''}
              onChange={handleChange}
              style={styles.input}
            />
          </div>
        )}

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

        {!isLogin && (
          <div style={styles.inputGroup}>
            <label htmlFor="confirmPassword" style={styles.label}>Confirm Password</label>
            <input
              id="confirmPassword"
              name="confirmPassword"
              type="password"
              value={formData.confirmPassword || ''}
              onChange={handleChange}
              style={styles.input}
            />
          </div>
        )}

        <Button 
          type="submit" 
          style={{
            ...styles.button,
            opacity: (!formData.email || !formData.password || errors.email || errors.password) ? 0.7 : 1,
            cursor: (!formData.email || !formData.password || errors.email || errors.password) ? 'not-allowed' : 'pointer'
          }}
        >
          {isLogin ? 'Login' : 'Sign Up'}
        </Button>

        <p style={{textAlign: 'center', marginTop: '15px', color: '#666'}}>
          {isLogin ? "Don't have an account?" : "Already have an account?"}
          <button 
            type="button"
            onClick={() => setIsLogin(!isLogin)}
            style={{
              background: 'none',
              border: 'none',
              color: '#1a73e8',
              cursor: 'pointer',
              marginLeft: '5px'
            }}
          >
            {isLogin ? 'Sign up' : 'Login'}
          </button>
        </p>
      </form>
    </div>
  );
}

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