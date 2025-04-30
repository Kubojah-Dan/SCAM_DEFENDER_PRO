import React from 'react';
import { FaGoogle, FaFacebook } from 'react-icons/fa';
import Button from './Button';
import './theme.css';

export default function SocialLogin() {
  return (
    <div className="social-login">
      <div className="divider">OR</div>
      <Button className="google-btn">
        <FaGoogle /> Continue with Google
      </Button>
      <Button className="facebook-btn">
        <FaFacebook /> Continue with Facebook
      </Button>
    </div>
  );
}