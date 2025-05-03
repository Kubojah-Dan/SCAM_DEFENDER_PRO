# app/__init__.py
from flask import Flask
from app.utils.config import load_config
import os
import pickle

def create_app():
    app = Flask(__name__)
    
    # Loading config
    config = load_config()
    app.config.update(config)

    # Loading models
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    try:
        with open(os.path.join(model_dir, 'email_model.pkl'), 'rb') as f:
            app.config['email_model'] = pickle.load(f)
        with open(os.path.join(model_dir, 'message_model.pkl'), 'rb') as f:
            app.config['message_model'] = pickle.load(f)
        with open(os.path.join(model_dir, 'url_model.pkl'), 'rb') as f:
            app.config['url_model'] = pickle.load(f)
    except Exception as e:
        print("Error loading models:", e)
    return app

