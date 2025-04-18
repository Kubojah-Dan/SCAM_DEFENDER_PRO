
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class PredictionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction_type = db.Column(db.String(50))  
    input_text = db.Column(db.Text)            
    prediction_result = db.Column(db.String(50))
    details = db.Column(db.Text)                
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f"<PredictionLog {self.prediction_type} - {self.prediction_result}>"
