import os
import json
import datetime
from celery import Celery
from kafka import KafkaConsumer, KafkaProducer
import psycopg2
from sqlalchemy import create_engine
from app.utils.clean_text import clean_text

# 1️⃣ Configure Celery
celery = Celery(
    "scam_defender_tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
)

# 2️⃣ Database setup (SQLAlchemy)
DB_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/scamdb")
engine = create_engine(DB_URL)

# 3️⃣ Kafka setup
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode()
)

@celery.task
def consume_predictions():
    """
    Consume 'predictions' topic, and insert records into PostgreSQL.
    """
    consumer = KafkaConsumer(
        'predictions',
        bootstrap_servers=KAFKA_BROKER,
        group_id='db-writer',
        auto_offset_reset='earliest',
        value_deserializer=lambda m: json.loads(m.decode())
    )
    conn = engine.connect()
    for record in consumer:
        data = record.value
        # e.g., data = {"type":"email","text":"...","prediction":"1","confidence":0.98,"timestamp":"..."}
        ins = f"""
            INSERT INTO predictions(type, text, prediction, confidence, created_at)
            VALUES(%s,%s,%s,%s,%s)
        """
        conn.execute(ins, (
            data["type"],
            clean_text(data["text"]),
            data["prediction"],
            float(data["confidence"]),
            datetime.datetime.utcnow()
        ))
    conn.close()

@celery.task
def monitor_and_retrain():
    """
    Check model performance and trigger retraining if drift is detected.
    """
    # Placeholder: implement actual drift detection logic via stored metrics
    # If drift detected:
    from scripts.train import main as retrain_all
    retrain_all()  # re-trains all models
    producer.send('model-events', {"event":"retrained","timestamp":str(datetime.datetime.utcnow())})

# 4️⃣ Schedule via Celery Beat (in celeryconfig.py or in your Celery call)
# Example celeryconfig.py:
# beat_schedule = {
#     'consume-preds-every-minute': {
#         'task': 'app.tasks.consume_predictions',
#         'schedule': 60.0,
#     },
#     'monitor-and-retrain-hourly': {
#         'task': 'app.tasks.monitor_and_retrain',
#         'schedule': 3600.0,
#     },
# }
# enable in your Celery start command: celery -A app.tasks worker -B


