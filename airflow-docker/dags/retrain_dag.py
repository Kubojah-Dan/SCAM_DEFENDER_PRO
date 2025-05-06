from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import subprocess

default_args = {"owner":"airflow","retries":1}
with DAG('retrain_pipeline', default_args=default_args,
         schedule_interval='@daily', start_date=days_ago(1)) as dag:

    def retrain_all():
        # invoke your train.py
        subprocess.run(["python","app/train.py"], check=True)

    t1 = PythonOperator(task_id="retrain_models", python_callable=retrain_all)

    t1
