FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --default-timeout=100 --retries=10 --no-cache-dir -r requirements.txt
RUN pip install -i https://pypi.org/simple --timeout=100 --no-cache-dir -r requirements.txt
RUN pip install gunicorn

COPY . .
CMD ["gunicorn","-w","4","-b","0.0.0.0:5000","app.api:app"]


