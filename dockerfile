# Use official Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose the Flask port
EXPOSE 5000

# Run the app
CMD ["python", "app/api.py"]

