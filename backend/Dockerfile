FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY backend/ .

# Create a directory for model artifacts
RUN mkdir -p /app/models

# This will create sample data if not exists and train the model
# For production, you might want to skip this step and mount pre-trained models
RUN python train.py

# Environmental variables
ENV MODEL_PATH=/app/models/weather_model_lstm.h5
ENV SCALER_PATH=/app/models/weather_scaler_temp.pkl
ENV CONFIG_PATH=/app/models/training_config.pkl
ENV SEQUENCE_PATH=/app/models/last_sequence.npy

# Install and use uvicorn for the API (uncomment when API is implemented)
# EXPOSE 8000
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# For now, use a simple placeholder
CMD ["echo", "Backend container is running. Replace this with API server when implemented."]