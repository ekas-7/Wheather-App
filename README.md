# Weather Forecast Application

A full-stack weather forecasting application with an interactive React frontend and a machine learning-powered backend for predictive weather analytics.



## 📋 Overview

This application combines real-time weather data visualization with predictive forecasting using machine learning. The system consists of:

- **Frontend**: A responsive React-based dashboard for viewing current weather and forecasts
- **Backend**: A TensorFlow-based forecasting system that predicts weather patterns using historical data

## 🚀 Features

### Frontend
- Interactive weather dashboard with real-time data display
- 5-day weather forecast visualization
- City-based weather search functionality
- Responsive design for all device sizes
- Beautiful weather condition icons and intuitive UI

### Backend
- Machine learning-based weather prediction using LSTM/GRU neural networks
- Time series forecasting for up to 7 days ahead
- Model training pipeline with data preprocessing
- Model persistence for production deployment
- FastAPI integration (planned)

## 🛠️ Tech Stack

### Frontend
- React (Vite)
- Tailwind CSS for styling
- Lucide React for weather icons
- Modern JavaScript (ES6+)

### Backend
- Python 3.x
- TensorFlow/Keras for deep learning models
- NumPy/Pandas for data processing
- Scikit-learn for preprocessing
- FastAPI (future implementation)

## 🏗️ Project Structure

```
weather-forecast-app/
├── backend/
│   ├── requirements.txt      # Python dependencies
│   └── train.py              # ML model training script
│
├── frontend/
│   ├── public/               # Static assets
│   ├── src/
│   │   ├── App.jsx           # Main React component
│   │   ├── App.css           # App-specific styles
│   │   ├── assets/           # Images and SVG icons
│   │   ├── index.css         # Global styles
│   │   └── main.jsx          # React entry point
│   └── index.html            # HTML template
│
└── README.md                 # This file
```

## 🚀 Getting Started

### Prerequisites
- Node.js (v14+)
- Python 3.8+
- npm or yarn

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn
   ```

3. Start the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

4. Access the application at `http://localhost:5173`

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the weather prediction model:
   ```bash
   python train.py
   ```
   This will create the model files needed for prediction.
# Docker Setup for Weather Forecast Application

This document explains how to set up and run the Weather Forecast Application using Docker.

## 📄 Docker Files

The application includes three Docker configuration files:

1. `backend/Dockerfile` - Container configuration for the Python ML backend
2. `frontend/Dockerfile` - Container configuration for the React frontend
3. `docker-compose.yml` - Orchestration of both services

## 🚀 Running with Docker Compose

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Steps to Launch

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/weather-forecast-app.git
   cd weather-forecast-app
   ```

2. Create the Docker containers and start the services:
   ```bash
   docker-compose up -d
   ```

3. Access the application:
   - Frontend: http://localhost:80
   - Backend API (when implemented): http://localhost:8000

### Managing the Containers

- View running containers:
  ```bash
  docker-compose ps
  ```

- View logs:
  ```bash
  docker-compose logs
  # For a specific service:
  docker-compose logs frontend
  docker-compose logs backend
  ```

- Stop the services:
  ```bash
  docker-compose down
  ```

- Rebuild and restart containers (after changes):
  ```bash
  docker-compose up -d --build
  ```

## 🔧 Configuration

### Environment Variables

The Docker setup supports several environment variables:

#### Backend
- `MODEL_PATH`: Path to trained ML model
- `SCALER_PATH`: Path to data scaler
- `CONFIG_PATH`: Path to training configuration
- `SEQUENCE_PATH`: Path to last sequence data

#### Frontend
- `VITE_API_URL`: Backend API URL

### Volumes

The Docker setup uses volumes for data persistence:

- `model-data`: Stores trained model files between container restarts

### Docker Networks

- `weather-network`: Internal network for service communication

## 🧪 Production Considerations

For production deployment, consider:

1. Setting up proper SSL/TLS certificates
2. Adding authentication for the backend API
3. Implementing health checks
4. Setting up a CI/CD pipeline for automated deployment
5. Adding monitoring and logging solutions
6. Modifying the nginx configuration for better performance

## 🛠️ Troubleshooting

### Common Issues

1. **Port conflicts**: If ports 80 or 8000 are already in use on your machine, modify the port mappings in `docker-compose.yml`.

2. **Memory issues during model training**: Increase Docker's allocated memory in Docker Desktop settings.

3. **Permission issues with volumes**: Check file permissions on the host machine.

### Checking Logs

```bash
# Check the last 100 lines of logs
docker-compose logs --tail=100 backend
```

## 📝 Docker File Structure Updates

Make sure to update your project structure to include the Docker files:

```
weather-forecast-app/
├── backend/
│   ├── Dockerfile          # Backend container configuration
│   ├── requirements.txt
│   └── train.py
│
├── frontend/
│   ├── Dockerfile          # Frontend container configuration
│   ├── public/
│   ├── src/
│   └── ...
│
├── docker-compose.yml      # Service orchestration
└── README.md
```

## 📊 ML Model Architecture

The backend uses a deep learning approach for time series forecasting:

- **Model Types**: LSTM (Long Short-Term Memory), GRU (Gated Recurrent Unit), or Bidirectional LSTM
- **Architecture**: 
  - Input layer → LSTM/GRU layer (100 units) → Dropout (0.2) → 
  - LSTM/GRU layer (80 units) → Dropout (0.2) → Dense output layer
- **Training**: Uses early stopping, learning rate reduction, and model checkpointing
- **Sequence Length**: Default 30 days of historical data to predict 7 days ahead

## 🔄 Data Flow

1. Historical weather data is used to train the ML model
2. The trained model is saved along with preprocessing components
3. Frontend UI allows users to view current weather and forecasts
4. (Future) API endpoints will serve model predictions for requested locations

## 🧪 Example Usage

- View current weather conditions for major cities
- Check the 5-day forecast for any supported location
- (Future) Get ML-powered weather predictions based on historical patterns

## 📝 Future Enhancements

- [ ] Connect frontend to real weather API
- [ ] Implement backend FastAPI service for model predictions
- [ ] Add geolocation for automatic local weather
- [ ] Implement weather alerts and notifications
- [ ] Add historical weather data visualization
- [ ] Support for more weather parameters (UV index, air quality, etc.)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Dhruv
- Divyansh
- Ekas

## 🙏 Acknowledgments

- Weather icons provided by [Lucide React](https://lucide.dev/)
- React framework and Vite for frontend tooling
- TensorFlow and Keras for ML capabilities
