# 🏠 Boston House Price Prediction - MLOps Project

A complete end-to-end MLOps pipeline for predicting Boston house prices using Artificial Neural Networks (ANN). This project demonstrates MLOps best practices with data versioning, model tracking, containerization, CI/CD, and cloud deployment.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Live Demo](#live-demo)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
- [Usage](#usage)
  - [Running the ML Pipeline](#running-the-ml-pipeline)
  - [Running Locally](#running-locally)
  - [Running with Docker](#running-with-docker)
- [API Documentation](#api-documentation)
- [Model Information](#model-information)
- [CI/CD Pipeline](#cicd-pipeline)
- [Deployment](#deployment)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## 🎯 Overview

This project implements a comprehensive end-to-end MLOps pipeline for predicting Boston house prices. It demonstrates best practices in MLOps workflows and includes:

- **Data Pipeline**: Automated data ingestion, preprocessing, and feature engineering using DVC
- **Model Training**: Neural network model training with TensorFlow/Keras
- **Model Registry**: Model versioning and tracking with MLflow (hosted on DagsHub)
- **API Service**: FastAPI-based REST API for model inference
- **Web Interface**: Interactive Streamlit application for predictions
- **CI/CD**: Automated testing, building, and deployment with GitHub Actions
- **Containerization**: Docker images for both API and Streamlit services
- **Cloud Deployment**: Live deployment on Render (free tier)

## ✨ Features

- 🔄 **Automated ML Pipeline**: DVC-managed pipeline with data versioning
- 📊 **Model Tracking**: MLflow integration for experiment tracking and model registry
- 🤖 **Model Promotion**: Automated model promotion from staging to production
- 🐳 **Containerization**: Docker images for easy deployment
- 🚀 **CI/CD**: GitHub Actions for automated testing and deployment
- 🌐 **REST API**: FastAPI with automatic documentation
- 💻 **Web UI**: User-friendly Streamlit interface
- ☁️ **Cloud Deployment**: Live deployment on Render
- ✅ **Testing**: Comprehensive test suite for models and API
- 📈 **Monitoring**: Health checks and logging

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit UI  │  (Port 8501)
│   (Frontend)    │
└────────┬────────┘
         │ HTTP Requests
         ▼
┌─────────────────┐
│   FastAPI       │  (Port 8000)
│   (Backend)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   MLflow        │
│   (Model Store) │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   DagsHub       │
│   (Remote)      │
└─────────────────┘
```

### Pipeline Flow

1. **Data Ingestion**: Load raw Boston housing dataset
2. **Data Preprocessing**: Handle missing values, outliers (winsorization), train/test split
3. **Feature Engineering**: Feature selection, scaling, transformation
4. **Model Building**: Train neural network with early stopping
5. **Model Evaluation**: Calculate metrics (R², MAE, RMSE)
6. **Model Registration**: Register model in MLflow
7. **Model Promotion**: Automatically promote to production if metrics improve

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.13**: Programming language
- **TensorFlow 2.20.0**: Deep learning framework
- **Keras 3.13.0**: High-level neural network API
- **FastAPI 0.127.0**: Modern web framework for APIs
- **Streamlit 1.52.2**: Web application framework
- **DVC 3.65.0**: Data version control
- **MLflow 3.8.0**: Model lifecycle management

### Data & ML Libraries
- **Pandas 2.3.3**: Data manipulation
- **NumPy 2.4.0**: Numerical computing
- **Scikit-learn 1.8.0**: Machine learning utilities
- **Joblib 1.5.3**: Model serialization

### DevOps & Deployment
- **Docker**: Containerization
- **GitHub Actions**: CI/CD
- **Render**: Cloud hosting
- **DagsHub**: MLflow tracking server

## 📁 Project Structure

```
Boston House MLOps/
├── .github/
│   └── workflows/
│       └── ci.yaml              # CI/CD pipeline
├── data/
│   ├── raw/                     # Raw data
│   ├── interim/                 # Intermediate processed data
│   └── processed/               # Final processed data
├── models/                      # Trained models and artifacts
├── notebooks/                   # Jupyter notebooks
├── reports/                     # Metrics and experiment info
├── scripts/
│   └── model_promotion.py       # Model promotion logic
├── src/
│   ├── api/                     # FastAPI application
│   │   ├── app.py              # Main API application
│   │   ├── config.py           # API configuration
│   │   ├── predict.py          # Prediction logic
│   │   └── schemas.py          # Pydantic schemas
│   ├── data/                    # Data pipeline
│   │   ├── data_ingestion.py
│   │   └── data_preprocessing.py
│   ├── features/                # Feature engineering
│   │   └── feature_engineering.py
│   ├── model/                   # Model pipeline
│   │   ├── model_building.py
│   │   ├── model_evaluation.py
│   │   └── model_registration.py
│   ├── streamlit_app/           # Streamlit application
│   │   ├── app.py
│   │   └── utils.py
│   └── config.py                # Shared configuration
├── tests/                       # Test suite
├── Dockerfile.api               # FastAPI Docker image
├── Dockerfile.streamlit         # Streamlit Docker image
├── docker-compose.yml           # Docker Compose configuration
├── dvc.yaml                     # DVC pipeline definition
├── params.yaml                  # Pipeline parameters
├── requirements.txt             # All dependencies
├── requirements-api.txt         # API dependencies
├── requirements-streamlit.txt   # Streamlit dependencies
└── README.md                    # This file
```

## 🌐 Live Demo

The application is deployed on Render (free tier):

### 🚀 FastAPI Service
- **URL**: https://boston-api-latest.onrender.com
- **API Docs**: https://boston-api-latest.onrender.com/docs
- **Health Check**: https://boston-api-latest.onrender.com/health

### 🎨 Streamlit Application
- **URL**: https://boston-streamlit-latest.onrender.com

> **⚠️ Important**: The FastAPI service must be started before using the Streamlit app, as Streamlit depends on the API for predictions.

> **Note**: Free tier instances on Render may spin down after inactivity. The first request may take a few seconds to wake up the service.

## 🚀 Getting Started

### Prerequisites

- Python 3.13 or higher
- Git
- Docker (optional, for containerized deployment)
- DagsHub account (for MLflow tracking)
- GitHub account (for CI/CD)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tarun304/Boston-House-Price-Prediction-using-Artificial-Neural-Networks.git
   cd "Boston House MLOps"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install uv
   uv pip install -r requirements.txt
   ```

### Environment Setup

1. **Create a `.env` file** in the root directory:
   ```env
   # DagsHub Configuration
   DAGSHUB_USERNAME=your_dagshub_username
   DAGSHUB_TOKEN=your_dagshub_token
   
   # MLflow Configuration
   MLFLOW_TRACKING_URI=https://dagshub.com/your_username/your_repo.mlflow
   MLFLOW_TRACKING_USERNAME=your_dagshub_username
   MLFLOW_TRACKING_PASSWORD=your_dagshub_token
   ```

2. **Configure DVC remote** (if using DVC for data versioning):
   ```bash
   dvc remote add origin <your-dvc-remote-url>
   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user $DAGSHUB_USERNAME
   dvc remote modify origin --local password $DAGSHUB_TOKEN
   ```

## 💻 Usage

### Running the ML Pipeline

1. **Pull data from DVC** (if data is versioned):
   ```bash
   dvc pull -r origin
   ```

2. **Run the complete pipeline**:
   ```bash
   dvc repro
   ```

   This will execute:
   - Data ingestion
   - Data preprocessing
   - Feature engineering
   - Model building
   - Model evaluation
   - Model registration

3. **Promote model to production** (optional):
   ```bash
   python scripts/model_promotion.py
   ```

### Running Locally

#### Option 1: Run FastAPI and Streamlit separately

1. **Start FastAPI** (Terminal 1):
   ```bash
   uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start Streamlit** (Terminal 2):
   ```bash
   streamlit run src/streamlit_app/app.py
   ```

3. **Access the applications**:
   - FastAPI: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Streamlit: http://localhost:8501

#### Option 2: Run with Docker Compose

1. **Build and start services**:
   ```bash
   docker-compose up --build
   ```

2. **Access the applications**:
   - FastAPI: http://localhost:8000
   - Streamlit: http://localhost:8501

### Running with Docker

#### Build Docker Images

1. **Build FastAPI image**:
   ```bash
   docker build -f Dockerfile.api -t boston-api:latest .
   ```

2. **Build Streamlit image**:
   ```bash
   docker build -f Dockerfile.streamlit -t boston-streamlit:latest .
   ```

#### Run Containers

1. **Run FastAPI container**:
   ```bash
   docker run -p 8000:8000 \
     -e DAGSHUB_USERNAME=$DAGSHUB_USERNAME \
     -e DAGSHUB_TOKEN=$DAGSHUB_TOKEN \
     -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
     -e MLFLOW_TRACKING_PASSWORD=$DAGSHUB_TOKEN \
     boston-api:latest
   ```

2. **Run Streamlit container**:
   ```bash
   docker run -p 8501:8501 \
     -e API_URL=http://localhost:8000 \
     boston-streamlit:latest
   ```

## 📚 API Documentation

### Endpoints

#### `GET /`
Returns API information and available endpoints.

**Response**:
```json
{
  "message": "Boston House Price Prediction API",
  "version": "1.0.0",
  "description": "MLOps-powered API for predicting Boston house prices using ANN",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "docs": "/docs"
  }
}
```

#### `GET /health`
Health check endpoint to verify API and model status.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### `POST /predict`
Predict house price based on input features.

**Request Body**:
```json
{
  "CRIM": 0.00632,
  "ZN": 18.0,
  "INDUS": 2.31,
  "CHAS": 0,
  "NOX": 0.538,
  "RM": 6.575,
  "AGE": 65.2,
  "DIS": 4.0900,
  "RAD": 1,
  "TAX": 296.0,
  "PTRATIO": 15.3,
  "B": 396.90,
  "LSTAT": 4.98
}
```

**Response**:
```json
{
  "predicted_price": 24000.00
}
```

### Interactive API Documentation

Visit `/docs` for Swagger UI or `/redoc` for ReDoc documentation.

## 🤖 Model Information

### Model Architecture
- **Type**: Artificial Neural Network (ANN)
- **Framework**: TensorFlow/Keras
- **Architecture**: 
  - Input Layer: 5 features (after feature selection)
  - Hidden Layer(s): 64 units (configurable)
  - Output Layer: 1 unit (regression)
- **Activation**: ReLU (hidden), Linear (output)
- **Optimizer**: Adam
- **Learning Rate**: 0.01
- **Regularization**: Early stopping with patience

### Features
The model uses 13 input features:
- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centres
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property-tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)² where Bk is the proportion of blacks by town
- **LSTAT**: % lower status of the population

After feature engineering, 5 most important features are selected.

### Model Metrics
- **R² Score**: Typically > 0.70
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error

## 🔄 CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment.

### Pipeline Stages

1. **ML Pipeline**:
   - Checkout code
   - Setup Python environment
   - Install dependencies
   - Configure DVC
   - Pull data from DVC
   - Run DVC pipeline (`dvc repro`)
   - Run tests (model, API, data pipeline)
   - Push artifacts to DVC
   - Promote model to production (if metrics improve)

2. **Docker Build & Push**:
   - Build Docker images (only if relevant files changed or model promoted)
   - Push to GitHub Container Registry
   - Cache layers for faster builds

### Workflow Triggers
- Push to `main` or `dev` branches
- Pull requests to `main` branch

### Secrets Required
- `DAGSHUB_TOKEN`: DagsHub authentication token
- `DAGSHUB_USERNAME`: DagsHub username
- `MLFLOW_TRACKING_URI`: MLflow tracking URI
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

## 🚢 Deployment

### Render Deployment

The application is deployed on Render using free tier instances.

#### FastAPI Service
- **Service Name**: `boston-api:latest`
- **Image**: `tarun304/boston-api:latest`
- **URL**: https://boston-api-latest.onrender.com
- **Port**: 8000

#### Streamlit Service
- **Service Name**: `boston-streamlit:latest`
- **Image**: `tarun304/boston-streamlit:latest`
- **URL**: https://boston-streamlit-latest.onrender.com
- **Port**: 8501

#### Environment Variables (Render)
Configure these in Render dashboard:
- `DAGSHUB_USERNAME`
- `DAGSHUB_TOKEN`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_PASSWORD`
- `API_URL` (for Streamlit service, set to FastAPI URL)

### Manual Deployment Steps

1. **Build Docker images** (locally or via CI/CD)
2. **Push to container registry** (Docker Hub, GitHub Container Registry, etc.)
3. **Create services on Render**:
   - Create new Web Service
   - Select Docker image
   - Configure environment variables
   - Set health check path (`/health` for FastAPI)
4. **Deploy**: Render will automatically pull and deploy the image

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run specific test files
pytest tests/test_model.py -v
pytest tests/test_api.py -v
pytest tests/test_datapipeline.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Structure
- `tests/test_model.py`: Model training and evaluation tests
- `tests/test_api.py`: API endpoint tests
- `tests/test_datapipeline.py`: Data pipeline tests

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Tarun Kumar Behera**

- GitHub: [@Tarun304](https://github.com/Tarun304)
- Project Link: https://github.com/Tarun304/Boston-House-Price-Prediction-using-Artificial-Neural-Networks
---

## 🙏 Acknowledgments

- Boston Housing Dataset
- DagsHub for MLflow hosting
- Render for free cloud hosting
- Open source community for amazing tools

---

**⭐ If you find this project helpful, please consider giving it a star!**

