# Vehicle Insurance Domain - MLOps Project

A comprehensive Machine Learning Operations (MLOps) project for vehicle insurance response prediction using a modular pipeline architecture with FastAPI deployment.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Data Pipeline](#data-pipeline)
- [Model Information](#model-information)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project implements an end-to-end machine learning solution for predicting vehicle insurance customer response to cross-sell offers. It follows MLOps best practices with:

- **Modular Architecture**: Separated concerns with data ingestion, validation, transformation, and training
- **MongoDB Integration**: Data storage and retrieval from MongoDB
- **AWS S3 Integration**: Model and preprocessing artifact storage
- **FastAPI Web Interface**: Interactive web application for predictions
- **Logging & Monitoring**: Comprehensive logging throughout the pipeline
- **Docker Support**: Containerized deployment

## Features

- **Data Ingestion**: Automated data fetching from MongoDB collections
- **Data Validation**: Schema validation and data quality checks
- **Data Transformation**: Feature engineering and preprocessing
- **Model Training**: Scikit-learn based model training with evaluation
- **Model Evaluation**: Comprehensive metrics and performance analysis
- **Model Serving**: S3-based model storage and retrieval
- **Web Interface**: User-friendly FastAPI web application for predictions
- **Prediction Pipeline**: Real-time inference on new data
- **Error Handling**: Custom exception handling with detailed traceback
- **Logging**: Centralized logging system for debugging

## Project Structure

```
Vehicle-Insurance-Domain-MLOPS/
├── src/
│   ├── __init__.py
│   ├── components/           # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   ├── model_evaluation.py
│   │   ├── model_pusher.py
│   │   ├── model_trainer.py
│   │   └── __init__.py
│   ├── configuration/        # External service connections
│   │   ├── aws_connection.py
│   │   ├── mongo_db_connection.py
│   │   └── __init__.py
│   ├── cloud_storage/        # Cloud storage handlers
│   │   ├── aws_storage.py
│   │   └── __init__.py
│   ├── constants/            # Project constants
│   │   └── __init__.py
│   ├── data_access/          # Data access layer
│   │   ├── proj1_data.py
│   │   └── __init__.py
│   ├── entity/               # Data models and entities
│   │   ├── artifact_entity.py
│   │   ├── config_entity.py
│   │   ├── estimator.py
│   │   ├── s3_estimator.py
│   │   └── __init__.py
│   ├── exception/            # Custom exceptions
│   │   └── __init__.py
│   ├── logger/               # Logging configuration
│   │   └── __init__.py
│   ├── pipeline/             # ML pipelines
│   │   ├── prediction_pipeline.py
│   │   ├── training_pipeline.py
│   │   └── __init__.py
│   └── utils/                # Utility functions
│       ├── main_utils.py
│       └── __init__.py
├── config/
│   ├── model.yaml            # Model configuration
│   └── schema.yaml           # Data schema configuration
├── templates/
│   └── vehicledata.html      # Web interface template
├── static/                   # Static assets (CSS, JS, images)
├── notebook/
│   ├── data.csv              # Sample data
│   ├── exp-notebook.ipynb    # Experimentation notebook
│   └── mongoDB_demo.ipynb    # MongoDB demo notebook
├── logs/                     # Application logs
├── artifact/                 # ML artifacts (models, preprocessors)
├── app.py                    # FastAPI application
├── demo.py                   # Demo script
├── Dockerfile                # Docker configuration
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── pyproject.toml            # Project metadata
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Requirements

- Python 3.10+
- MongoDB
- AWS Account (for S3 integration)
- pip or conda

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Vehicle-Insurance-Domain-MLOPS
```

### 2. Create Virtual Environment

```bash
# Using conda
conda create -n vehicle python=3.10
conda activate vehicle

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Package

```bash
pip install -e .
```

## Configuration

### 1. MongoDB Setup

Set the MongoDB connection URL as environment variable:

```powershell
# PowerShell
$env:MONGODB_URL="mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?appName=Cluster0"

# Linux/Mac
export MONGODB_URL="mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?appName=Cluster0"
```

Or for local MongoDB:

```bash
export MONGODB_URL="mongodb://localhost:27017"
```

### 2. AWS Credentials

Set AWS credentials for S3 access:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

### 3. Configuration Files

Update configuration files in the `config/` directory:

- **model.yaml**: Model hyperparameters and configuration
- **schema.yaml**: Data schema and validation rules

## Usage

### Running Training Pipeline

```bash
python demo.py
```

This will:
1. Fetch data from MongoDB
2. Validate data schema
3. Transform and preprocess data
4. Train the model
5. Evaluate performance
6. Push model to S3

### Running Web Application

```bash
python app.py
```

Access the web interface at: **http://localhost:5000**

### Training Pipeline (Python)

```python
from src.pipeline.training_pipeline import TrainPipeline

pipeline = TrainPipeline()
pipeline.run_pipeline()
```

### Prediction Pipeline

```python
from src.pipeline.prediction_pipeline import VehicleData, VehicleDataClassifier

# Create input data
vehicle_data = VehicleData(
    Gender=1,
    Age=35,
    Driving_License=1,
    Region_Code=28.0,
    Previously_Insured=0,
    Annual_Premium=45000,
    Policy_Sales_Channel=26.0,
    Vintage=8,
    Vehicle_Age_lt_1_Year=0,
    Vehicle_Age_gt_2_Years=1,
    Vehicle_Damage_Yes=1
)

# Make prediction
predictor = VehicleDataClassifier()
vehicle_df = vehicle_data.get_vehicle_input_data_frame()
prediction = predictor.predict(dataframe=vehicle_df)

print(f"Prediction: {prediction[0]}")  # 0 or 1
```

## API Endpoints

### 1. Home Page

```
GET /
```
Returns the main prediction form interface.

### 2. Train Model

```
GET /train
```
Triggers the full training pipeline.

**Response:**
```json
{
  "status": "Training successful!!!"
}
```

### 3. Make Prediction

```
POST /
Content-Type: application/x-www-form-urlencoded
```

**Parameters:**
- Gender (int): 0/1
- Age (int): Customer age
- Driving_License (int): 0/1
- Region_Code (float): Region code
- Previously_Insured (int): 0/1
- Annual_Premium (float): Annual premium amount
- Policy_Sales_Channel (float): Sales channel code
- Vintage (int): Customer policy duration in years
- Vehicle_Age_lt_1_Year (int): 0/1
- Vehicle_Age_gt_2_Years (int): 0/1
- Vehicle_Damage_Yes (int): 0/1

**Response:**
```html
<!-- HTML page with prediction result: Response-Yes or Response-No -->
```

## Data Pipeline

### 1. Data Ingestion
- Connects to MongoDB
- Fetches data from specified collection
- Exports to CSV format

### 2. Data Validation
- Validates schema against configuration
- Checks data types and constraints
- Generates validation report

### 3. Data Transformation
- Handles missing values
- Performs feature engineering
- Applies scaling/normalization
- Saves preprocessing object

### 4. Model Training
- Splits data into train/test sets
- Trains scikit-learn model
- Performs hyperparameter tuning

### 5. Model Evaluation
- Calculates performance metrics (accuracy, precision, recall, F1)
- Generates evaluation report
- Compares with baseline

### 6. Model Pusher
- Uploads model to AWS S3
- Stores preprocessing artifacts
- Maintains model versioning

## Model Information

**Model Type**: Binary Classification (Scikit-learn based)

**Target Variable**: Response (0/1)
- 0: Customer will not respond to cross-sell offer
- 1: Customer will respond to cross-sell offer

**Features**: 11 vehicle and customer-related features

**Data Split**: 75% training, 25% testing

**Artifacts Stored:**
- `model.pkl`: Trained model
- `preprocessing.pkl`: Preprocessing transformations

## Deployment

### Docker Build

```bash
docker build -t vehicle-insurance:latest .
```

### Docker Run

```bash
docker run -p 5000:5000 \
  -e MONGODB_URL="mongodb+srv://..." \
  -e AWS_ACCESS_KEY_ID="..." \
  -e AWS_SECRET_ACCESS_KEY="..." \
  vehicle-insurance:latest
```

### Environment Variables Required

- `MONGODB_URL`: MongoDB connection string
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key

## Logging

Logs are stored in the `logs/` directory with timestamps:

```
logs/
├── 2026_07_14_12_30_45.log
├── 2026_07_14_13_15_22.log
└── ...
```

Access logs for debugging:

```bash
tail -f logs/*.log
```

## Exception Handling

Custom exceptions with detailed traceback information:

```python
from src.exception import MyException
import sys

try:
    # Your code
    pass
except Exception as e:
    raise MyException(e, sys)
```

## Contributing

1. Create a feature branch
2. Commit your changes
3. Push to the branch
4. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or suggestions, please open an issue in the repository.

## Reference

- **Framework**: FastAPI
- **ML Library**: Scikit-learn
- **Database**: MongoDB
- **Cloud Storage**: AWS S3
- **Containerization**: Docker

---

**Last Updated**: 2026-07-14

**Version**: 1.0.0
