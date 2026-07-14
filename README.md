# Vehicle Insurance MLOps Pipeline

An end-to-end MLOps project that demonstrates how to build, train, evaluate, and deploy a machine learning model for vehicle insurance data — complete with cloud data storage, model versioning, containerization, and a fully automated CI/CD pipeline.

This project is designed as a reference implementation for anyone looking to understand how production-grade ML systems are built, showcasing the tools, patterns, and best practices used in real-world MLOps workflows.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Architecture / Workflow](#architecture--workflow)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [1. Project Template](#step-1-project-template)
  - [2. Package Management](#step-2-package-management)
  - [3. Virtual Environment & Dependencies](#step-3-virtual-environment-and-dependencies)
- [Data Management (MongoDB)](#data-management-mongodb)
  - [4. MongoDB Atlas Configuration](#step-4-mongodb-atlas-configuration)
  - [5. Pushing Data to MongoDB](#step-5-pushing-data-to-mongodb)
- [Logging, Exceptions & EDA](#logging-exceptions--eda)
  - [6. Logging and Exception Handling](#step-6-set-up-logging-and-exception-handling)
  - [7. Exploratory Data Analysis](#step-7-exploratory-data-analysis-eda-and-feature-engineering)
- [Data Ingestion](#data-ingestion)
  - [8. Data Ingestion Pipeline](#step-8-data-ingestion-pipeline)
- [Data Validation, Transformation & Training](#data-validation-transformation--model-training)
  - [9. Data Validation](#step-9-data-validation)
  - [10. Data Transformation](#step-10-data-transformation)
  - [11. Model Training](#step-11-model-training)
- [AWS Setup for Evaluation & Deployment](#aws-setup-for-model-evaluation--deployment)
  - [12. AWS Setup](#step-12-aws-setup)
  - [13. Model Evaluation & S3 Push](#step-13-model-evaluation-and-pushing-to-s3)
- [Prediction Pipeline & Web App](#model-evaluation-model-pusher-and-prediction-pipeline)
  - [14. Model Evaluation & Pusher](#step-14-model-evaluation--model-pusher)
  - [15. Static & Template Directories](#step-15-static-and-template-directory)
- [CI/CD Setup](#cicd-setup-with-docker-github-actions-and-aws)
  - [16. Docker & GitHub Actions](#step-16-docker-and-github-actions)
  - [17. AWS EC2 & ECR](#step-17-aws-ec2-and-ecr)
  - [18. Final Deployment](#step-18-final-steps)
- [Project Workflow Summary](#project-workflow-summary)
- [Additional Resources](#additional-resources)
- [Contact](#connect)

---

## Overview

This project simulates a real-world MLOps pipeline for a **vehicle insurance prediction system**. It covers the complete machine learning lifecycle — from raw data ingestion out of a cloud database, through validation, transformation, and model training, all the way to cloud-based model storage and a fully containerized deployment served through an automated CI/CD pipeline on AWS.

Rather than just training a model in a notebook, this project focuses on the **engineering side of ML**: modular pipeline components, configuration-driven design, logging and exception handling, cloud integrations, and automated deployment — the practices that separate a prototype from a production-ready system.

## Key Features

- **Modular pipeline architecture** — clean separation of ingestion, validation, transformation, training, evaluation, and deployment components
- **MongoDB Atlas** integration for cloud-hosted data storage
- **Custom logging & exception handling** for traceable, debuggable pipelines
- **EDA & feature engineering** notebooks for data understanding
- **Schema-driven data validation**
- **AWS S3** for model artifact storage and versioning
- **Flask-based web app** for serving predictions
- **Dockerized** application for consistent deployments
- **CI/CD automation** with GitHub Actions
- **AWS EC2 + ECR** self-hosted deployment pipeline

## Tech Stack

| Category | Tools / Services |
|---|---|
| Language | Python 3.10 |
| Data Storage | MongoDB Atlas |
| Cloud Provider | AWS (S3, EC2, ECR, IAM) |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Web Framework | Flask (`app.py`) |
| Environment Management | Conda |
| Package Management | `setup.py`, `pyproject.toml` |

## Architecture / Workflow

```
Raw Data → MongoDB Atlas
     ↓
Data Ingestion
     ↓
Data Validation
     ↓
Data Transformation
     ↓
Model Training
     ↓
Model Evaluation ──→ AWS S3 (model registry)
     ↓
Model Pusher
     ↓
Prediction Pipeline (Flask App)
     ↓
Docker Image → GitHub Actions → AWS ECR → AWS EC2 (self-hosted runner)
     ↓
Live App on Port 5080
```

## Project Structure

A high-level look at how the codebase is organized once the pipeline is fully built out:

```
├── src/
│   ├── components/          # Core pipeline stages (ingestion, validation, transformation, training, evaluation, pusher)
│   ├── configuration/       # MongoDB & cloud connection configurations
│   ├── data_access/         # Data fetching and transformation utilities
│   ├── entity/               # Config, artifact entities, and estimator classes
│   ├── aws_storage/          # S3 push/pull utilities
│   ├── constants/            # Project-wide constants
│   └── utils/                 # Shared utility functions (main_utils, etc.)
├── notebook/                 # EDA, feature engineering, and MongoDB demo notebooks
├── static/                   # Static assets for the web app
├── templates/                 # HTML templates for the web app
├── config/
│   └── schema.yaml           # Data validation schema
├── app.py                    # Flask application entry point
├── demo.py                   # Demo/test script for logging, exceptions, ingestion
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── setup.py
├── pyproject.toml
└── crashcourse.txt           # Notes on setup.py / pyproject.toml
```

---

## Getting Started

### Prerequisites
- Python 3.10
- Conda (or another virtual environment manager)
- A MongoDB Atlas account
- An AWS account with permissions to create IAM users, S3 buckets, EC2 instances, and ECR repositories
- Docker installed (for containerization steps)
- A GitHub account (for CI/CD via GitHub Actions)

### Step 1: Project Template
Start by executing the `template.py` file to generate the initial project template, including the required folder structure and placeholder files.

```bash
python template.py
```

### Step 2: Package Management
Write the setup for importing local packages in the `setup.py` and `pyproject.toml` files.

> **Tip**: Learn more about how these files work in `crashcourse.txt`.

### Step 3: Virtual Environment and Dependencies
Create a virtual environment and install the required dependencies from `requirements.txt`:

```bash
conda create -n vehicle python=3.10 -y
conda activate vehicle
pip install -r requirements.txt
```

Verify that your local packages were installed correctly:

```bash
pip list
```

---

## Data Management (MongoDB)

### Step 4: MongoDB Atlas Configuration
1. Sign up for [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) and create a new project.
2. Set up a free **M0 cluster**, configure a username and password, and allow network access from any IP address (`0.0.0.0/0`).
3. Retrieve the MongoDB connection string for Python and save it — remember to replace `<password>` with your actual password.

### Step 5: Pushing Data to MongoDB
1. Create a folder named `notebook`, add your dataset to it, and create a notebook file `mongoDB_demo.ipynb`.
2. Use the notebook to push your dataset into the MongoDB database.
3. Verify that the data landed correctly in MongoDB Atlas under **Database → Browse Collections**.

---

## Logging, Exceptions & EDA

### Step 6: Set Up Logging and Exception Handling
Create custom logging and exception handling modules, and test them using the demo file `demo.py`. Good logging and exception handling early on makes every downstream pipeline stage far easier to debug.

### Step 7: Exploratory Data Analysis (EDA) and Feature Engineering
Analyze and engineer features inside the `EDA` and `Feature Engg` notebook. This step informs how the data ingestion and transformation components should be designed later in the pipeline.

---

## Data Ingestion

### Step 8: Data Ingestion Pipeline
1. Define MongoDB connection functions in `configuration.mongo_db_connections.py`.
2. Develop the data ingestion components in `data_access` and `components.data_ingestion.py` to fetch and transform raw data.
3. Update `entity/config_entity.py` and `entity/artifact_entity.py` with the relevant ingestion configurations.
4. Set your MongoDB connection as an environment variable, then run `demo.py` to test the pipeline.

**Setting the MongoDB environment variable:**

```bash
# Bash
export MONGODB_URL="mongodb+srv://<username>:<password>...."

# PowerShell
$env:MONGODB_URL = "mongodb+srv://<username>:<password>...."
```

> **Note**: On Windows, you can also set environment variables through the System Settings GUI.

---

## Data Validation, Transformation & Model Training

### Step 9: Data Validation
Define the expected data schema in `config.schema.yaml`, and implement the corresponding validation functions in `utils.main_utils.py`.

### Step 10: Data Transformation
Implement the data transformation logic in `components.data_transformation.py`, and create the `estimator.py` file inside the `entity` folder.

### Step 11: Model Training
Define and implement the model training steps in `components.model_trainer.py`, using the estimator class built in `estimator.py`.

---

## AWS Setup for Model Evaluation & Deployment

### Step 12: AWS Setup
1. Log in to the AWS console, create an IAM user, and grant it `AdministratorAccess`.
2. Set your AWS credentials as environment variables:

```bash
# Bash
export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
```

3. Configure your S3 bucket details and add the access keys in `constants.__init__.py`.

> **Security tip**: Never commit AWS credentials directly into your codebase. Use environment variables or GitHub Secrets (covered in the CI/CD section below).

### Step 13: Model Evaluation and Pushing to S3
1. Create an S3 bucket named `my-model-mlopsproj` in the `us-east-1` region.
2. Develop the code to push and pull trained models to/from the S3 bucket in `src.aws_storage` and `entity/s3_estimator.py`.

---

## Model Evaluation, Model Pusher, and Prediction Pipeline

### Step 14: Model Evaluation & Model Pusher
Implement the model evaluation component (to compare new models against the currently deployed one) and the model pusher component (to promote and store the accepted model). Then build out the **Prediction Pipeline** and set up `app.py` for API integration.

### Step 15: Static and Template Directory
Add `static` and `templates` directories to support the web UI served by the Flask application.

---

## CI/CD Setup with Docker, GitHub Actions, and AWS

### Step 16: Docker and GitHub Actions
1. Create a `Dockerfile` and `.dockerignore` file for the application.
2. Set up GitHub Actions, and configure AWS authentication by adding the following secrets in your GitHub repository settings:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_DEFAULT_REGION`
   - `ECR_REPO`

### Step 17: AWS EC2 and ECR
1. Set up an EC2 instance to serve as the deployment target.
2. Install Docker on the EC2 machine.
3. Connect the EC2 instance to your GitHub repository as a **self-hosted runner**.

### Step 18: Final Steps
1. Open port `5080` on the EC2 instance's security group.
2. Access your deployed application by visiting:

```
http://<public_ip>:5080
```

---

## Project Workflow Summary

1. **Data Ingestion** → **Data Validation** → **Data Transformation**
2. **Model Training** → **Model Evaluation** → **Model Deployment**
3. **CI/CD Automation** with GitHub Actions, Docker, AWS EC2, and ECR

---

## Additional Resources

- **Crash Course on `setup.py` and `pyproject.toml`**: See `crashcourse.txt` for details.
- **GitHub Secrets**: Used to securely manage credentials for the CI/CD pipeline.

---

## Connect

If you found this project helpful, have questions, or want to collaborate — feel free to reach out or open an issue!

---

*This README provides a structured walkthrough of the MLOps project, showcasing the end-to-end pipeline, cloud integration, CI/CD setup, and robust data handling capabilities.*
