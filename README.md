# Telco Customer Churn — MLE Project
Binary Classification with full lifecycle:

* Tracking + registry: MLflow
* Data checks: Great Expectations
* APi: FastAPI
* CI: GitHub Actions
* Container: Docker
* Cloud Target: AWS ECS

# Quick Start
* uv: see below
* vent fallback: see below


Project Flow for Classification Problem

1. Load Data using Pandas
2. Initial EDA(Exploratory Data Analysis)
3. Drop Irrelevant Columns
4. Datatype Correction.
5. Data Conversion(All values to numeric - Binary Encoding, One-Hot Encoding)
6.   Data Cleaning(Check & Handle Missing Data, Duplicate rows)
7. Correlation & Multicollinearity
8. Save processed data(optional- use only if processing the data in a copy of the original data)
9. Load the processed data( only do if 8 is done.)
10. Train/Test Split
11. Feature Scaling
12. Model Exploration, Cross Validation, Evaluation Metrics - (Choose the baseline model)
13. Hyperparameter Tuning
14. Fit, Predict

