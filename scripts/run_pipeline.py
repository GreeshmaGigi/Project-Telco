"""
 Master pipeline 
 Sequentially executes load -> validate -> preprocess -> feature engineering -> train -> evaluate
 """

# importing required libraries
import os
import sys
import time
# argparse to read configs passed directly from the terminal
import argparse
# json package to clean, write and export dict.
import json
# joblib to serialise and save active python objects to the disk.
import joblib
import pandas as pd
import mlflow

# main project root folder lookup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# import path for local modules
# Load data from the disk
from src.data.load_data import load_data
# data quality check using Great Expectation
from src.data.validate_data import validate_data
# cleaning and preprocessing
from src.data.preprocess import data_preprocess
# feature engineering
from src.features.build_features import build_features
# hyperparameter tuning using Optuna
from src.models.tune import tune_model
# fitting
from src.models.train import train_model
# evaluation
from src.models.evaluate import evaluate_model
# logging
from src.utils.utils import setup_logger

def main(args):
    # pipeline start time
    pipeline_start_time = time.time()

    # log directory
    log_file_path = os.path.join(PROJECT_ROOT, "logs", "pipeline.log")
    #  setting up the logs
    logger = setup_logger(name="TelcoPipeline", log_file=log_file_path)

    # Initialising the pipeline
    logger.info("Initialising Telco Churn Prediction pipeline using LightGBM......")

    # MLFlow tracking
    mlruns_path = args.mlflow_uri or f"sqlite:///{PROJECT_ROOT}/mlflow.db"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    # Starting mlflow
    with mlflow.start_run():
        mlflow.log_param("model_type", "lightgbm")
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("tuning_enabled", args.tune)

        # PHASE 1: Data Ingestion and Quality Validation

        # Loading data
        logger.info("Loading Data......")
        df = load_data(args.input)
        logger.info(f"Successfully loaded data of dimensions {df.shape}")

        # Data validation using Great Expectations
        logger.info("Validating data using Great Expectations.....")
        is_valid, failed = validate_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        # in case where data validation failed
        if not is_valid:
            logger.error(f"Data Validation Anomalies detected: {failed} ")
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Data quality check failed. Issues detected: {failed}")

        logger.info("Data validation checks successful.")

        # PHASE 2: Preprocessing

        logger.info("Removing irrelevant columns and cleanign invalid string layouts")
        # cleaning and preprocessing
        df = data_preprocess(df)

        # Save the processed file to the processed folder
        processed_path = os.path.join(PROJECT_ROOT, "data", "processed", "Telco-Customer-Churn-Processed.csv")
        # create if processed folder does not exist
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        # Save a copy of the cleaned and processed data to disk
        df.to_csv(processed_path, index=False)

        logger.info(f"Preprocessed dataset saved to disk: {processed_path}")

        # PHASE 3: Feature Engineering

        logger.info("Applying Binary Encoding and One-Hot Encoding to the data.....")
        # assign the target label from argument lists
        target = args.target
        # Check if target_column present or not
        if target not in df.columns:
            raise ValueError(f"Target variable '{target}' missing.")

        # feature engineering
        df_fe = build_features(df, target_column=target)
        logger.info(f"Feature Engineering completed and resulting dimensions of data: {df_fe.shape}")

        # PHASE 4: Metadata Tracking phase

        artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # list of feature columns
        feature_columns = list(df_fe.drop(columns=[target]).columns)

        # create feature_columns.json -> features metadata
        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_columns, f)

        mlflow.log_text("\n".join(feature_columns), artifact_file = "feature_columns.txt")

        # PHASE 5: Hyperparameter tuning using Optuna

        # check Tuning requested or not
        if args.tune:
            # tuning start time
            tune_start_time = time.time()
            logger.info("Starting Hyperparameter tuning using Optuna.....")
            best_parameters = tune_model(df_fe, target_column=target)
            # tuning end time
            tune_time = time.time() - tune_start_time
            logger.info("Hyperparameter successfully completed.")
            print(f"Optuna Hyperparameter tuning time: {tune_time:.2f} seconds.")
            mlflow.log_metric("total_tuning_time_seconds", tune_time)
        else:
            best_parameters=None
            logger.info("Utilising the best sandbox parameter defaults.")

        # PHASE 6: Model training

        # Start Training
        logger.info("Starting final model training.....")

        start_time = time.time()

        model, scaler = train_model(df_fe, target_column=target, custom_params=best_parameters)

        # training time
        train_time = time.time() - start_time

        logger.info("Model training successfully completed.")
        print(f"Model Training time: {train_time:.2f} seconds")
        mlflow.log_metric("total_train_time_seconds", train_time)

        # save the scaler artifact to disk for serving i/p transformations consistency
        scaler_artifact_path = os.path.join(artifacts_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_artifact_path)
        mlflow.log_artifact(scaler_artifact_path)

        # For FASTAPI Serving Layer
        model_artifact_path = os.path.join(artifacts_dir, "model.pkl")
        joblib.dump(model, model_artifact_path)

        # save general preprocessing metadat dictionary layouts
        preprocessing_metadata ={
            "feature_columns": feature_columns,
            "target": target
        }
        joblib.dump(preprocessing_metadata, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        logger.info("Scaler and preprocessing artifacts saved to the the artifacts directory.")

        # STEP 7: Performance Evaluaation

        logger.info("Starting model evaluation.....")
        # splitting features and target
        x_eval = df_fe.drop(columns=[target])
        y_eval = df_fe[target]

        x_eval_scaled = scaler.transform(x_eval)

        evaluate_model(model, x_eval_scaled, y_eval)
        logger.info("Telco Churn Prediction pipeline using LightGBM completed successfully.")

        # pipeline end time
        pipeline_end_time = time.time() - pipeline_start_time

        print(f"Total pipeline execution time: {pipeline_end_time:.2f} seconds.")
        mlflow.log_metric("Total_pipeline_execution_time_seconds", pipeline_end_time)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Production LightGBM End-to-End Pipeline Instance")
    p.add_argument("--input", type=str, required=True,
                   help="path to raw data CSV (e.g. data/raw/Telco-Customer-Churn.csv")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco_Customer_Churn_Prediction")
    p.add_argument("--tune", action="store_true",
                   help="If passed, runs Optuna tuning optimisation before model training.")
    p.add_argument("--mlflow_uri", type=str, default=None,
                   help="override tracking URI engine path setting targets")

    args = p.parse_args()
    main(args)

