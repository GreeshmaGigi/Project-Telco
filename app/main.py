# importing necessary libraries
import os
import sys
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import gradio as gr

# file path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# importing local modules
from src.data.preprocess import data_preprocess
from src.features.build_features import build_features

# data quality check for new data through API using Pydantic
class CustomerInput(BaseModel):
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=1)
    PhoneService: str = Field(..., example="No")
    MultipleLines: str = Field(..., example="No phone service")
    InternetService: str = Field(..., example="DSL")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0.0, example=29.85)
    TotalCharges: str = Field(..., example="29.85")

class PredictionOutput(BaseModel):
    churn_prediction: int
    churn_probability: float
    status: str

# FastAPI
app = FastAPI(
    title="Telco Customer Churn Prediction Engine",
    description="Customer Churn Prediction Service",
    version="1.0.0"
)

# Global runtime variables to cache artifacts in memory
MODEL = None
SCALER = None
FEATURE_COLUMNS = None

@app.on_event("startup")
def load_production_artifacts():
    """
        setting up the artifacts for webserver 
    """
    global MODEL, SCALER, FEATURE_COLUMNS
    try:
        artifacts_path = os.path.join(PROJECT_ROOT, "artifacts")

        # Load aligned encoded column blueprint schema
        with open(os.path.join(artifacts_path, "feature_columns.json"), "r") as f:
            FEATURE_COLUMNS = json.load(f)

        # load scaling matrix configuration
        SCALER = joblib.load(os.path.join(artifacts_path, "scaler.pkl"))

        # load the trained model
        model_path = os.path.join(artifacts_path, "model.pkl")
        if os.path.exists(model_path):
            MODEL = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Production model not found at {model_path} ")

        print(f"All MLOps serving artifacts successfully mounted in memory.")

    except Exception as e:
        print(f"Initialisation failed. Error: {str(e)}")
        raise RuntimeError(e)

@app.get("/")
def health_check():
    """
        pulse check for web server
    """
    return {"status": "healthy", "model_loaded": MODEL is not None}

def run_pipeline_prediction(raw_dict: dict) -> tuple:
    """
        Transforms raw data and calculate predictions.

        Arguments:
            raw_dict(dict): single cx raw data

        Returns:
            (tuple): contains prediction and its probability(model confidence score)
    """
    if MODEL is None or SCALER is None or FEATURE_COLUMNS is None:
        raise ValueError("Serving runtime matrices are uninitialised.")

    # dict to Dataframe
    df_raw = pd.DataFrame([raw_dict])

    # data cleaning
    df_cleaned = data_preprocess(df_raw)

    # feature engineering and set target_column=None
    df_features = build_features(df_cleaned, target_column="Churn")

    # one hot encoding
    df_encoded = pd.get_dummies(df_features)

    df_aligned = pd.DataFrame(0, index=df_encoded.index, columns=FEATURE_COLUMNS)

    # handle unseen categorical or structural changes by matching missing columns
    for col in FEATURE_COLUMNS:
        if col in df_encoded.columns:
            df_aligned[col] = df_encoded[col].astype(float)
        else:
            df_aligned[col]=0.0

    # enforce the sequential arrangements of columns
    df_aligned = df_aligned[FEATURE_COLUMNS]

    # feature scaling 
    scaled_array = SCALER.transform(df_aligned)

    # prediction
    probability = float(MODEL.predict_proba(scaled_array)[:, 1][0])
    prediction = 1 if probability >= 0.5 else 0

    print(f"Success: Inference pipeline complete! Prediction: {prediction} with probability: {probability:.4f}")

    return prediction, probability

@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(customer: CustomerInput):
    """
        the microservice exposes REST API endpoints and 
        an interactive Gradio interface to evaluate 
        subscriber attrition risks in real time.
    """
    try:
        prediction, probability = run_pipeline_prediction(customer.model_dump())
        return {
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "status": "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")

# Gradio Interaction User Interface
def gradio_ui_wrapper(*args):
    """
        Maps array elements out of the web form grid interface
        to inference.
    """
    feature_keys = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", 
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
        "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ]

    raw_payload = dict(zip(feature_keys, args))

    # enforce precise runtime validation formatting types
    raw_payload["SeniorCitizen"] = int(raw_payload["SeniorCitizen"])
    raw_payload["tenure"] = int(raw_payload["tenure"])
    raw_payload["MonthlyCharges"] = float(raw_payload["MonthlyCharges"])
    raw_payload["TotalCharges"] = str(raw_payload["TotalCharges"])

    try:
        prediction, probability = run_pipeline_prediction(raw_payload)
        if prediction==1:
            return f" HIGH RISK: Customer Likely to Churn. Probability: {probability:.2%}"
        return f"SAFE: Customer Unlikely to Churn. Probability: {probability:.2%}"
    except Exception as e:
        return f"System Error Processing Request: {str(e)}"                                                       

# buliding interface
io_interface = gr.Interface(
    fn = gradio_ui_wrapper,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(["0", "1"], label="Senior Citizen (1=Yes, 0=No)"),
        gr.Dropdown(["Yes", "No"], label="Partner Status"),
        gr.Dropdown(["Yes", "No"], label="Dependents Status"),
        gr.Slider(minimum=0, maximum=100, step=1, value=12, label="Tenure (Months)"),
        gr.Dropdown(["Yes", "No"], label="Phone Service"),
        gr.Dropdown(["No phone service", "No", "Yes"], label="Multiple Lines Config"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service Class"),
        gr.Dropdown(["No", "Yes", "No internet service"], label="Online Security Shield"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup Matrix"),
        gr.Dropdown(["No", "Yes", "No internet service"], label="Device Protection Profile"),
        gr.Dropdown(["No", "Yes", "No internet service"], label="Tech Support Tier"),
        gr.Dropdown(["No", "Yes", "No internet service"], label="Streaming TV Access"),
        gr.Dropdown(["No", "Yes", "No internet service"], label="Streaming Movies Access"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract Framework"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing Plan"),
        gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method Category"),
        gr.Number(value=50.0, label="Monthly Billing Charges ($)"),
        gr.Textbox(value="600.0", label="Aggregate Total Charges Value ($)")
    ],
    outputs=gr.Textbox(label="Model Retention Verdict Output"),
    title="Customer Churn Diagnostic Workspace",
    description="Operational sandbox running real-time assertions against the production tree-based model engine."
)

# Mount Gradio sub-application onto FastAPI main cluster domain path route /ui
app = gr.mount_gradio_app(app, io_interface, path="/ui")
