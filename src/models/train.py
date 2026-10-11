import mlflow
import mlflow.lightgbm
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_model(df: pd.DataFrame, target_column:str, custom_params: dict = None):
    """
        LighGBM model training. 
        Feature Scaling using StandardScaler
        Logs with mlflow

        Arguments:
            df(pd.Dataframe): Encoded feature engineered dataset.
            target_column(str):  Name of target variable("Churn)
            custom_params(dict, optional): Dynamic parameters from tune

        Returns:
            model(LGBMClassifier): Trained LightGBM model
            scaler(StandardScaler): Fitted feature scaler

    """
    x = df.drop(columns=[target_column])
    y = df[target_column]

    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    # Feature Scaling - StandardScaler
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # model parameters
    if custom_params:
        model_params = custom_params.copy()
    else:
        # sandbox winning parameters
        model_params = {
            'max_depth': 4,
            'num_leaves': 10,
            'scale_pos_weight': 3.96992056841012,
            'learning_rate': 0.010679592221992763,
            'n_estimators': 300,
            'subsample': 0.8694816242977589,
            'colsample_bytree': 0.5210773277288029,
            'reg_alpha': 0.004046992198811344,
            'reg_lambda': 0.027130564722099078
            }

    # core system configuration keys
    model_params.update(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
    )

    # training
    model = LGBMClassifier(**model_params)


    model.fit(x_train_scaled, y_train)

    # predictions
    predictions = model.predict(x_test_scaled)

    # evaluations
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    # Logging
    mlflow.log_params(model_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("recall", recall)

    # Log the native LightGBM tracking module artifact
    mlflow.lightgbm.log_model(model, name="model")

    # Log dataset metadata for tracking data lineage in MLflow
    training_ds = mlflow.data.from_pandas(df, source="training_data")
    mlflow.log_input(training_ds, context="training")

    print(f"LightGBM Model trained successfully. Accuracy: {accuracy:.4f}, Recall: {recall:.4f}")

    return model, scaler