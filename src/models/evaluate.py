import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, x_test_scaled: np.ndarray, y_test:pd.Series):
    """
        Model evaluation, generates confusion matrix and evaluation metrics report.

        Arguments:
            model: LightGBM model
            x_test_scaled(np.ndarray): scaled test features from train.py
            y_test(pd.Series): raw test target labels
    """

    # Predictions
    predictions = model.predict(x_test_scaled)

    # Evaluation Metrics Report
    print("Evaluation Metrics Report")
    print(classification_report(y_test, predictions))

    # Confusion Matrix
    print("Confusion Matrix")
    print(confusion_matrix(y_test, predictions))