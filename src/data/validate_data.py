import great_expectations as ge
from typing import Tuple, List
import pandas as pd

def validate_data(df:pd.DataFrame) -> Tuple[bool, List[str]]:
    """
        Data Validation using Great Expectations
        It implements critical data validation checks 
        like data integrity, business logic constraints, 
        and statistical properties that ML model expects
        before passing to the ML model.

        Arguments
            df(pd.DataFrame): dataset as dataframe

        Returns:
            A tuple of (validity, list of error messages).
    """
    print("Initiating Data Validation with Great Expectations...")

    # Checking whether df is empty or not
    if df is None or df.empty:
        raise ValueError("Pipeline stopped. The DataFrame is empty")

    # Create a copy so that original file is untouched.
    df_check = df.copy()

    # Handling the missing value in df_check so that it wont throw error during validation.
    if "TotalCharges" in df_check.columns and pd.api.types.is_string_dtype(df_check["TotalCharges"]):
        df_check["TotalCharges"] = pd.to_numeric(
            df_check["TotalCharges"].str.strip(), errors="coerce"
            ).fillna(0).astype(float)

    # Force MonthlyCharges to match the same float64 standard configuration layer
    if "MonthlyCharges" in df_check.columns:
        df_check["MonthlyCharges"] = pd.to_numeric(df_check["MonthlyCharges"], errors="coerce").fillna(0).astype("float64")

    # dataframe to great Expectation dataset
    df_ge = ge.from_pandas(df_check)

    # SCHEMA VALIDATION

    print("Checking Schema Definitions.....")
    # Checking whether the required columns exist or not
    required_cols = ["customerID", "gender", "Partner", "Dependents", "PhoneService",
                    "InternetService", "Contract", "tenure", "MonthlyCharges", "TotalCharges"]
    
    for col in required_cols:
        df_ge.expect_column_to_exist(col)

    # BUSINESS LOGIC VALIDATION - Setting Expectations to the Column Values

    print("Auditing Business Constraints.....")
    df_ge. expect_column_values_to_not_be_null("customerID")
    df_ge.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    df_ge.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    df_ge.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    df_ge.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    df_ge.expect_column_values_to_be_in_set("Contract", ['Month-to-month', 'One year', 'Two year'])
    df_ge.expect_column_values_to_be_in_set("InternetService", ['DSL', 'Fiber optic', 'No'])

    #  NUMERIC RANGE & STATISTICAL VALIDATION

    print("Verifying Continuous Value Limits.....")
    df_ge.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    df_ge.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
    df_ge.expect_column_values_to_be_between("TotalCharges", min_value=0)
    df_ge. expect_column_values_to_not_be_null("tenure")
    df_ge. expect_column_values_to_not_be_null("MonthlyCharges")

    # DATA CONSISTENCY VALIDATION

    print("Checking Data Consistency.....")
    df_ge.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges", column_B="MonthlyCharges", or_equal=True, mostly=0.95
    )

    # VALIDATION SUITE

    print("Running complete validation.....")
    results = df_ge.validate()

    # Extract all failed Expectations for thorough log reporting 
    failed_expectations = [
        r["expectation_config"]["expectation_type"]
        for r in results["results"] if not r["success"]
    ]

    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])

    if results["success"]:
        print(f"Data Validation Passed: {passed_checks}/{total_checks}")
    else:
        print(f"Data Validation Failed: {total_checks-passed_checks}/{total_checks}")
        print(f"Failed Criteria Items: {failed_expectations}")

    # returning whether success or not and failed_expectaions
    return results["success"], failed_expectations
    

    