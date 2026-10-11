import pandas as pd

def data_preprocess(df:pd.DataFrame, target_column:str = "Churn") -> pd.DataFrame:
    """
        Preproccessing the data by checking and handling missing data,
        drop irrelevant columns, trim column names, fix TotalCharges,
        map target variable values to 0 and 1 and NA handling.

        Arguments:
            df(DataFrame):  raw input data
            target_column(str):  Name of target variable("Churn)
        
        Returns:
            df(pd.DataFrame): cleaned DataFrame
    """

    # Create a copy to isolate from raw data
    df = df.copy()

    # Tidy  column namesby stripping any whitespaces
    df.columns = df.columns.str.strip()

    # TotalCharges - convert to numeric (found during SandBoxing)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].str.strip(), errors="coerce"
        )

    # Filling missing value of TotalCharges with 0 given that tenure=0 indicates new cx
    if "tenure" in df.columns and "TotalCharges" in df.columns:
        df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0

    # Handle SeniorCitizen to 0/1 int if present
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    # Mapping target values to binary integers
    if target_column in df.columns and df[target_column].dtype == "object":
        df[target_column] = df[target_column].str.strip().map({"No":0, "Yes":1})

    # Safety fill all remaining numeric missing values with 0
    numeric_columns = df.select_dtypes(include=["number"]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    # Dropping irrelevant columns
    irrelevant_columns = ["CustomerID", "customer_id", "CustomerId", "Customer_ID", "Customer_Id", "customerID"]
    for col in irrelevant_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    print("Data Cleaning and Business logic mapping is done")

    # returns the cleaned DatFrame
    return df