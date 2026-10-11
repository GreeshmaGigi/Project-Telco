import pandas as pd

def map_binary_series(s: pd.Series) -> pd.Series:
    """
        Binary encoding to 2-category features.

        Arguments:
            s(pd.Series): series of binary category column.

        Returns:
            s(pd.Series): binary coded Series
    """

    # Get unique values and clean string format
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # Yes/No mapping
    if valset.issubset({"Yes", "No"}):
        return s.map({"No":0, "Yes":1})

    # Gender mapping
    if valset.issubset({"Male", "Female"}):
        return s.map({"Male":1, "Female":0})

    # General fallback for other 2-category feature
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]:0, sorted_vals[1]:1}
        return s.astype(str).map(mapping)

    # if none of the ifs work it return the s
    return s

def build_features(df: pd.DataFrame, target_column: str = "Churn") -> pd.DataFrame:
    """
        Feature Engineering for training data.

        Arguments:
            df(DataFrame): input DataFrame
            target_column(str): Name of target variable("Churn)

        df(pd.DataFrame): cleaned DataFrame
    """

    # Copy of df
    df = df.copy()
    print(f"Starting Feature Engineering on {df.shape[1]} columns.....")

    # Identify Feature Types
    object_columns = [
        c for c in df.select_dtypes(include=["object"]).columns
        if c != target_column
    ]

    numeric_columns = [
        c for c in df.select_dtypes(include=["int64", "float64"]).columns
    ]

    print(f"Found {len(object_columns)} categorical and {len(numeric_columns)} numeric columns")

    # Split Categorical by Cardinality
    binary_columns = [
        c for c in object_columns if df[c].dropna().nunique() ==2
    ]
    multi_columns = [
        c for c in object_columns if df[c].dropna().nunique() > 2
    ]
    
    print(f"{binary_columns} Binary features and {multi_columns}  Multi-category features.")


    # Binary Encoding
    if binary_columns:
        print(f"Applying binary encoding to {len(binary_columns)} binary columns.....")
        for c in binary_columns:
            original_dtype = df[c].dtype
            df[c] = map_binary_series(df[c])
            print(f"{c}: {original_dtype} -> binary(0/1)")

    #  Map the Churn target column from string text to binary 0 and 1
    if target_column in df.columns:
        print(f"Applying binary encoding to the targrt column '{target_column}'.....")
        original_dtype = df[target_column].dtype
        df[target_column] = map_binary_series(df[target_column])
        print(f"{target_column}: {original_dtype} -> binary(0/1)")

    # One-Hot Encoding
    if multi_columns:
        print(f"Applying One-Hot Encoding to {len(multi_columns)} multi-category columns.....")
        original_column_num = df.shape[1]
        df = pd.get_dummies(df, columns=multi_columns, drop_first=True)
        #  current col no - original col no + len(multi_col) because we are dropping first column
        new_features = df.shape[1] - original_column_num + len(multi_columns)
        print(f"Create {new_features} new features from {len(multi_columns)} categorical columns.")


    # Global Integer Conversion
    all_features = [
        c for c in df.columns if c != target_column
    ]
    for c in all_features:
        if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int)
    print(f"Feature engineering done. {df.shape[1]} final features.")

    return df