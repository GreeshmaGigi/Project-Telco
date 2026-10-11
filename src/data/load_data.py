import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
        Loads csv inton a dataframe

        Arguments:
            file_path(str): csv file path

        Returns:
            loaded csv as dataframe
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)