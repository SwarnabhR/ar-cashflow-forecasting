import pandas as pd
import numpy as np

def detect_outliers_iqr(df: pd.DataFrame, column: str = 'CashInflow') -> pd.DataFrame:
    """
    Detect outliers using IQR method and label them.
    Adds a new column: 'Outlier_IQR' (True/False)

    Args:
        df (pd.DataFrame): Cleaned DataFrame of the Stocks Price
        column (str, optional): Column required to detect outliers. Defaults to 'CashInflow'.

    Returns:
        pd.DataFrame: A new DataFrame with a new column 'Outlier_IQR'
    """
    df = df.copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df['Outlier_IQR'] = (df[column] < lower_bound) | (df[column] > upper_bound)
    return df


def detect_outliers_zscore(df: pd.DataFrame, column: str = 'CashInflow', threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect outliers using Z-score method and label them.
    Adds a new column: 'Outlier_Z' (True/False).

    Args:
        df (pd.DataFrame): 
        coloumn (str, optional): _description_. Defaults to 'CashInflow'.
        threshold (float, optional): _description_. Defaults to 3.0.

    Returns:
        pd.DataFrame: _description_
    """
    df = df.copy()
    mean = df[column].mean()
    std = df[column].std()
    
    df['Z_score'] = (df[column] - mean) / std
    df['Outlier_Z'] = df['Z_score'].abs() > threshold
    return df