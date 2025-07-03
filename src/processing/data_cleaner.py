# src/processing/data_cleaner.py

import pandas as pd

def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the stock DataFrame to ensure it has:
    - 'Date', 'Close', 'Volume' columns
    - Proper types
    - No NaNs or invalid rows

    Args:
        df (pd.DataFrame): Stock Price dataframe

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    
    df=df.copy()
    print("[DEBUG] Incoming columns:", df.columns.tolist())
    
    # Lowercase and trim column names for matching
    lower_cols = {col: col.lower().strip() for col in df.columns}
    df.rename(columns = lower_cols, inplace=True)
    
    # Match the actual columns by content
    date_col = next((col for col in df.columns if 'date' in col), None)
    close_col = next((col for col in df.columns if 'close' in col), None)
    volume_col = next((col for col in df.columns if 'volume' in col), None)
    
    print(f"[DEBUG] Mapped: Date -> {date_col}, Close -> {close_col}, Volume -> {volume_col}")
    
    if not all([date_col, close_col, volume_col]):
        raise KeyError(f"Missing required columns. Available: {df.columns.tolist()}")
    
    # Rename to standard names
    df.rename(columns={
        date_col: 'Date',
        close_col: 'Close',
        volume_col: 'Volume'
    }, inplace=True)
    
    # Drop missing and fix types
    df.dropna(subset=['Date', 'Close', 'Volume'],inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors = 'coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df.dropna(subset=['Date', 'Close', 'Volume'], inplace=True)
    
    # Filter invalid entries
    df = df[df['Volume'] > 0]
    
    # Sort
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df