# src/ingestion/data_loader.py

import pandas as pd
import yfinance as yf
import os

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily stock data using yfinance.

    Args:
        ticker (str): Stock symbol (e.g., 'AAPL', 'DHER.DE')
        start_date (str): 'YYYY-MM-DD'
        end_date (str): 'YYYY-MM-DD'

    Returns:
        pd.DataFrame: DataFrame with columns [Date, Close, Volume]
    """
    print(f"[INFO] Fetching stock data for {ticker} from {start_date} to {end_date}")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker} in the given date range.")

    # Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Reset index to bring Date into column
    df.reset_index(inplace=True)
    print("[DEBUG] Flattened columns:", df.columns.tolist())
    return df[['Date', 'Close', 'Volume']]

def simulate_cash_inflow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate AR cash inflow as: CashInflow = Close x Volume

    Args:
        df (pd.DataFrame): Must contain 'Close' and 'Volume' columns

    Returns:
        pd.DataFrame: With additional 'CashInflow' column
    """
    df = df.copy()
    if 'Close' not in df.columns or 'Volume' not in df.columns:
        raise KeyError("Missing 'Close' or 'Volume' in dataframe to simulate cash inflow.")
    
    df['CashInflow'] = df['Close'] * df['Volume']
    
    return df

def save_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to a CSV file. automatically creates folders if needed.

    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path to the output file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Data saved to {output_path}")