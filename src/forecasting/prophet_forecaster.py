import pandas as pd
from prophet import Prophet
import os
import matplotlib.pyplot as plt

def prepare_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataframe to work with Prophet by renaming columns.

    Args:
        df (pd.DataFrame): Cleaned DataFrame

    Returns:
        pd.DataFrame: A new DataFrame with Renamed Columns
    """
    df = df.copy()
    if 'Date' not in df.columns or 'CashInflow' not in df.columns:
        raise KeyError("Missing required columns: 'Date' and 'CashInflow'")
    
    df = df[['Date', 'CashInflow']].rename(columns={'Date': 'ds', 'CashInflow': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    return df

def forecast_cash_inflow(df: pd.DataFrame, periods: int = 30) -> tuple:
    """
    Forecast future cash inflow using Prophet

    Args:
        df (pd.DataFrame): DataFrame with 'Date' and 'CashInflow'
        periods (int, optional): Number of days to forecast. Defaults to 30.

    Returns:
        tuple: forecast: DataFrame with predictions
               model: Trained Prophet model
    """
    df_prophet = prepare_for_prophet(df)
    
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods = periods)
    forecast = model.predict(future)
    
    return forecast, model

def plot_forecast(model, forecast, original_df=None, title="Cash Inflow Forecast"):
    """
    Plot forecast using Prophet.
    If original_df is provided, plot it alongside forecast.

    Args:
        model (object): Trained Prophet model
        forecast (pd.DataFrame): Actual prediction result
        original_df (pd.DataFrame, optional): Original DataFrame. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "Cash Inflow Forecast".
    """
    
    fig = model.plot(forecast)
    plt.title(title)
    
    if original_df is not None:
        plt.plot(original_df['Date'], original_df['CashInflow'], 'k.', alpha=0.4, label='Actuals')
        plt.legend()
    
    plt.xlabel("Date")
    plt.ylabel("Cash Inflow")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def export_forecast_to_csv(forecast: pd.DataFrame, path: str = "../data/forecast_30d.csv"):
    """Export forecast data to CSV.

    Args:
        forecast (pd.DataFrame): Forecasted DataFrame
        path (str, optional): Path where to save the CSV file. Defaults to "../data/forecast_30d.csv".
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    forecast.to_csv(path, index=False)
    print(f"[INFO] Forecast saved to {path}")