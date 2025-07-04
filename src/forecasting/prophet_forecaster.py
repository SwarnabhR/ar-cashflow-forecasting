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

def plot_forecast_with_aging(forecast: pd.DataFrame, df_actual, aging_df: pd.DataFrame = None,title = "Forecast vs Actual"):
    """
    Plot the forecast with CI and actuals (optional)

    Args:
        forecast (pd.DataFrame): Output from Prophet's predict()
        df_actual (pd.DataFrame, optional): Original data with 'Date' and 'CashInflow'. Defaults to None.
        title (str, optional): Plot title. Defaults to "Forecast vs Actual".
    """
    plt.figure(figsize=(14, 7))
    
    # Plot actual inflows as bars
    plt.bar(aging_df['Date'], aging_df['bucket_0_30'], label='0-30d', color= '#4caf50')
    plt.bar(aging_df['Date'], aging_df['bucket_31_60'], bottom=aging_df['bucket_0_30'], label='31-60d', color='#ff9800')
    plt.bar(aging_df['Date'], aging_df['bucket_61_90'], bottom=aging_df['bucket_0_30'] + aging_df['bucket_31_60'], label='61-90d', color='#f44336')
    plt.bar(aging_df['Date'], aging_df['bucket_91_plus'], bottom=aging_df['bucket_0_30'] + aging_df['bucket_31_60'] + aging_df['bucket_61_90'], label='90+ d', color='#9c27b0')
    
    # Forecast prediction
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast (yhat)', color='orange', linewidth=2)
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.2, label='Confidence Interval')
    
    # Historical actuals
    if df_actual is not None:
        plt.plot(df_actual['Date'], df_actual['CashInflow'], label='Actual Inflow', color='blue')
    
    # Basics
    plt.xlabel("Date")
    plt.ylabel("Cash Inflow")
    plt.title(title)
    plt.legend()
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