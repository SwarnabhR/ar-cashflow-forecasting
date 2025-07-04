# src/evaluation/backtest.py

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def backtest_prophet(df: pd.DataFrame, forecast_days: int = 30):
    """
    Backtest Prophet on the last N days of data to evaluate forecast accuracy.

    Args:
        df (pd.DataFrame): DataFrame with columns ['Date', 'CashInflow']
        forecast_days (int, optional): Number of days to forecast and test. Defaults to 30.
    """
    df = df.copy().sort_values('Date').reset_index(drop=True)
    
    # Split into train and test
    train_df = df[:-forecast_days]
    test_df = df[-forecast_days:]
    
    #Format for Prophet
    prophet_df = train_df[['Date', 'CashInflow']].rename(columns={'Date': 'ds', 'CashInflow': 'y'})
    
    # Train Prophet 
    model = Prophet(
        daily_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=15.0
    )
    model.fit(prophet_df)
    
    # Make future df
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    # Merge with actuals
    merged = pd.merge(forecast[['ds', 'yhat']], test_df, left_on='ds', right_on='Date')
    
    #Evaluation
    y_true = merged['CashInflow']
    y_pred = merged['yhat']
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    metrics = {
        'MAPE': round(mape * 100, 2),
        'RMSE': round(rmse, 2)
    }
    return metrics, forecast, model