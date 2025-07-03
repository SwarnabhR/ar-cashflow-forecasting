# src/visualization/plotter.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_cash_inflow(df: pd.DataFrame, output_path: str = None):
    """
    Plots daily cash inflow and rolling average

    Args:
        df (pd.DataFrame): Takes the cleaned data as input
        output_path (str, optional): Path where the plotting must be stored. Defaults to None.
    """
    
    df = df.copy()
    if 'CashInflow' not in df.columns:
        raise KeyError("Missing 'CashInflow' column in DataFrame.")

    df['Rolling_7d'] = df['CashInflow'].rolling(window=7).mean()
    
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df, x='Date', y='CashInflow', label='Daily Cash Inflow', color='skyblue')
    sns.lineplot(data=df, x='Date', y='Rolling_7d', label='7-Day Rolling Avg', color='orange')
    plt.title("Daily Cash Inflow with Rolling Average")
    plt.xlabel("Date")
    plt.ylabel("Cash Inflow (Close x Volume)")
    plt.legend()
    plt.grid(True)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[INFO] Plot saved to {output_path}")
    else:
        plt.show()