import pandas as pd

def compute_aging_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate AR aging buckets from cash inflow.
    
    We'll assume:
        - 70% collected in 0-30 days
        - 20% collected in 31-60 days
        - 7% collected in 61-90 days
        - 3% delayed beyong 90 days

    Args:
        df (pd.DataFrame): Cleaned DataFrame to compute AR_aging by CashInflow

    Returns:
        pd.DataFrame: Added AR_aging column(s)
    """
    df = df.copy()
    df['bucket_0_30'] = df['CashInflow'] * 0.70
    df['bucket_31_60'] = df['CashInflow'] * 0.20
    df['bucket_61_90'] = df['CashInflow'] * 0.07
    df['bucket_91_plus'] = df['CashInflow'] * 0.03
    return df