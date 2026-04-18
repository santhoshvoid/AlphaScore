"""
indicators.py — EMA calculation
Handles both a pandas Series and a plain Python list as input,
so it works correctly whether called from app.py or main.py.
"""
import pandas as pd


def calculate_ema(series, span):
    """
    Calculate Exponential Moving Average.

    Args:
        series: pandas Series OR plain Python list of close prices
        span:   EMA period (int), e.g. 20, 50, 200

    Returns:
        pandas Series of EMA values (same index as input if Series given)
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    return series.ewm(span=span, adjust=False).mean()