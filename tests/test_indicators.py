from src.indicators import calculate_ema
import pandas as pd

def test_ema_output_length():
    data = pd.Series([1,2,3,4,5,6,7,8,9,10])
    ema = calculate_ema(data, 3)

    assert len(ema) == len(data)