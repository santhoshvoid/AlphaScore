from src.signals import detect_crossovers
import pandas as pd

def test_detect_crossovers():
    data = pd.DataFrame({
        "EMA20": [1,2,3,2,1],
        "EMA50": [2,2,2,2,2]
    })

    signals = detect_crossovers(data, "EMA20", "EMA50")

    assert isinstance(signals, list)