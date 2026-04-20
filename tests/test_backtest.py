from src.backtest import backtest
import pandas as pd

def test_backtest_runs():
    data = pd.DataFrame({
        "Close": [100, 105, 110, 120]
    })
    data.index = pd.date_range(start="2023-01-01", periods=4)

    signals = [("BUY", data.index[0]), ("SELL", data.index[-1])]

    trades, equity, final_capital = backtest(data, signals)

    assert final_capital > 0