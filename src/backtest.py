"""
backtest.py — Simulates Buy-at-GoldenCross / Sell-at-DeathCross strategy
"""


def classify_trade(r):
    if r > 5:
        return "Big Win"
    elif r > 0:
        return "Small Win"
    else:
        return "Loss"


def backtest(data, signals):
    """
    Walks through every crossover signal and simulates trades.
    BUY on Golden Cross, SELL on Death Cross.
    Returns a list of trade dicts with 'return' and 'quality' keys.
    """
    position    = None
    entry_price = 0.0
    trades      = []

    for signal, date in signals:
        # Use .loc for label-based access; scalar() converts Series to float
        price_raw = data.loc[date, "Close"]
        price     = float(price_raw.iloc[0]) if hasattr(price_raw, "iloc") else float(price_raw)

        if signal == "BUY" and position is None:
            position    = "HOLD"
            entry_price = price

        elif signal == "SELL" and position == "HOLD":
            profit  = (price - entry_price) / entry_price * 100
            quality = classify_trade(profit)
            trades.append({
                "return":  round(profit, 4),
                "quality": quality,
            })
            position = None

    return trades


def calculate_metrics(trades):
    """
    Summarise a list of trade dicts into key performance metrics.
    """
    if not trades:
        return {
            "Total Return (%)":  0.0,
            "Win Rate (%)":      0.0,
            "Number of Trades":  0,
        }

    total_return = sum(t["return"] for t in trades)
    wins         = sum(1 for t in trades if t["return"] > 0)

    return {
        "Total Return (%)": round(total_return, 4),
        "Win Rate (%)":     round((wins / len(trades)) * 100, 2),
        "Number of Trades": len(trades),
    }