def classify_trade(r):
    if r > 5:
        return "Big Win"
    elif r > 0:
        return "Small Win"
    else:
        return "Loss"


def backtest(data, signals):
    position = None
    entry_price = 0
    trades = []

    for signal, date in signals:
        price = data.loc[data.index == date, 'Close'].values[0]

        print(f"{signal} at {date} → Price: {price}")  # DEBUG

        if signal == "BUY" and position is None:
            position = "HOLD"
            entry_price = price

        elif signal == "SELL" and position == "HOLD":
            profit = float((price - entry_price) / entry_price * 100)

            # ✅ ADD QUALITY
            quality = classify_trade(profit)

            trades.append({
                "return": profit,
                "quality": quality
            })

            position = None

    return trades


def calculate_metrics(trades):
    if len(trades) == 0:
        return {
            "Total Return (%)": 0,
            "Win Rate (%)": 0,
            "Number of Trades": 0
        }

    total_return = sum(t["return"] for t in trades)
    wins = sum(1 for t in trades if t["return"] > 0)

    return {
        "Total Return (%)": total_return,
        "Win Rate (%)": (wins / len(trades)) * 100,
        "Number of Trades": len(trades)
    }