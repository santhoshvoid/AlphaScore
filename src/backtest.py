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


def backtest(data, signals, initial_capital=100000):
    position = None
    entry_price = 0.0
    capital = initial_capital

    trades = []
    equity_curve = [{"date": "Start", "capital": capital}]

    for signal, date in signals:
        price = float(data.loc[date, "Close"])

        if signal == "BUY" and position is None:
            position = "HOLD"
            entry_price = price

        elif signal == "SELL" and position == "HOLD":
            ret_pct = ((price - entry_price) / entry_price)

            # ✅ CAPITAL UPDATE (REAL TRADING)
            capital = capital * (1 + ret_pct)

            trades.append({
                "return_pct": round(ret_pct * 100, 4),
                "capital": round(capital, 2)
            })

            equity_curve.append({
                "date": date.strftime("%Y-%m-%d"),
                "capital": round(capital, 2)
            })

            position = None

    return trades, equity_curve, capital


def calculate_metrics(trades, initial_capital, final_capital):
    if not trades:
        return {
            "Total Return (%)": 0.0,
            "Win Rate (%)": 0.0,
            "Number of Trades": 0,
        }

    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
    wins = sum(1 for t in trades if t["return_pct"] > 0)

    return {
        "Total Return (%)": round(total_return_pct, 4),
        "Win Rate (%)": round((wins / len(trades)) * 100, 2),
        "Number of Trades": len(trades),
    }