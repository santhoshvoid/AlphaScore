"""
backtest.py — Simulates Buy-at-GoldenCross / Sell-at-DeathCross strategy

Bug fixes:
  1. signal_map keys normalised to YYYY-MM-DD so signals are matched correctly.
  2. Open position at end of data: final_capital now returns the actual current
     market value (shares * last price) instead of the original cash value.
     Without this, a BUY with no matching SELL shows final value = initial_capital.
"""


def classify_trade(r):
    if r > 5:
        return "Big Win"
    elif r > 0:
        return "Small Win"
    else:
        return "Loss"


def _to_date_str(date_val) -> str:
    """
    Normalise any date-like value to a plain 'YYYY-MM-DD' string.
    Handles: pandas Timestamp, datetime.date, datetime.datetime, str.
    """
    if hasattr(date_val, 'date'):
        return str(date_val.date())
    s = str(date_val)
    return s[:10]


def backtest(data, signals, initial_capital=100000):
    capital      = initial_capital
    position     = False
    entry_price  = 0.0
    shares       = 0.0

    trades       = []
    equity_curve = []

    # Normalise signal keys to YYYY-MM-DD so they match the loop's date_str
    signal_map = {_to_date_str(date): signal for signal, date in signals}

    for date, row in data.iterrows():
        price    = float(row["Close"])
        date_str = _to_date_str(date)

        signal = signal_map.get(date_str, None)

        # BUY
        if signal == "BUY" and not position:
            position    = True
            entry_price = price
            shares      = capital / price   # full capital deployed

        # SELL
        elif signal == "SELL" and position:
            ret_pct = (price - entry_price) / entry_price
            capital = shares * price

            trades.append({
                "return_pct": round(ret_pct * 100, 4),
                "capital":    round(capital, 2)
            })

            position = False
            shares   = 0.0

        current_value = shares * price if position else capital
        equity_curve.append({
            "date":    date_str,
            "capital": round(current_value, 2)
        })

    # Fallback safety
    if not equity_curve:
        equity_curve.append({
            "date":    _to_date_str(data.index[0]),
            "capital": initial_capital
        })

    # BUG FIX: if still in an open position at end of data, `capital` was never
    # updated (no SELL fired), so it still equals initial_capital.
    # The equity_curve correctly tracked shares * price every day — use its
    # last entry as the true terminal portfolio value.
    if position and equity_curve:
        final_capital = equity_curve[-1]["capital"]
    else:
        final_capital = capital

    return trades, equity_curve, final_capital


def calculate_metrics(trades, initial_capital, final_capital):
    if not trades:
        return {
            "Total Return (%)": 0.0,
            "Win Rate (%)":     0.0,
            "Number of Trades": 0,
        }

    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
    wins = sum(1 for t in trades if t["return_pct"] > 0)

    return {
        "Total Return (%)": round(total_return_pct, 4),
        "Win Rate (%)":     round((wins / len(trades)) * 100, 2),
        "Number of Trades": len(trades),
    }