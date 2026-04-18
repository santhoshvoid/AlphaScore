"""
advanced_metrics.py — Extended performance analytics for AlphaCross
"""


def calculate_cagr(total_return_pct, start_date, end_date):
    """
    Compound Annual Growth Rate.
    CAGR = (1 + R)^(1/years) - 1
    Returns 0.0 if there is less than 2 days of data.
    """
    try:
        days = (end_date - start_date).days
        if days <= 1:
            return 0.0
        years = days / 365.25
        cagr  = ((1 + total_return_pct / 100) ** (1.0 / years) - 1) * 100
        return round(cagr, 2)
    except Exception:
        return 0.0


def calculate_max_drawdown(prices):
    """
    Maximum peak-to-trough drawdown as a % of the peak price.
    """
    if not prices or len(prices) < 2:
        return 0.0
    peak   = prices[0]
    max_dd = 0.0
    for p in prices:
        if p > peak:
            peak = p
        if peak > 0:
            dd = (peak - p) / peak * 100
            if dd > max_dd:
                max_dd = dd
    return round(max_dd, 2)


def calculate_sharpe(detailed_trades, risk_free_annual=6.0):
    """
    Simplified trade-level Sharpe Ratio.

    Uses per-trade returns. Risk-free rate is annualised then
    approximated to a per-trade equivalent assuming ~20 trading days
    average hold per trade (252 trading days / year).

    Returns 0.0 when there are fewer than 2 trades.
    Clamps the result to [-10, 10] to prevent explosion when
    two trades have nearly identical returns (std ≈ 0).
    """
    if not detailed_trades or len(detailed_trades) < 2:
        return 0.0

    returns = [t["return"] for t in detailed_trades]
    n       = len(returns)
    avg     = sum(returns) / n

    variance = sum((r - avg) ** 2 for r in returns) / (n - 1)
    std      = variance ** 0.5

    # Floor: prevent division by near-zero std
    # (happens when both trades have almost identical returns)
    MIN_STD = 0.5   # 0.5% minimum standard deviation
    if std < MIN_STD:
        std = MIN_STD

    rf_per_trade = risk_free_annual / (252 / 20)   # ≈ 0.476 % per trade
    sharpe       = (avg - rf_per_trade) / std

    # Hard clamp: a Sharpe outside [-10, 10] is meaningless for 2–4 trades
    sharpe = max(-10.0, min(10.0, sharpe))

    return round(sharpe, 2)


def get_best_worst_trade(detailed_trades):
    """Returns (best_return_pct, worst_return_pct)."""
    if not detailed_trades:
        return 0.0, 0.0
    returns = [t["return"] for t in detailed_trades]
    return round(max(returns), 2), round(min(returns), 2)


def calculate_portfolio_growth(detailed_trades, initial_capital):
    """
    Simulates compounding portfolio growth through all trades.
    Buys at every Golden Cross, sells at every Death Cross.
    Returns a dict with history list and final stats.
    """
    capital = float(initial_capital)

    history = [{"date": "Start", "capital": round(capital, 2), "label": "Initial"}]

    for trade in detailed_trades:
        capital   = capital * (1.0 + trade["return"] / 100.0)
        sell_label = str(trade.get("sell_date", "")).replace(" 00:00:00", "")
        history.append({
            "date":         sell_label,
            "capital":      round(capital, 2),
            "label":        trade.get("quality", ""),
            "trade_return": trade["return"],
        })

    final          = round(capital, 2)
    total_gain     = round(final - float(initial_capital), 2)
    total_gain_pct = round(
        (final - float(initial_capital)) / float(initial_capital) * 100, 2
    )

    return {
        "history":        history,
        "final":          final,
        "total_gain":     total_gain,
        "total_gain_pct": total_gain_pct,
    }