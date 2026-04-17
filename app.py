from flask import Flask, render_template, request, jsonify
from src.data_fetch import get_stock_data, store_price_data
from src.indicators import calculate_ema
from src.signals import detect_crossovers
from src.backtest import backtest, calculate_metrics
from src.analysis import signal_outcome_analysis, summarize_results, build_signal_table
from src.symbol_resolver import resolve_symbol
from src.advanced_metrics import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe,
    get_best_worst_trade,
    calculate_portfolio_growth,
)
import threading
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


app = Flask(__name__)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def format_period(user_input):
    user_input = user_input.lower().strip().replace(" ", "")
    if user_input == "max":
        return "max"
    num = ""
    for char in user_input:
        if char.isdigit():
            num += char
        else:
            break
    if num == "":
        return "1y"
    num = int(num)
    if num <= 0:
        return "1y"
    if "mo" in user_input:
        return f"{num}mo"
    if "y" in user_input:
        return f"{num}y"
    return f"{num}mo" if num <= 12 else f"{num}y"


def run_strategy(symbol, period, short_ema, long_ema):
    """
    Core strategy runner — shared between main route and /api/compare.
    Returns (result_dict, data_df, short_col, long_col) or raises on error.
    """
    from src.db_loader import load_price_data
    from src.data_fetch import get_stock_data, store_price_data

    # 🔥 Step 1: Try DB first
    data = load_price_data(symbol)

    if data is not None:
        print("⚡ Loaded from DB")

    else:
        print("📡 Fetching from API...")
        data = get_stock_data(symbol, period)

        if data is not None:
            # 🚀 Store in background (NON-BLOCKING)
            threading.Thread(
                target=store_price_data,
                args=(symbol, data),
                daemon=True
            ).start()

    if data is None or data.empty:
        raise ValueError(f"No data found for '{symbol}'. Please check the symbol and try again.")

    short_col = f"EMA{short_ema}"
    long_col  = f"EMA{long_ema}"

    data[short_col] = calculate_ema(data["Close"].tolist(), short_ema)
    data[long_col]  = calculate_ema(data["Close"].tolist(), long_ema)

    signals = detect_crossovers(data, short_col, long_col)
    trades  = backtest(data, signals)
    metrics = calculate_metrics(trades)

    # Detailed trades
    detailed_trades = []
    i = 0
    while i < len(signals) - 1:
        s1, d1 = signals[i]
        s2, d2 = signals[i + 1]
        if s1 == "BUY" and s2 == "SELL":
            buy_price  = data.loc[data.index == d1, "Close"].values[0]
            sell_price = data.loc[data.index == d2, "Close"].values[0]
            ret = ((sell_price - buy_price) / buy_price) * 100
            quality = "Big Win" if ret > 5 else ("Small Win" if ret > 0 else "Loss")
            detailed_trades.append({
                "buy_date":   d1,
                "buy_price":  round(buy_price,  2),
                "sell_date":  d2,
                "sell_price": round(sell_price, 2),
                "return":     round(ret, 2),
                "quality":    quality,
            })
            i += 2
        else:
            i += 1

    # Advanced metrics
    prices_list   = data["Close"].tolist()
    start_date    = data.index[0].to_pydatetime()
    end_date      = data.index[-1].to_pydatetime()
    portfolio = calculate_portfolio_growth(detailed_trades, 100000)
    total_return = portfolio["total_gain_pct"]

    metrics["Total Return (%)"] = total_return

    cagr         = calculate_cagr(total_return, start_date, end_date)
    max_drawdown = calculate_max_drawdown(prices_list)
    sharpe       = calculate_sharpe(detailed_trades)
    best_trade, worst_trade = get_best_worst_trade(detailed_trades)

    # Analysis
    analysis = summarize_results(signal_outcome_analysis(data, signals))
    table    = build_signal_table(data, signals, short_col, long_col)

    # Health score
    avg_confidence = 0
    if table:
        avg_confidence = sum(r["Confidence"] for r in table) / len(table)
    health_score = round(
        max(0, min(100, total_return * 0.4 + metrics["Win Rate (%)"] * 0.4 + avg_confidence * 0.2)),
        2
    )

    # Bias (based on actual performance — not signal count)
    if total_return > 2:
        bias = "Bullish Bias 📈"
        bias_description = "Bullish Bias 📈 => More buy signals than sell signals (strategy favors upward trends.)"
    elif total_return < -2:
        bias = "Bearish Bias 📉"
        bias_description = "Bearish Bias 📉 => More sell signals than buy signals (strategy favors downward trends.)"
    else:
        bias = "Neutral ⚖️"
        bias_description = "Neutral ⚖️ => Buy and sell signals are balanced (no clear trend advantage)."
    result = {
        "metrics":        metrics,
        "analysis":       analysis,
        "table":          table,
        "detailed_trades": detailed_trades,
        "symbol":         symbol,
        "period":         period,
        "short_ema":      short_ema,
        "long_ema":       long_ema,
        "health_score":   health_score,
        "bias":           bias,
        "bias_description": bias_description,
        "signals":        signals,
        # Advanced
        "cagr":           cagr,
        "max_drawdown":   max_drawdown,
        "sharpe":         sharpe,
        "best_trade":     best_trade,
        "worst_trade":    worst_trade,
        "portfolio": portfolio,
    }

    return result, data, short_col, long_col


# ─────────────────────────────────────────────
# MAIN ROUTE
# ─────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error  = None

    prices = ema_short_data = ema_long_data = dates = None
    buy_points  = []
    sell_points = []
    portfolio   = None

    if request.method == "POST":
        try:
            symbol_input = request.form.get("symbol", "").strip()
            period_input = request.form.get("period", "1y")
            short_ema    = int(request.form.get("short_ema", 20))
            long_ema     = int(request.form.get("long_ema",  50))

            if not symbol_input:
                raise ValueError("Please enter a stock symbol (e.g. RELIANCE.NS).")

            if short_ema >= long_ema:
                raise ValueError("Short EMA must be less than Long EMA.")

            symbol = resolve_symbol(symbol_input)
            period = format_period(period_input)

            result, data, short_col, long_col = run_strategy(symbol, period, short_ema, long_ema)

            # Chart data
            prices         = data["Close"].tolist()
            ema_short_data = data[short_col].tolist()
            ema_long_data  = data[long_col].tolist()
            dates          = data.index.strftime("%Y-%m-%d").tolist()

            for signal, date in result["signals"]:
                price          = data.loc[data.index == date, "Close"].values[0]
                formatted_date = date.strftime("%Y-%m-%d")
                if signal == "BUY":
                    buy_points.append({"x": formatted_date, "y": float(price)})
                else:
                    sell_points.append({"x": formatted_date, "y": float(price)})

            # Portfolio growth (default ₹1,00,000)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        result=result,
        error=error,
        prices=prices,
        ema_short=ema_short_data,
        ema_long=ema_long_data,
        dates=dates,
        buy_points=buy_points,
        sell_points=sell_points,
        portfolio = result["portfolio"] if result else None,
    )


# ─────────────────────────────────────────────
# STRATEGY COMPARISON API
# ─────────────────────────────────────────────

EMA_INSIGHTS = {
    "9/21":   "Ultra short-term — very fast signals, high noise, suited for swing traders.",
    "12/26":  "Classic MACD basis — balanced sensitivity, widely used in momentum strategies.",
    "20/50":  "Medium-term trend — reduces noise, popular for positional trading.",
    "50/200": "Golden/Death Cross — long-term trend filter, fewer but high-conviction signals.",
}

@app.route("/api/compare", methods=["POST"])
def api_compare():
    try:
        data_json  = request.get_json()
        symbol     = data_json.get("symbol", "").strip()
        period     = data_json.get("period", "1y")
        short_ema  = int(data_json.get("short_ema", 20))
        long_ema   = int(data_json.get("long_ema",  50))

        if not symbol:
            return jsonify({"error": "Symbol missing"}), 400
        if short_ema >= long_ema:
            return jsonify({"error": "Short EMA must be less than Long EMA"}), 400

        resolved = resolve_symbol(symbol)
        res, _, _, _ = run_strategy(resolved, period, short_ema, long_ema)

        key     = f"{short_ema}/{long_ema}"
        insight = EMA_INSIGHTS.get(key, f"Custom EMA {short_ema}/{long_ema} strategy.")

        return jsonify({
            "label":        key,
            "total_return": res["metrics"]["Total Return (%)"],
            "win_rate":     res["metrics"]["Win Rate (%)"],
            "trades":       res["metrics"]["Number of Trades"],
            "cagr":         res["cagr"],
            "sharpe":       res["sharpe"],
            "max_dd":       res["max_drawdown"],
            "insight":      insight,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)