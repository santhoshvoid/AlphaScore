"""
app.py  —  AlphaCross Flask backend
New in this version:
  - load_price_data now receives the period (Feature 1)
  - save_stock() called after every analysis (Feature 2 — stocks table)
  - /api/watchlist  GET  endpoint (Feature 4)
  - /api/export/trades and /api/export/signals  GET  endpoints (Feature 6)
"""

from flask import Flask, render_template, request, jsonify, Response
import threading
import csv
import io

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
from src.db_loader import (
    load_price_data,
    save_stock,
    get_watchlist,
    save_signals,
    save_trades,
    save_metrics,
)

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
    if not num:
        return "1y"
    num = int(num)
    if num <= 0:
        return "1y"
    if "mo" in user_input:
        return f"{num}mo"
    if "y" in user_input:
        return f"{num}y"
    return f"{num}mo" if num <= 12 else f"{num}y"


# ─────────────────────────────────────────────
# CORE STRATEGY RUNNER
# ─────────────────────────────────────────────

def run_strategy(symbol, period, short_ema, long_ema):
    """
    Full pipeline: data → EMAs → signals → backtest → advanced metrics → DB save.
    Returns (result_dict, data_df, short_col, long_col).
    """

    # ── 1. Price data (DB cache → yfinance fallback) ──────────
    # Pass period so DB only returns rows that match the requested date range
    data = load_price_data(symbol, period)

    if data is not None:
        print(f"⚡ [{symbol}] Loaded from DB (period={period})")
    else:
        print(f"📡 [{symbol}] Fetching from yfinance (period={period})…")
        data = get_stock_data(symbol, period)
        if data is not None:
            threading.Thread(
                target=store_price_data,
                args=(symbol, data),
                daemon=True
            ).start()

    if data is None or data.empty:
        raise ValueError(
            f"No data found for '{symbol}'. "
            "Please enter a valid NSE symbol e.g. RELIANCE.NS or TCS.NS."
        )

    # ── 2. Indicators + signals + basic backtest ──────────────
    short_col = f"EMA{short_ema}"
    long_col  = f"EMA{long_ema}"

    data[short_col] = calculate_ema(data["Close"], short_ema)
    data[long_col]  = calculate_ema(data["Close"], long_ema)

    signals = detect_crossovers(data, short_col, long_col)
    trades  = backtest(data, signals)
    metrics = calculate_metrics(trades)

    # ── 3. Detailed trades ─────────────────────────────────────
    detailed_trades = []
    i = 0
    while i < len(signals) - 1:
        s1, d1 = signals[i]
        s2, d2 = signals[i + 1]
        if s1 == "BUY" and s2 == "SELL":
            buy_price  = float(data.at[d1, "Close"])
            sell_price = float(data.at[d2, "Close"])
            ret        = ((sell_price - buy_price) / buy_price) * 100
            quality    = (
                "Big Win"   if ret > 5
                else "Small Win" if ret > 0
                else "Loss"
            )
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

    # ── 4. Advanced metrics ────────────────────────────────────
    prices_list = data["Close"].tolist()
    start_dt    = data.index[0].to_pydatetime()
    end_dt      = data.index[-1].to_pydatetime()

    cagr                    = calculate_cagr(metrics["Total Return (%)"], start_dt, end_dt)
    max_drawdown            = calculate_max_drawdown(prices_list)
    sharpe                  = calculate_sharpe(detailed_trades)
    best_trade, worst_trade = get_best_worst_trade(detailed_trades)
    portfolio               = calculate_portfolio_growth(detailed_trades, 100_000)

    # ── 5. Signal analysis + table ─────────────────────────────
    analysis = summarize_results(signal_outcome_analysis(data, signals))
    table    = build_signal_table(data, signals, short_col, long_col)

    avg_conf = sum(r["Confidence"] for r in table) / len(table) if table else 0
    health_score = round(
        max(0, min(100,
            metrics["Total Return (%)"] * 0.4
            + metrics["Win Rate (%)"]   * 0.4
            + avg_conf                  * 0.2
        )), 2
    )

    # ── 6. Market bias ──────────────────────────────────────────
    tr = metrics["Total Return (%)"]
    if tr > 2:
        bias, bias_description = (
            "Bullish Bias 📈",
            "More buy signals than sell signals — strategy favours upward trends."
        )
    elif tr < -2:
        bias, bias_description = (
            "Bearish Bias 📉",
            "More sell signals than buy signals — strategy favours downward trends."
        )
    else:
        bias, bias_description = (
            "Neutral ⚖️",
            "Buy and sell signals are balanced — no clear trend advantage."
        )

    # ── 7. Background DB save (non-blocking) ───────────────────
    _sig_copy  = list(signals)
    _trd_copy  = list(detailed_trades)
    _met_db    = {**metrics, "Sharpe": sharpe, "Max Drawdown": max_drawdown}

    def _save():
        try:
            save_stock(symbol)                                  # stocks table
            save_signals(symbol, _sig_copy, short_ema, long_ema)
            save_trades(symbol, _trd_copy)
            save_metrics(symbol, _met_db, short_ema, long_ema)
            print(f"✅ [{symbol}] Saved to DB")
        except Exception as exc:
            print(f"⚠️ DB save failed (non-critical): {exc}")

    threading.Thread(target=_save, daemon=True).start()

    # ── 8. Result dict ─────────────────────────────────────────
    return {
        "metrics":          metrics,
        "analysis":         analysis,
        "table":            table,
        "detailed_trades":  detailed_trades,
        "symbol":           symbol,
        "period":           period,
        "short_ema":        short_ema,
        "long_ema":         long_ema,
        "health_score":     health_score,
        "bias":             bias,
        "bias_description": bias_description,
        "signals":          signals,
        "cagr":             cagr,
        "max_drawdown":     max_drawdown,
        "sharpe":           sharpe,
        "best_trade":       best_trade,
        "worst_trade":      worst_trade,
        "portfolio":        portfolio,
    }, data, short_col, long_col


# ─────────────────────────────────────────────
# MAIN ROUTE
# ─────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    result      = None
    error       = None
    prices      = None
    ema_short_d = None
    ema_long_d  = None
    dates       = None
    buy_points  = []
    sell_points = []

    # Always load watchlist for the sidebar (both GET and POST)
    watchlist = get_watchlist(8)

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

            result, data, short_col, long_col = run_strategy(
                symbol, period, short_ema, long_ema
            )

            prices      = data["Close"].tolist()
            ema_short_d = data[short_col].tolist()
            ema_long_d  = data[long_col].tolist()
            dates       = data.index.strftime("%Y-%m-%d").tolist()

            for signal, date in result["signals"]:
                price = float(data.loc[data.index == date, "Close"].values[0])
                fd    = date.strftime("%Y-%m-%d")
                (buy_points if signal == "BUY" else sell_points).append(
                    {"x": fd, "y": price}
                )

            # Refresh watchlist after saving this symbol
            watchlist = get_watchlist(8)

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        result      = result,
        error       = error,
        prices      = prices,
        ema_short   = ema_short_d,
        ema_long    = ema_long_d,
        dates       = dates,
        buy_points  = buy_points,
        sell_points = sell_points,
        portfolio   = result["portfolio"] if result else None,
        watchlist   = watchlist,
    )


# ─────────────────────────────────────────────
# WATCHLIST API  (Feature 4)
# ─────────────────────────────────────────────

@app.route("/api/watchlist", methods=["GET"])
def api_watchlist():
    """Returns recently analysed symbols as JSON."""
    return jsonify(get_watchlist(8))


# ─────────────────────────────────────────────
# CSV EXPORT ENDPOINTS  (Feature 6)
# ─────────────────────────────────────────────

@app.route("/api/export/trades", methods=["POST"])
def export_trades():
    """
    Re-runs the strategy for the submitted symbol/period/EMA and
    returns the detailed trades as a downloadable CSV file.
    """
    try:
        symbol_input = request.form.get("symbol", "").strip()
        period_input = request.form.get("period", "1y")
        short_ema    = int(request.form.get("short_ema", 20))
        long_ema     = int(request.form.get("long_ema",  50))

        symbol = resolve_symbol(symbol_input)
        period = format_period(period_input)
        result, _, _, _ = run_strategy(symbol, period, short_ema, long_ema)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Buy Date", "Buy Price", "Sell Date",
                         "Sell Price", "Return (%)", "Quality"])
        for t in result["detailed_trades"]:
            writer.writerow([
                str(t["buy_date"]).replace(" 00:00:00", ""),
                t["buy_price"],
                str(t["sell_date"]).replace(" 00:00:00", ""),
                t["sell_price"],
                t["return"],
                t["quality"],
            ])

        filename = f"trades_{symbol}_{period}.csv"
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/export/signals", methods=["POST"])
def export_signals():
    """Returns the Signal Intelligence table as a CSV file."""
    try:
        symbol_input = request.form.get("symbol", "").strip()
        period_input = request.form.get("period", "1y")
        short_ema    = int(request.form.get("short_ema", 20))
        long_ema     = int(request.form.get("long_ema",  50))

        symbol = resolve_symbol(symbol_input)
        period = format_period(period_input)
        result, _, _, _ = run_strategy(symbol, period, short_ema, long_ema)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Date", "Type", "Price", "EMA S", "EMA L",
                         "Strength", "Confidence (%)", "Label",
                         "+1D (%)", "+3D (%)", "+1W (%)", "+1M (%)"])
        for row in result["table"]:
            writer.writerow([
                row["Date"], row["type"], row["Price"],
                row["EMA_Short"], row["EMA_Long"],
                row["Strength"], row["Confidence"], row["Label"],
                row.get("+1D", ""), row.get("+3D", ""),
                row.get("+1W", ""), row.get("+1M", ""),
            ])

        filename = f"signals_{symbol}_{period}.csv"
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ─────────────────────────────────────────────
# STRATEGY COMPARISON API
# ─────────────────────────────────────────────

EMA_INSIGHTS = {
    "9/21":   "Ultra short-term — very fast signals, high noise. Best for active swing traders who can monitor daily.",
    "12/26":  "Classic MACD basis — balanced sensitivity. Widely used in momentum strategies worldwide.",
    "20/50":  "Medium-term positional — filters day-to-day noise. Good balance of timeliness and reliability.",
    "50/200": "The legendary Golden/Death Cross — fewest signals but highest conviction. Suits long-term investors.",
}


@app.route("/api/compare", methods=["POST"])
def api_compare():
    try:
        body      = request.get_json()
        symbol    = body.get("symbol", "").strip()
        period    = body.get("period",   "1y")
        short_ema = int(body.get("short_ema", 20))
        long_ema  = int(body.get("long_ema",  50))

        if not symbol:
            return jsonify({"error": "Symbol is required."}), 400
        if short_ema >= long_ema:
            return jsonify({"error": "Short EMA must be less than Long EMA."}), 400

        res, _, _, _ = run_strategy(resolve_symbol(symbol), period, short_ema, long_ema)
        key = f"{short_ema}/{long_ema}"

        return jsonify({
            "label":        key,
            "total_return": res["metrics"]["Total Return (%)"],
            "win_rate":     res["metrics"]["Win Rate (%)"],
            "trades":       res["metrics"]["Number of Trades"],
            "cagr":         res["cagr"],
            "sharpe":       res["sharpe"],
            "max_dd":       res["max_drawdown"],
            "insight":      EMA_INSIGHTS.get(key, f"Custom EMA {key} — results vary by stock and market regime."),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)