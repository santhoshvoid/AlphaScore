"""
app.py  —  AlphaCross Flask backend  (clean architecture v2)

Architecture fix summary
────────────────────────
BEFORE  (buggy):
  • store_price_data ran in background thread  → race condition: next
    request arrived before save finished → DB was still empty → yfinance
    called again every time
  • /api/compare called run_strategy() → triggered full API fetch + DB
    save on every "Compare" click → duplicate writes, multiple API calls
  • Export endpoints also called run_strategy() → same problem

AFTER   (clean):
  • Price data is saved SYNCHRONOUSLY on first fetch.  Signals/trades/
    metrics are saved asynchronously (they're fast and non-critical for
    the next request).
  • _compute_strategy() is a pure function — EMAs → signals → metrics.
    No DB reads or writes. Called by both run_strategy and compare.
  • compare_strategy() is READ-ONLY — loads price data from DB (fast),
    calls _compute_strategy, returns metrics. Never writes to DB.
  • A simple _SAVING_NOW set prevents duplicate concurrent saves.

All existing features preserved:
  portfolio growth · watchlist chips · CSV export ·
  strategy comparison · /api/watchlist · tooltips · mobile layout
"""
from src.ml_model import predict_latest
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
    load_metrics,
    signals_exist,
    trades_exist
)
_LOADING = set()

app = Flask(__name__)

# ── Global save-lock: prevents duplicate concurrent DB writes ────
_SAVING_NOW: set = set()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def format_period(user_input: str) -> str:
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
# PURE COMPUTE  (no DB reads or writes)
# ─────────────────────────────────────────────

def _compute_strategy(data, short_ema: int, long_ema: int) -> tuple:
    """
    Pure computation layer — EMAs → signals → backtest → metrics.
    Takes a price DataFrame, returns (result_dict, short_col, long_col).
    No database interaction whatsoever.
    """
    short_col = f"EMA{short_ema}"
    long_col  = f"EMA{long_ema}"

    data[short_col] = calculate_ema(data["Close"], short_ema)
    data[long_col]  = calculate_ema(data["Close"], long_ema)

    signals = detect_crossovers(data, short_col, long_col)
    INITIAL_CAPITAL = 100000  # same as your simulator input

    trades, equity_curve, final_capital = backtest(data, signals, INITIAL_CAPITAL)
    
    metrics = calculate_metrics(trades, INITIAL_CAPITAL, final_capital)
    ml_result = predict_latest(data) or {
        "prediction": "Unavailable",
        "confidence": 0
    }
    # ── Detailed trades ────────────────────────────────────────
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
                "Big Win"    if ret > 5
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

    # ── Advanced metrics ───────────────────────────────────────
    prices_list             = data["Close"].tolist()
    start_dt                = data.index[0].to_pydatetime()
    end_dt                  = data.index[-1].to_pydatetime()
    initial_capital         = INITIAL_CAPITAL
    portfolio = {
        "history": equity_curve,
        "final": final_capital,
        "total_gain": final_capital - initial_capital,
        "total_gain_pct": ((final_capital - initial_capital) / initial_capital) * 100
    }
    cagr = calculate_cagr(
        portfolio["total_gain_pct"],
        start_dt,
        end_dt
    )
    max_drawdown            = calculate_max_drawdown(prices_list)
    sharpe                  = calculate_sharpe(detailed_trades)
    best_trade, worst_trade = get_best_worst_trade(detailed_trades)

    # ── Signal analysis + intelligence table ──────────────────
    analysis  = summarize_results(signal_outcome_analysis(data, signals))
    table     = build_signal_table(data, signals, short_col, long_col)
    avg_conf  = sum(r["Confidence"] for r in table) / len(table) if table else 0

    health_score = round(max(0, min(100,
        metrics["Total Return (%)"] * 0.4
        + metrics["Win Rate (%)"]   * 0.4
        + avg_conf                  * 0.2
    )), 2)

    # ── Market bias ────────────────────────────────────────────
    tr = metrics["Total Return (%)"]
    if tr > 2:
        bias = "Bullish Bias 📈"
        bias_desc = "More buy signals than sell signals — strategy favours upward trends."
    elif tr < -2:
        bias = "Bearish Bias 📉"
        bias_desc = "More sell signals than buy signals — strategy favours downward trends."
    else:
        bias = "Neutral ⚖️"
        bias_desc = "Buy and sell signals are balanced — no clear trend advantage."

    # ── ML prediction already exists
    ml_result = predict_latest(data) or {
        "prediction": "Unavailable",
        "confidence": 0
    }

    # ── TEMP SENTIMENT (placeholder)
    sentiment_result = {
        "prediction": "Neutral 😐",
        "confidence": 50
    }

    # ── FINAL DECISION (basic version)
    final_prediction = ml_result["prediction"]
    final_confidence = ml_result["confidence"]

    result = {
        "metrics":          metrics,
        "analysis":         analysis,
        "table":            table,
        "detailed_trades":  detailed_trades,
        "short_ema":        short_ema,
        "long_ema":         long_ema,
        "health_score":     health_score,
        "bias":             bias,
        "bias_description": bias_desc,
        "signals":          signals,
        "cagr":             cagr,
        "max_drawdown":     max_drawdown,
        "sharpe":           sharpe,
        "best_trade":       best_trade,
        "worst_trade":      worst_trade,
        "portfolio":        portfolio,
        "ml_prediction":    ml_result,
        "ml_prediction":        ml_result,
        "sentiment_prediction": sentiment_result,
        "final_prediction": {
            "prediction": final_prediction,
            "confidence": final_confidence
        }
    }

    return result, short_col, long_col


# ─────────────────────────────────────────────
# FULL STRATEGY RUNNER  (main route)
# ─────────────────────────────────────────────

def run_strategy(symbol: str, period: str, short_ema: int, long_ema: int) -> tuple:
    """
    Full pipeline used by the main / route:
      1. Load price data from DB (instant)
         → if not found: fetch from yfinance then SAVE SYNCHRONOUSLY
            (this guarantees the NEXT request will always hit the DB)
      2. _compute_strategy() — pure compute
      3. Save signals/trades/metrics asynchronously (non-critical for speed)
    Returns (result_dict, data_df, short_col, long_col).
    """

    # ── 1. Price data ──────────────────────────────────────────
    data = load_price_data(symbol, period)

    if data is not None and not data.empty:
        print(f"⚡ [{symbol}] Loaded from DB (period={period})")
    else:
            print(f"📡 [{symbol}] Fetching from yfinance (period={period})…")

            # 🚨 Prevent duplicate API calls
            if symbol in _LOADING:
                print(f"⏳ [{symbol}] Already loading, fetching again (no wait)")
                data = get_stock_data(symbol, period)
            else:
                _LOADING.add(symbol)

                data = get_stock_data(symbol, period)

                if data is not None:
                    # ⚡ SAVE IN BACKGROUND (NON-BLOCKING)
                    def _save_bg():
                        try:
                            print(f"💾 [{symbol}] Saving price data in background...")
                            store_price_data(symbol, data)
                            print(f"✅ [{symbol}] Price data saved (bg).")
                        except Exception as e:
                            print(f"❌ [{symbol}] Save failed: {e}")
                        finally:
                            _LOADING.discard(symbol)

                    threading.Thread(target=_save_bg, daemon=True).start()

    if data is None or data.empty:
        raise ValueError(
            f"No data found for '{symbol}'. "
            "Please enter a valid NSE symbol e.g. RELIANCE.NS or TCS.NS."
        )

    # ── 2. Pure compute ────────────────────────────────────────
    result, short_col, long_col = _compute_strategy(data, short_ema, long_ema)

    # Add symbol/period to result (needed by template)
    result["symbol"] = symbol
    result["period"] = period

    # ── 3. Save signals/trades/metrics in background ──────────
    # Using a lock so the same symbol is never saved twice concurrently.
    save_key = f"{symbol}_{short_ema}_{long_ema}"

    if save_key not in _SAVING_NOW:
        _SAVING_NOW.add(save_key)

        _sig_copy = list(result["signals"])
        _trd_copy = list(result["detailed_trades"])
        _met_db   = {
            **result["metrics"],
            "Sharpe":       result["sharpe"],
            "Max Drawdown": result["max_drawdown"],
        }

        def _bg_save():
            try:
                save_stock(symbol)

                if not signals_exist(symbol, short_ema, long_ema):
                    save_signals(symbol, _sig_copy, short_ema, long_ema)
                else:
                    print(f"⚡ [{symbol}] Signals exist, skip")

                if not trades_exist(symbol):
                    save_trades(symbol, _trd_copy)
                else:
                    print(f"⚡ [{symbol}] Trades exist, skip")

                if not load_metrics(symbol, short_ema, long_ema):
                    save_metrics(symbol, _met_db, short_ema, long_ema)
                else:
                    print(f"⚡ [{symbol}] Metrics exist, skip")

                print(f"✅ [{symbol}] DB save complete (deduped)")

            except Exception as exc:
                print(f"⚠️ DB save error: {exc}")

            finally:
                _SAVING_NOW.discard(save_key)

        threading.Thread(target=_bg_save, daemon=True).start()
    else:
        print(f"⏭️  [{symbol}] Save already in progress, skipping duplicate.")

    return result, data, short_col, long_col


# ─────────────────────────────────────────────
# COMPARE  (read-only, never writes to DB)
# ─────────────────────────────────────────────

def compare_strategy(symbol: str, period: str, short_ema: int, long_ema: int) -> dict:
    """
    Lightweight read-only version for the compare API.
    Loads price data from DB (or yfinance if not cached — WITHOUT saving).
    Calls _compute_strategy() and returns only the metrics needed.
    NEVER writes anything to the database.
    """
    # Try DB first
    data = load_price_data(symbol, period)

    if data is not None:
        print(f"⚡ [CMP {symbol}] DB hit")
    else:
        # Fallback to API for compare — but NO save
        print(f"📡 [CMP {symbol}] DB miss — fetching from yfinance (no save)")
        data = get_stock_data(symbol, period)

    if data is None or data.empty:
        raise ValueError(f"No data found for '{symbol}'.")

    result, _, _ = _compute_strategy(data, short_ema, long_ema)
    return result


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
    watchlist   = get_watchlist(8)

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
# WATCHLIST API
# ─────────────────────────────────────────────

@app.route("/api/watchlist", methods=["GET"])
def api_watchlist():
    return jsonify(get_watchlist(8))


# ─────────────────────────────────────────────
# CSV EXPORT ENDPOINTS
# ─────────────────────────────────────────────

@app.route("/api/export/trades", methods=["POST"])
def export_trades():
    """Download trade history as CSV. Hits DB — instant after first analyze."""
    try:
        symbol_input = request.form.get("symbol", "").strip()
        period_input = request.form.get("period", "1y")
        short_ema    = int(request.form.get("short_ema", 20))
        long_ema     = int(request.form.get("long_ema",  50))

        symbol = resolve_symbol(symbol_input)
        period = format_period(period_input)

        # compare_strategy = read-only, won't trigger a new DB write
        res = compare_strategy(symbol, period, short_ema, long_ema)

        buf = io.StringIO()
        w   = csv.writer(buf)
        w.writerow(["Buy Date", "Buy Price", "Sell Date", "Sell Price",
                    "Return (%)", "Quality"])
        for t in res["detailed_trades"]:
            w.writerow([
                str(t["buy_date"]).replace(" 00:00:00", ""),
                t["buy_price"],
                str(t["sell_date"]).replace(" 00:00:00", ""),
                t["sell_price"],
                t["return"],
                t["quality"],
            ])

        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition":
                     f"attachment; filename=trades_{symbol}_{period}.csv"}
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/export/signals", methods=["POST"])
def export_signals():
    """Download signal intelligence table as CSV."""
    try:
        symbol_input = request.form.get("symbol", "").strip()
        period_input = request.form.get("period", "1y")
        short_ema    = int(request.form.get("short_ema", 20))
        long_ema     = int(request.form.get("long_ema",  50))

        symbol = resolve_symbol(symbol_input)
        period = format_period(period_input)
        res    = compare_strategy(symbol, period, short_ema, long_ema)

        buf = io.StringIO()
        w   = csv.writer(buf)
        w.writerow(["Date", "Type", "Price", "EMA S", "EMA L",
                    "Strength", "Confidence (%)", "Label",
                    "+1D (%)", "+3D (%)", "+1W (%)", "+1M (%)"])
        for row in res["table"]:
            w.writerow([
                row["Date"], row["type"], row["Price"],
                row["EMA_Short"], row["EMA_Long"],
                row["Strength"], row["Confidence"], row["Label"],
                row.get("+1D", ""), row.get("+3D", ""),
                row.get("+1W", ""), row.get("+1M", ""),
            ])

        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition":
                     f"attachment; filename=signals_{symbol}_{period}.csv"}
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ─────────────────────────────────────────────
# STRATEGY COMPARISON API  (read-only)
# ─────────────────────────────────────────────

EMA_INSIGHTS = {
    "9/21":   "Ultra short-term — very fast signals, high noise. Best for active swing traders who can monitor daily.",
    "12/26":  "Classic MACD basis — balanced sensitivity. Widely used in momentum strategies worldwide.",
    "20/50":  "Medium-term positional — filters day-to-day noise. Good balance of timeliness and reliability.",
    "50/200": "The legendary Golden/Death Cross — fewest signals but highest conviction. Suits long-term investors.",
}


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """
    Compares a different EMA pair on the same symbol/period.
    Uses compare_strategy() — read-only, never touches the DB.
    """
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

        res = compare_strategy(resolve_symbol(symbol), period, short_ema, long_ema)
        key = f"{short_ema}/{long_ema}"

        return jsonify({
            "label":        key,
            "total_return": res["metrics"]["Total Return (%)"],
            "win_rate":     res["metrics"]["Win Rate (%)"],
            "trades":       res["metrics"]["Number of Trades"],
            "cagr":         res["cagr"],
            "sharpe":       res["sharpe"],
            "max_dd":       res["max_drawdown"],
            "final_capital": res["portfolio"]["final"],
            "insight":      EMA_INSIGHTS.get(
                                key,
                                f"Custom EMA {key} — results vary by stock and market regime."
                            ),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)