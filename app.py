"""
app.py  —  AlphaCross Flask backend  (v3 — full ML + explainability)

What changed in this version
──────────────────────────────
1. STARTUP SPEED FIX
   - FinBERT and XGBoost are now lazy-loaded (loaded on first use, not
     at import time). Startup from 8-12 seconds → under 1 second.

2. PER-USER WATCHLIST (browser cookie-based)
   - get_watchlist() now returns the list stored in the user's browser
     cookie, not a shared DB list. Each user sees only their own history.
   - The server still saves to the stocks table for analytics/ML purposes,
     but the sidebar chips are driven by a cookie named "user_history".

3. EXPLAINABILITY LAYER
   - _compute_strategy() now returns ml_score, ema_score, sentiment_score
     as numeric values alongside their labels and confidences.
   - These are used by the template to render the Decision Breakdown panel.

4. SENTIMENT VALIDATION
   - article_count and data_quality are passed through to the template
     so the UI can warn when sentiment data is insufficient.

5. All existing features preserved:
   portfolio growth · CSV export · strategy comparison · tooltips ·
   mobile layout · DB caching · watchlist chips
"""

from flask import Flask, render_template, request, jsonify, Response, make_response
import threading
import csv
import io
import json

# ── Lazy imports for heavy models (avoid startup delay) ──────
# The actual model loading happens inside _compute_strategy on first call.
from src.ml_model import predict_latest
from src.sentiment import get_sentiment

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
    save_signals,
    save_trades,
    save_metrics,
    load_metrics,
    signals_exist,
    trades_exist,
)

app = Flask(__name__)

_SAVING_NOW: set = set()
_LOADING:    set = set()

COOKIE_NAME     = "user_history"
COOKIE_MAX_AGE  = 60 * 60 * 24 * 30   # 30 days
COOKIE_MAX_ITEMS = 8


# ─────────────────────────────────────────────
# COOKIE-BASED WATCHLIST  (per-user, private)
# ─────────────────────────────────────────────

def _get_cookie_history(request_obj) -> list:
    """Read user's personal symbol history from browser cookie."""
    raw = request_obj.cookies.get(COOKIE_NAME, "[]")
    try:
        items = json.loads(raw)
        if isinstance(items, list):
            return [str(s) for s in items[:COOKIE_MAX_ITEMS]]
    except Exception:
        pass
    return []


def _push_cookie_history(response_obj, symbol: str, current: list) -> list:
    """Add symbol to front of list, deduplicate, trim, set cookie."""
    updated = [symbol] + [s for s in current if s != symbol]
    updated = updated[:COOKIE_MAX_ITEMS]
    response_obj.set_cookie(
        COOKIE_NAME,
        json.dumps(updated),
        max_age=COOKIE_MAX_AGE,
        httponly=False,    # JS needs to read this if required
        samesite="Lax",
    )
    return updated


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


def _label_score(score: float, component: str) -> str:
    """Convert a numeric score [-1, +1] to a human label for explainability."""
    if component == "EMA":
        strong = 0.03
        weak   = 0.008
    else:
        strong = 0.50
        weak   = 0.15

    if score > strong:
        return "Strong Bullish"
    elif score > weak:
        return "Weak Bullish"
    elif score < -strong:
        return "Strong Bearish"
    elif score < -weak:
        return "Weak Bearish"
    else:
        return "Neutral"


# ─────────────────────────────────────────────
# PURE COMPUTE  (no DB reads or writes)
# ─────────────────────────────────────────────

def _compute_strategy(data, short_ema: int, long_ema: int, symbol: str = "") -> tuple:
    """
    Pure computation layer — EMAs → signals → backtest → metrics → ML → sentiment.
    Returns (result_dict, short_col, long_col).
    No database interaction whatsoever.
    """
    short_col = f"EMA{short_ema}"
    long_col  = f"EMA{long_ema}"

    data[short_col] = calculate_ema(data["Close"], short_ema)
    data[long_col]  = calculate_ema(data["Close"], long_ema)

    signals = detect_crossovers(data, short_col, long_col)

    INITIAL_CAPITAL = 100_000
    trades, equity_curve, final_capital = backtest(data, signals, INITIAL_CAPITAL)
    metrics = calculate_metrics(trades, INITIAL_CAPITAL, final_capital)

    # ── Detailed trades ─────────────────────────────────────────
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

    # ── Advanced metrics ─────────────────────────────────────────
    prices_list             = data["Close"].tolist()
    start_dt                = data.index[0].to_pydatetime()
    end_dt                  = data.index[-1].to_pydatetime()

    portfolio = {
        "history":        equity_curve,
        "final":          final_capital,
        "total_gain":     final_capital - INITIAL_CAPITAL,
        "total_gain_pct": ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100,
    }

    cagr                    = calculate_cagr(portfolio["total_gain_pct"], start_dt, end_dt)
    max_drawdown            = calculate_max_drawdown(prices_list)
    sharpe                  = calculate_sharpe(detailed_trades)
    best_trade, worst_trade = get_best_worst_trade(detailed_trades)

    # ── Signal analysis + intelligence table ─────────────────────
    analysis     = summarize_results(signal_outcome_analysis(data, signals))
    table        = build_signal_table(data, signals, short_col, long_col)
    avg_conf     = sum(r["Confidence"] for r in table) / len(table) if table else 0

    health_score = round(max(0, min(100,
        metrics["Total Return (%)"] * 0.4
        + metrics["Win Rate (%)"]   * 0.4
        + avg_conf                  * 0.2
    )), 2)

    # ── Market bias ──────────────────────────────────────────────
    tr = metrics["Total Return (%)"]
    if tr > 2:
        bias      = "Bullish Bias 📈"
        bias_desc = "More buy signals than sell signals — strategy favours upward trends."
    elif tr < -2:
        bias      = "Bearish Bias 📉"
        bias_desc = "More sell signals than buy signals — strategy favours downward trends."
    else:
        bias      = "Neutral ⚖️"
        bias_desc = "Buy and sell signals are balanced — no clear trend advantage."

    # ── ML PREDICTION ────────────────────────────────────────────
    ml_result = predict_latest(data) or {
        "prediction":     "Unavailable",
        "confidence":     0,
        "raw_score":      0.0,
        "regime":         "UNKNOWN",
        "volatility_pct": 0.0,
    }

    ml_conf  = ml_result.get("confidence", 0) / 100
    raw_ml   = ml_result.get("raw_score", 0.0)

    if "Bullish" in ml_result["prediction"]:
        ml_score = ml_conf
    elif "Bearish" in ml_result["prediction"]:
        ml_score = -ml_conf
    else:
        ml_score = 0.0

    # ── SENTIMENT ────────────────────────────────────────────────
    try:
        sentiment_result = get_sentiment(symbol) if symbol else {
            "prediction": "Neutral ⚖️", "confidence": 0,
            "article_count": 0, "data_quality": "no_symbol",
        }
    except Exception:
        sentiment_result = {
            "prediction": "Neutral ⚖️", "confidence": 0,
            "article_count": 0, "data_quality": "error",
        }

    sent_conf  = sentiment_result.get("confidence", 0) / 100
    raw_sent   = sentiment_result.get("raw_score", 0.0)
    data_qual  = sentiment_result.get("data_quality", "unknown")
    art_count  = sentiment_result.get("article_count", 0)

    if "Bullish" in sentiment_result["prediction"]:
        sentiment_score = sent_conf
    elif "Bearish" in sentiment_result["prediction"]:
        sentiment_score = -sent_conf
    else:
        sentiment_score = 0.0

    # ── EMA TREND SCORE ──────────────────────────────────────────
    latest      = data.iloc[-1]
    price       = float(latest["Close"])
    ema_diff    = float(latest[short_col]) - float(latest[long_col])
    ema_score   = max(min((ema_diff / price) * 3, 1.0), -1.0)

    # ── MARKET REGIME ────────────────────────────────────────────
    trend_strength = abs(ema_diff) / price
    regime = ml_result.get("regime", "TRENDING" if trend_strength > 0.02 else "SIDEWAYS")

    # ── ADAPTIVE WEIGHTING ───────────────────────────────────────
    if regime == "TRENDING":
        ml_w, ema_w, sent_w = 0.40, 0.40, 0.20
    else:
        # Sideways: ML is more reliable than raw EMA trend
        ml_w, ema_w, sent_w = 0.60, 0.20, 0.20

    # Dynamic scaling: stronger signals get more weight
    ml_w   *= (0.5 + abs(ml_score))
    ema_w  *= (0.5 + abs(ema_score))

    # Scale down sentiment if data quality is low
    if data_qual in ("insufficient", "low", "error", "no_api_key"):
        sent_w *= 0.3
    elif data_qual == "moderate":
        sent_w *= 0.65
    # good quality → full weight

    sent_w *= (0.5 + abs(sentiment_score))

    # Confidence influence
    ml_w   *= ml_conf
    sent_w *= sent_conf

    total_w = ml_w + ema_w + sent_w
    if total_w == 0:
        ml_w, ema_w, sent_w = 0.33, 0.33, 0.34
        total_w = 1.0

    ml_w   /= total_w
    ema_w  /= total_w
    sent_w /= total_w

    # ── FINAL DECISION ────────────────────────────────────────────
    final_score = ml_score * ml_w + ema_score * ema_w + sentiment_score * sent_w

    # Disagreement penalty
    scores      = [ml_score, ema_score, sentiment_score]
    disagreement = 0.0
    if ml_score * ema_score < 0:
        disagreement += 0.4
    if ml_score * sentiment_score < 0:
        disagreement += 0.3
    if ema_score * sentiment_score < 0:
        disagreement += 0.2
    disagreement  = min(disagreement, 0.8)
    final_score  *= (1 - disagreement)

    if   final_score >  0.05:
        final_prediction = "Bullish 📈"
    elif final_score < -0.05:
        final_prediction = "Bearish 📉"
    else:
        final_prediction = "Neutral ⚖️"

    # ── CONFIDENCE ───────────────────────────────────────────────
    positive     = sum(1 for s in scores if s > 0)
    negative     = sum(1 for s in scores if s < 0)
    agree_score  = (positive - negative) / 3
    confidence   = abs(final_score)
    confidence  *= (1 + agree_score * 0.5)
    if regime == "TRENDING":
        confidence *= 1.1
    confidence   = confidence ** 0.7
    final_conf   = round(max(min(confidence, 1.0), 0.0) * 100, 2)

    # ── EXPLAINABILITY SCORES ─────────────────────────────────────
    explainability = {
        "ml_score":        round(ml_score, 4),
        "ml_label":        _label_score(ml_score, "ML"),
        "ml_conf":         ml_result.get("confidence", 0),
        "ema_score":       round(ema_score, 4),
        "ema_label":       _label_score(ema_score, "EMA"),
        "sent_score":      round(sentiment_score, 4),
        "sent_label":      _label_score(sentiment_score, "SENT"),
        "regime":          regime,
        "disagreement":    round(disagreement, 2),
        "article_count":   art_count,
        "data_quality":    data_qual,
        # Effective weights (after normalisation) as percentages
        "ml_weight_pct":   round(ml_w   * 100, 1),
        "ema_weight_pct":  round(ema_w  * 100, 1),
        "sent_weight_pct": round(sent_w * 100, 1),
    }

    result = {
        "metrics":              metrics,
        "analysis":             analysis,
        "table":                table,
        "detailed_trades":      detailed_trades,
        "short_ema":            short_ema,
        "long_ema":             long_ema,
        "health_score":         health_score,
        "bias":                 bias,
        "bias_description":     bias_desc,
        "signals":              signals,
        "cagr":                 cagr,
        "max_drawdown":         max_drawdown,
        "sharpe":               sharpe,
        "best_trade":           best_trade,
        "worst_trade":          worst_trade,
        "portfolio":            portfolio,
        "ml_prediction":        ml_result,
        "sentiment_prediction": sentiment_result,
        "final_prediction": {
            "prediction": final_prediction,
            "confidence": final_conf,
        },
        "explainability":       explainability,
    }

    return result, short_col, long_col


# ─────────────────────────────────────────────
# FULL STRATEGY RUNNER  (main route)
# ─────────────────────────────────────────────

def run_strategy(symbol: str, period: str, short_ema: int, long_ema: int) -> tuple:
    """
    Full pipeline: price data (DB → yfinance) → compute → async DB save.
    Returns (result_dict, data_df, short_col, long_col).
    Price data is fetched then saved in background (non-blocking UI).
    """

    # ── 1. Price data ──────────────────────────────────────────
    data = load_price_data(symbol, period)

    if data is not None and not data.empty:
        print(f"⚡ [{symbol}] Loaded from DB (period={period})")
    else:
        print(f"📡 [{symbol}] Fetching from yfinance (period={period})…")

        if symbol in _LOADING:
            data = get_stock_data(symbol, period)
        else:
            _LOADING.add(symbol)
            data = get_stock_data(symbol, period)

            if data is not None:
                def _save_bg():
                    try:
                        store_price_data(symbol, data)
                        print(f"✅ [{symbol}] Price data saved (bg).")
                    except Exception as e:
                        print(f"❌ [{symbol}] Price save failed: {e}")
                    finally:
                        _LOADING.discard(symbol)
                threading.Thread(target=_save_bg, daemon=True).start()

    if data is None or data.empty:
        raise ValueError(
            f"No data found for '{symbol}'. "
            "Please enter a valid NSE symbol e.g. RELIANCE.NS or TCS.NS."
        )

    # ── 2. Pure compute ─────────────────────────────────────────
    result, short_col, long_col = _compute_strategy(data, short_ema, long_ema, symbol)
    result["symbol"] = symbol
    result["period"] = period

    # ── 3. Async DB save (signals/trades/metrics) ───────────────
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

                if not trades_exist(symbol):
                    save_trades(symbol, _trd_copy)

                if not load_metrics(symbol, short_ema, long_ema):
                    save_metrics(symbol, _met_db, short_ema, long_ema)

                print(f"✅ [{symbol}] DB save complete.")
            except Exception as exc:
                print(f"⚠️ DB save error: {exc}")
            finally:
                _SAVING_NOW.discard(save_key)

        threading.Thread(target=_bg_save, daemon=True).start()
    else:
        print(f"⏭️  [{symbol}] Save already in progress, skipping.")

    return result, data, short_col, long_col


# ─────────────────────────────────────────────
# COMPARE  (read-only, never writes to DB)
# ─────────────────────────────────────────────

def compare_strategy(symbol: str, period: str, short_ema: int, long_ema: int) -> dict:
    """
    Lightweight read-only compute for the compare API.
    Loads from DB if available; falls back to yfinance WITHOUT saving.
    Never writes to the database.
    """
    data = load_price_data(symbol, period)

    if data is not None:
        print(f"⚡ [CMP {symbol}] DB hit")
    else:
        print(f"📡 [CMP {symbol}] DB miss — fetching from yfinance (no save)")
        data = get_stock_data(symbol, period)

    if data is None or data.empty:
        raise ValueError(f"No data found for '{symbol}'.")

    result, _, _ = _compute_strategy(data, short_ema, long_ema, symbol)
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

    # Read per-user history from cookie
    cookie_history = _get_cookie_history(request)
    watchlist = [{"symbol": s} for s in cookie_history]

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

        except Exception as exc:
            error = str(exc)

    # Build response object so we can set cookie
    resp = make_response(render_template(
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
    ))

    # Update cookie if we successfully analyzed a stock
    if result:
        _push_cookie_history(resp, result["symbol"], cookie_history)

    return resp


# ─────────────────────────────────────────────
# WATCHLIST API  (reads cookie → JSON)
# ─────────────────────────────────────────────

@app.route("/api/watchlist", methods=["GET"])
def api_watchlist():
    """Returns this user's personal symbol history from their cookie."""
    history = _get_cookie_history(request)
    return jsonify([{"symbol": s} for s in history])


# ─────────────────────────────────────────────
# CSV EXPORT ENDPOINTS
# ─────────────────────────────────────────────

@app.route("/api/export/trades", methods=["POST"])
def export_trades():
    """Download trade history as CSV."""
    try:
        symbol = resolve_symbol(request.form.get("symbol", "").strip())
        period = format_period(request.form.get("period", "1y"))
        s_ema  = int(request.form.get("short_ema", 20))
        l_ema  = int(request.form.get("long_ema",  50))

        res = compare_strategy(symbol, period, s_ema, l_ema)

        buf = io.StringIO()
        w   = csv.writer(buf)
        w.writerow(["Buy Date", "Buy Price", "Sell Date", "Sell Price", "Return (%)", "Quality"])
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
            headers={"Content-Disposition": f"attachment; filename=trades_{symbol}_{period}.csv"}
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/export/signals", methods=["POST"])
def export_signals():
    """Download signal intelligence table as CSV."""
    try:
        symbol = resolve_symbol(request.form.get("symbol", "").strip())
        period = format_period(request.form.get("period", "1y"))
        s_ema  = int(request.form.get("short_ema", 20))
        l_ema  = int(request.form.get("long_ema",  50))

        res = compare_strategy(symbol, period, s_ema, l_ema)

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
            headers={"Content-Disposition": f"attachment; filename=signals_{symbol}_{period}.csv"}
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
            "label":         key,
            "total_return":  res["metrics"]["Total Return (%)"],
            "win_rate":      res["metrics"]["Win Rate (%)"],
            "trades":        res["metrics"]["Number of Trades"],
            "cagr":          res["cagr"],
            "sharpe":        res["sharpe"],
            "max_dd":        res["max_drawdown"],
            "final_capital": res["portfolio"]["final"],
            "insight":       EMA_INSIGHTS.get(
                                 key,
                                 f"Custom EMA {key} — results vary by stock and market regime."
                             ),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)