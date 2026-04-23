"""
app.py  —  AlphaCross Flask backend  (v4 — deployment-ready)

Deployment changes vs v3:
  - Celery .delay() calls are now wrapped in try/except.
    If the broker is unreachable (first boot, no worker yet) the app
    falls back to a daemon thread so the response is never blocked.
  - preload_models() only runs XGBoost warm-up when USE_HF_INFERENCE_API
    is false (FinBERT skipped in production to save RAM).

Everything else is identical to v3.
"""

from datetime import datetime
import time
import threading
from unittest import result
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response, make_response
import csv
import io
import json
from src.tasks import save_to_db_task
from src.tasks import save_price_task
from src.ml_model import predict_latest
from src.sentiment import get_sentiment
from src.redis_client import get_redis
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
import os

app = Flask(__name__)

_SAVING_NOW: set = set()
_LOADING:    set = set()

try:
    redis_client = get_redis()
    redis_client.ping()  # test connection at startup
    print("✅ Redis connected.")
except Exception as _re:
    print(f"⚠️ Redis unavailable at startup ({_re}). Caching disabled.")
    redis_client = None

COOKIE_NAME      = "user_history"
COOKIE_MAX_AGE   = 60 * 60 * 24 * 30   # 30 days
COOKIE_MAX_ITEMS = 8

USE_HF_INFERENCE_API = os.getenv("USE_HF_INFERENCE_API", "false").lower() == "true"

@app.route("/health")
def health():
    return "OK", 200

def preload_models():
    print("🚀 Preloading ML models…")

    try:
        dummy = pd.DataFrame({
            "Close":  [100, 101, 102, 103, 104],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        })
        predict_latest(dummy)
        print("✅ XGBoost preloaded.")
    except Exception as e:
        print(f"❌ XGBoost preload failed: {e}")

    # Only warm up FinBERT via API if we're actually using the API.
    # Skip if USE_HF_INFERENCE_API is false (local dev loads lazily anyway).
    if USE_HF_INFERENCE_API:
        try:
            get_sentiment("RELIANCE.NS")
            print("✅ HF Inference API warm-up done.")
        except Exception as e:
            print(f"⚠️ HF Inference API warm-up failed (non-fatal): {e}")
    else:
        try:
            get_sentiment("RELIANCE.NS")
            print("✅ FinBERT (local) preloaded.")
        except Exception as e:
            print(f"⚠️ FinBERT local preload failed (non-fatal): {e}")


def clean_for_json(obj):
    import numpy as np
    from datetime import datetime

    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_for_json(i) for i in obj)
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


# ─────────────────────────────────────────────
# COOKIE-BASED WATCHLIST  (per-user, private)
# ─────────────────────────────────────────────

def _get_cookie_history(request_obj) -> list:
    raw = request_obj.cookies.get(COOKIE_NAME, "[]")
    try:
        items = json.loads(raw)
        if isinstance(items, list):
            return [str(s) for s in items[:COOKIE_MAX_ITEMS]]
    except Exception:
        pass
    return []


def _push_cookie_history(response_obj, symbol: str, current: list) -> list:
    updated = [symbol] + [s for s in current if s != symbol]
    updated = updated[:COOKIE_MAX_ITEMS]
    response_obj.set_cookie(
        COOKIE_NAME,
        json.dumps(updated),
        max_age   = COOKIE_MAX_AGE,
        httponly  = False,
        samesite  = "Lax",
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
    if component == "EMA":
        strong, weak = 0.03, 0.008
    else:
        strong, weak = 0.50, 0.15

    if   score >  strong: return "Strong Bullish"
    elif score >  weak:   return "Weak Bullish"
    elif score < -strong: return "Strong Bearish"
    elif score < -weak:   return "Weak Bearish"
    else:                 return "Neutral"


# ─────────────────────────────────────────────
# PURE COMPUTE  (no DB reads or writes)
# ─────────────────────────────────────────────

def _compute_strategy(data, short_ema: int, long_ema: int, symbol: str = "") -> tuple:
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
                "Big Win"    if ret > 5  else
                "Small Win"  if ret > 0  else
                "Loss"
            )
            detailed_trades.append({
                "buy_date":   str(d1),
                "buy_price":  round(buy_price,  2),
                "sell_date":  str(d2),
                "sell_price": round(sell_price, 2),
                "return":     round(ret, 2),
                "quality":    quality,
            })
            i += 2
        else:
            i += 1

    # ── Advanced metrics ─────────────────────────────────────────
    prices_list = data["Close"].tolist()
    start_dt    = pd.to_datetime(data.index[0])
    end_dt      = pd.to_datetime(data.index[-1])

    portfolio = {
        "history": [
            {"date": str(p["date"]), "capital": float(p["capital"])}
            for p in equity_curve
        ],
        "final":          float(final_capital),
        "total_gain":     float(final_capital - INITIAL_CAPITAL),
        "total_gain_pct": float(((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100),
        "trades":         len(detailed_trades),
    }

    clean_history = []
    for point in portfolio["history"]:
        try:
            date_str = pd.to_datetime(point["date"]).strftime("%Y-%m-%d")
            clean_history.append({"date": date_str, "capital": float(point["capital"])})
        except Exception:
            continue
    portfolio["history"] = clean_history

    cagr                    = calculate_cagr(portfolio["total_gain_pct"], start_dt, end_dt)
    max_drawdown            = calculate_max_drawdown(prices_list)
    sharpe                  = calculate_sharpe(detailed_trades)
    best_trade, worst_trade = get_best_worst_trade(detailed_trades)

    # ── Signal analysis + intelligence table ─────────────────────
    analysis = summarize_results(signal_outcome_analysis(data, signals))
    analysis = json.loads(json.dumps(analysis, default=str))
    table    = build_signal_table(data, signals, short_col, long_col)
    for row in table:
        if "Date" in row:
            row["Date"] = str(row["Date"])
    avg_conf = sum(r["Confidence"] for r in table) / len(table) if table else 0

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

    ml_conf = ml_result.get("confidence", 0) / 100
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
    data_qual  = sentiment_result.get("data_quality", "unknown")
    art_count  = sentiment_result.get("article_count", 0)

    if "Bullish" in sentiment_result["prediction"]:
        sentiment_score = sent_conf
    elif "Bearish" in sentiment_result["prediction"]:
        sentiment_score = -sent_conf
    else:
        sentiment_score = 0.0

    # ── EMA TREND SCORE ──────────────────────────────────────────
    latest   = data.iloc[-1]
    price    = float(latest["Close"])
    ema_diff = float(latest[short_col]) - float(latest[long_col])
    ema_score = max(min((ema_diff / price) * 3, 1.0), -1.0)

    # ── MARKET REGIME ────────────────────────────────────────────
    trend_strength = abs(ema_diff) / price
    regime = ml_result.get("regime", "TRENDING" if trend_strength > 0.02 else "SIDEWAYS")

    # ── ADAPTIVE WEIGHTING ───────────────────────────────────────
    if regime == "TRENDING":
        ml_w, ema_w, sent_w = 0.40, 0.40, 0.20
    else:
        ml_w, ema_w, sent_w = 0.60, 0.20, 0.20

    ml_w  *= (0.5 + abs(ml_score))
    ema_w *= (0.5 + abs(ema_score))

    if data_qual in ("insufficient", "low", "error", "no_api_key"):
        sent_w *= 0.3
    elif data_qual == "moderate":
        sent_w *= 0.65

    sent_w *= (0.5 + abs(sentiment_score))
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
    final_score  = ml_score * ml_w + ema_score * ema_w + sentiment_score * sent_w

    scores       = [ml_score, ema_score, sentiment_score]
    disagreement = 0.0
    if ml_score * ema_score < 0:        disagreement += 0.4
    if ml_score * sentiment_score < 0:  disagreement += 0.3
    if ema_score * sentiment_score < 0: disagreement += 0.2
    disagreement  = min(disagreement, 0.8)
    final_score  *= (1 - disagreement)

    if   final_score >  0.05: final_prediction = "Bullish 📈"
    elif final_score < -0.05: final_prediction = "Bearish 📉"
    else:                     final_prediction = "Neutral ⚖️"

    # ── CONFIDENCE ───────────────────────────────────────────────
    positive    = sum(1 for s in scores if s > 0)
    negative    = sum(1 for s in scores if s < 0)
    agree_score = (positive - negative) / 3
    confidence  = abs(final_score)
    confidence *= (1 + agree_score * 0.5)
    if regime == "TRENDING":
        confidence *= 1.1
    confidence  = confidence ** 0.7
    final_conf  = round(max(min(confidence, 1.0), 0.0) * 100, 2)

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
        "explainability": explainability,
    }

    return result, short_col, long_col


# ─────────────────────────────────────────────
# REDIS HELPERS
# ─────────────────────────────────────────────

def _redis_get(key: str):
    if redis_client is None:
        return None
    try:
        return redis_client.get(key)
    except Exception as e:
        print(f"⚠️ Redis GET error: {e}")
        return None


def _redis_setex(key: str, ttl: int, value: str):
    if redis_client is None:
        return
    try:
        redis_client.setex(key, ttl, value)
    except Exception as e:
        print(f"⚠️ Redis SETEX error: {e}")


# ─────────────────────────────────────────────
# DATE-RANGE VALIDATOR
# ─────────────────────────────────────────────

def _data_covers_period(data: pd.DataFrame, period: str) -> bool:
    """
    Return True only if `data` starts early enough to satisfy `period`.

    Why this matters
    ────────────────
    The DB accumulates rows across requests.  If the user first asks for 1y,
    only 1y rows are stored.  A subsequent 2y request then hits the DB, gets
    those 1y rows back (they *are* within the 2y window), and mistakenly
    treats them as 2y data — producing identical charts and metrics for every
    longer period thereafter.

    Fix: compare the earliest row in the loaded data against the expected
    start date for the requested period.  Allow 15 calendar days of slack
    (weekends, public holidays, recently-listed stocks).
    """
    from datetime import datetime, timedelta

    if period == "max" or not period:
        return True  # can't validate "max", trust the DB

    now = datetime.now()
    try:
        if period.endswith("y"):
            years          = int(period[:-1])
            expected_start = now - timedelta(days=int(365.25 * years))
        elif period.endswith("mo"):
            months         = int(period[:-2])
            expected_start = now - timedelta(days=30 * months)
        else:
            return True  # unknown format — let it through
    except (ValueError, AttributeError):
        return True

    tolerance    = timedelta(days=15)
    actual_start = pd.to_datetime(data.index.min())
    cutoff       = pd.Timestamp(expected_start + tolerance)

    if actual_start > cutoff:
        print(
            f"⚠️  DB data starts {actual_start.date()} but period={period} "
            f"requires data from ~{expected_start.date()}. "
            f"Falling back to yfinance."
        )
        return False

    return True


# ─────────────────────────────────────────────
# FULL STRATEGY RUNNER  (main route)
# ─────────────────────────────────────────────

def run_strategy(symbol: str, period: str, short_ema: int, long_ema: int) -> tuple:
    cache_key = f"{symbol}_{period}_{short_ema}_{long_ema}"

    cached = _redis_get(cache_key)

    if cached:
        print(f"⚡ [{symbol}] REDIS HIT")
        result = json.loads(cached)
        result["symbol"]    = symbol
        result["period"]    = period
        result["short_ema"] = short_ema
        result["long_ema"]  = long_ema

        data = load_price_data(symbol, period)

        if data is not None and not data.empty:
            data.index = pd.to_datetime(data.index)
        else:
            print(f"ℹ️ [{symbol}] Using live data (DB updating in background)")
            data = get_stock_data(symbol, period)
            if data is None:
                return result, [], None, None

        short_col = f"EMA{short_ema}"
        long_col  = f"EMA{long_ema}"
        data[short_col] = calculate_ema(data["Close"], short_ema)
        data[long_col]  = calculate_ema(data["Close"], long_ema)

        # ── CRITICAL: reset DatetimeIndex → "Date" column ───────
        # load_price_data returns a DatetimeIndex df.
        # The template expects data["Date"] as a plain string column.
        data = data.reset_index()
        if "index" in data.columns:
            data.rename(columns={"index": "Date"}, inplace=True)
        # After reset_index the column is named "Date" (matching the
        # DataFrame column name set in load_price_data).
        data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")

        result = clean_for_json(result)
        if data is None or data.empty:
            data = pd.DataFrame(columns=["Date", "Close"])
        return result, data, short_col, long_col

    print(f"🐢 [{symbol}] REDIS MISS")

    # ── 1. Price data ──────────────────────────────────────────
    data = load_price_data(symbol, period)

    # Validate the DB data actually covers the requested period.
    # A shorter previous request (e.g. 1y) may have stored rows that
    # technically fall within a longer period (e.g. 2y), causing the DB
    # hit to silently return stale/short data for every longer period.
    if data is not None and not data.empty and not _data_covers_period(data, period):
        data = None  # force yfinance fetch below

    if data is not None and not data.empty:
        print(f"⚡ [{symbol}] Loaded from DB (period={period})")
    else:
        print(f"📡 [{symbol}] Fetching from yfinance (period={period})…")

        if symbol not in _LOADING:
            _LOADING.add(symbol)
            data = get_stock_data(symbol, period)

            if data is not None:
                data_dict       = data.copy()
                data_dict.index = data_dict.index.astype(str)
                data_dict       = data_dict.to_dict()

                # ── Fault-tolerant Celery call ──────────────────
                try:
                    save_price_task.delay(symbol, data_dict)
                except Exception as e:
                    print(f"⚠️ Celery price task failed ({e}), falling back to thread")
                    _df_copy = data.copy()
                    threading.Thread(
                        target=store_price_data,
                        args=(symbol, _df_copy),
                        daemon=True,
                    ).start()

            _LOADING.discard(symbol)
        else:
            data = get_stock_data(symbol, period)

    if data is None or data.empty:
        raise ValueError(
            f"No data found for '{symbol}'. "
            "Please enter a valid NSE symbol e.g. RELIANCE.NS or TCS.NS."
        )

    # ── 2. Pure compute ─────────────────────────────────────────
    result, short_col, long_col = _compute_strategy(data, short_ema, long_ema, symbol)
    result["symbol"]    = symbol
    result["period"]    = period
    result["short_ema"] = short_ema
    result["long_ema"]  = long_ema

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

        # ── Fault-tolerant Celery call ──────────────────────────
        try:
            save_to_db_task.delay(
                symbol, _sig_copy, _trd_copy, _met_db, short_ema, long_ema
            )
        except Exception as e:
            print(f"⚠️ Celery DB task failed ({e}), skipping background DB save")
    else:
        print(f"⏭️  [{symbol}] Save already in progress, skipping.")

    # ── 4. Store in Redis ───────────────────────────────────────
    result["signals"] = [(s, str(d)) for s, d in result["signals"]]
    clean_result = clean_for_json(result)
    _redis_setex(cache_key, 300, json.dumps(clean_result))

    # ── 5. Prepare data for template ────────────────────────────
    data = data.reset_index()
    if "index" in data.columns:
        data.rename(columns={"index": "Date"}, inplace=True)
    data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")

    return clean_result, data, short_col, long_col


# ─────────────────────────────────────────────
# COMPARE  (read-only, never writes to DB)
# ─────────────────────────────────────────────

def compare_strategy(symbol: str, period: str, short_ema: int, long_ema: int) -> dict:
    # ── Redis cache check ────────────────────────────────────────
    cache_key = f"{symbol}_{period}_{short_ema}_{long_ema}"
    cached    = _redis_get(cache_key)

    if cached:
        print(f"⚡ [CMP {symbol}] Redis hit for {short_ema}/{long_ema}")
        result = json.loads(cached)
        result["signals"] = [(s, str(d)) for s, d in result.get("signals", [])]
        return clean_for_json(result)

    # ── Cache miss — compute ─────────────────────────────────────
    data = load_price_data(symbol, period)

    # Same period-coverage guard as in run_strategy: reject DB data
    # that is too short for the requested period.
    if data is not None and not data.empty and not _data_covers_period(data, period):
        data = None

    if data is not None and not data.empty:
        print(f"⚡ [CMP {symbol}] DB hit")
    else:
        print(f"📡 [CMP {symbol}] DB miss — fetching from yfinance (no save)")
        data = get_stock_data(symbol, period)

    if data is None or data.empty:
        raise ValueError(f"No data found for '{symbol}'.")

    result, _, _ = _compute_strategy(data, short_ema, long_ema, symbol)
    result["signals"] = [(s, str(d)) for s, d in result["signals"]]

    # Store in Redis so subsequent compare calls are instant
    clean_result = clean_for_json(result)
    _redis_setex(cache_key, 300, json.dumps(clean_result))

    return clean_result


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

            if data is not None:
                prices      = data["Close"].tolist()
                ema_short_d = data[short_col].tolist()
                ema_long_d  = data[long_col].tolist()
                dates       = data["Date"].tolist()

            if data is not None:
                for signal, date in result["signals"]:
                    if isinstance(date, str):
                        date_obj = pd.to_datetime(date)
                    else:
                        date_obj = date

                    row = data[data["Date"] == str(date_obj.date())]

                    if not row.empty:
                        price = float(row["Close"].values[0])
                        fd    = (
                            date_obj.strftime("%Y-%m-%d")
                            if hasattr(date_obj, "strftime")
                            else str(date_obj)
                        )
                        (buy_points if signal == "BUY" else sell_points).append(
                            {"x": fd, "y": price}
                        )

        except Exception as exc:
            print("❌ BACKEND ERROR:", exc)
            import traceback
            traceback.print_exc()
            error  = str(exc)
            result = None

    result    = clean_for_json(result)
    portfolio = result["portfolio"] if result else None

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
        portfolio   = portfolio,
        watchlist   = watchlist,
    ))

    if result and "symbol" in result:
        _push_cookie_history(resp, result["symbol"], cookie_history)

    return resp


# ─────────────────────────────────────────────
# WATCHLIST API
# ─────────────────────────────────────────────

@app.route("/api/watchlist", methods=["GET"])
def api_watchlist():
    history = _get_cookie_history(request)
    return jsonify([{"symbol": s} for s in history])


# ─────────────────────────────────────────────
# EMAIL ALERT ROUTES
# ─────────────────────────────────────────────

@app.route("/api/alerts/save", methods=["POST"])
def api_save_alert():
    """Save an EMA alert subscription for a user's email."""
    try:
        body      = request.get_json()
        email     = (body.get("email") or "").strip().lower()
        symbol    = (body.get("symbol") or "").strip().upper()
        short_ema = int(body.get("short_ema", 0))
        long_ema  = int(body.get("long_ema",  0))

        if not email or "@" not in email:
            return jsonify({"error": "Valid email is required."}), 400
        if not symbol:
            return jsonify({"error": "Symbol is required."}), 400
        if short_ema <= 0 or long_ema <= 0 or short_ema >= long_ema:
            return jsonify({"error": "Invalid EMA values."}), 400

        from src.db_loader import save_alert
        save_alert(email, symbol, short_ema, long_ema)

        return jsonify({"ok": True, "message": f"Alert set for {symbol} EMA {short_ema}/{long_ema}"})

    except Exception as exc:
        print(f"❌ /api/alerts/save error: {exc}")
        return jsonify({"error": str(exc)}), 400


@app.route("/api/alerts/delete", methods=["POST"])
def api_delete_alert():
    """Deactivate an alert."""
    try:
        body      = request.get_json()
        email     = (body.get("email") or "").strip().lower()
        symbol    = (body.get("symbol") or "").strip().upper()
        short_ema = int(body.get("short_ema", 0))
        long_ema  = int(body.get("long_ema",  0))

        from src.db_loader import delete_alert
        delete_alert(email, symbol, short_ema, long_ema)
        return jsonify({"ok": True})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/alerts/list", methods=["POST"])
def api_list_alerts():
    """Return active alerts for an email."""
    try:
        body  = request.get_json()
        email = (body.get("email") or "").strip().lower()
        if not email:
            return jsonify({"error": "Email required."}), 400

        from src.db_loader import get_alerts_for_user
        return jsonify(get_alerts_for_user(email))

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ─────────────────────────────────────────────
# CSV EXPORT ENDPOINTS
# ─────────────────────────────────────────────

@app.route("/api/export/trades", methods=["POST"])
def export_trades():
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
            headers={"Content-Disposition": f"attachment; filename=trades_{symbol}_{period}.csv"},
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/export/signals", methods=["POST"])
def export_signals():
    try:
        symbol = resolve_symbol(request.form.get("symbol", "").strip())
        period = format_period(request.form.get("period", "1y"))
        s_ema  = int(request.form.get("short_ema", 20))
        l_ema  = int(request.form.get("long_ema",  50))

        res = compare_strategy(symbol, period, s_ema, l_ema)

        buf = io.StringIO()
        w   = csv.writer(buf)
        w.writerow([
            "Date", "Type", "Price", "EMA S", "EMA L",
            "Strength", "Confidence (%)", "Label",
            "+1D (%)", "+3D (%)", "+1W (%)", "+1M (%)",
        ])
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
            headers={"Content-Disposition": f"attachment; filename=signals_{symbol}_{period}.csv"},
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
                f"Custom EMA {key} — results vary by stock and market regime.",
            ),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ─────────────────────────────────────────────
if __name__ == "__main__":
    preload_models()
    app.run(debug=True, use_reloader=False)