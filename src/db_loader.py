"""
db_loader.py  — All PostgreSQL read/write helpers for AlphaCross
Fixes vs old version:
  - load_price_data now accepts a period param and filters by start_date
  - save_metrics uses ON CONFLICT DO UPDATE (upsert) instead of DO NOTHING
  - save_signals / save_trades use specific conflict column lists
  - Added save_stock()  → upserts symbol into the stocks table
  - Added get_watchlist() → returns recently analysed symbols
"""

from src.db import get_connection
import pandas as pd
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _period_to_start_date(period: str):
    """Convert a period string (1y, 6mo, 3mo…) to a Python date."""
    if not period or period == "max":
        return None
    now = datetime.now()
    if period.endswith("y"):
        years = int(period[:-1])
        return (now - timedelta(days=int(365.25 * years))).date()
    if period.endswith("mo"):
        months = int(period[:-2])
        return (now - timedelta(days=30 * months)).date()
    return None


# ─────────────────────────────────────────────
# PRICE DATA
# ─────────────────────────────────────────────

def load_price_data(symbol: str, period: str = None):
    """
    Load OHLCV data from the prices table.
    If `period` is given (e.g. '1y', '6mo') the rows are filtered to that
    date range so the DB result always matches what yfinance would return.
    Returns a DataFrame with DatetimeIndex, or None if no rows found.
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()

        start_date = _period_to_start_date(period)

        if start_date:
            cur.execute("""
                SELECT date, open, high, low, close, volume
                FROM   prices
                WHERE  symbol = %s
                  AND  date  >= %s
                ORDER  BY date ASC;
            """, (symbol, start_date))
        else:
            cur.execute("""
                SELECT date, open, high, low, close, volume
                FROM   prices
                WHERE  symbol = %s
                ORDER  BY date ASC;
            """, (symbol,))

        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return None

        df = pd.DataFrame(
            rows,
            columns=["Date", "Open", "High", "Low", "Close", "Volume"]
        )
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        return df

    except Exception as e:
        print(f"⚠️ load_price_data error: {e}")
        return None


# ─────────────────────────────────────────────
# STOCKS / WATCHLIST
# ─────────────────────────────────────────────

def save_stock(symbol: str):
    """
    Upsert a symbol into the stocks table (tracks recently analysed stocks).
    Requires a UNIQUE constraint on stocks.symbol — run this SQL once in Supabase:
        ALTER TABLE stocks ADD CONSTRAINT stocks_symbol_unique UNIQUE (symbol);
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO stocks (symbol, last_updated)
            VALUES (%s, NOW())
            ON CONFLICT (symbol)
            DO UPDATE SET last_updated = NOW();
        """, (symbol,))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ save_stock error (non-critical): {e}")


def get_watchlist(limit: int = 8):
    """
    Return the most recently analysed symbols from the stocks table.
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT symbol, last_updated
            FROM   stocks
            ORDER  BY last_updated DESC
            LIMIT  %s;
        """, (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [{"symbol": r[0], "last_updated": str(r[1])} for r in rows]
    except Exception as e:
        print(f"⚠️ get_watchlist error: {e}")
        return []


# ─────────────────────────────────────────────
# SIGNALS
# ─────────────────────────────────────────────

def save_signals(symbol: str, signals: list, short_ema: int, long_ema: int):
    """
    Insert crossover signals. Requires this UNIQUE constraint in Supabase:
        ALTER TABLE signals ADD CONSTRAINT signals_unique
            UNIQUE (symbol, date, ema_short, ema_long);
    """
    if not signals:
        return
    try:
        conn = get_connection()
        cur  = conn.cursor()
        for signal, date in signals:
            cur.execute("""
                INSERT INTO signals (symbol, date, signal, ema_short, ema_long)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date, ema_short, ema_long)
                DO NOTHING;
            """, (symbol, date, signal, short_ema, long_ema))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ save_signals error: {e}")


# ─────────────────────────────────────────────
# TRADES
# ─────────────────────────────────────────────

def save_trades(symbol: str, trades: list):
    """
    Insert completed trades. Requires:
        ALTER TABLE trades ADD CONSTRAINT trades_unique
            UNIQUE (symbol, buy_date, sell_date);
    """
    if not trades:
        return
    try:
        conn = get_connection()
        cur  = conn.cursor()
        for t in trades:
            cur.execute("""
                INSERT INTO trades
                    (symbol, buy_date, sell_date, buy_price, sell_price, return)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, buy_date, sell_date)
                DO NOTHING;
            """, (
                symbol,
                t["buy_date"],
                t["sell_date"],
                t["buy_price"],
                t["sell_price"],
                t["return"],
            ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ save_trades error: {e}")


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def save_metrics(symbol: str, metrics: dict, short_ema: int, long_ema: int):
    """
    Upsert strategy metrics. Requires:
        ALTER TABLE metrics ADD CONSTRAINT metrics_unique
            UNIQUE (symbol, short_ema, long_ema);
    Uses DO UPDATE so subsequent runs always refresh the stored values.
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO metrics
                (symbol, short_ema, long_ema,
                 total_return, win_rate, sharpe, max_drawdown)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, short_ema, long_ema)
            DO UPDATE SET
                total_return = EXCLUDED.total_return,
                win_rate     = EXCLUDED.win_rate,
                sharpe       = EXCLUDED.sharpe,
                max_drawdown = EXCLUDED.max_drawdown;
        """, (
            symbol,
            short_ema,
            long_ema,
            metrics.get("Total Return (%)", 0),
            metrics.get("Win Rate (%)",     0),
            metrics.get("Sharpe",           0),
            metrics.get("Max Drawdown",     0),
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ save_metrics error: {e}")


def get_cached_metrics(symbol: str, short_ema: int, long_ema: int):
    """Retrieve previously saved metrics for a symbol+EMA combo."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT total_return, win_rate, sharpe, max_drawdown
            FROM   metrics
            WHERE  symbol    = %s
              AND  short_ema = %s
              AND  long_ema  = %s;
        """, (symbol, short_ema, long_ema))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return None
        return {
            "total_return": row[0],
            "win_rate":     row[1],
            "sharpe":       row[2],
            "max_drawdown": row[3],
        }
    except Exception as e:
        print(f"⚠️ get_cached_metrics error: {e}")
        return None