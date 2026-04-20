"""
db_loader.py  — All PostgreSQL read/write helpers for AlphaCross
Changes vs previous version:
  - Added save_alert()           — store email alert subscription
  - Added get_alerts_for_user()  — get alerts for a specific email
  - Added get_all_active_alerts()— get all active alerts (for Celery task)
  - Added delete_alert()         — deactivate an alert
  All other functions are unchanged.
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
    If `period` is given the rows are filtered to that date range.
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
    """Upsert a symbol into the stocks table (tracks recently analysed stocks)."""
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
    """Return the most recently analysed symbols from the stocks table."""
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
    """Insert crossover signals (deduped via ON CONFLICT DO NOTHING)."""
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
    """Insert completed trades (deduped via ON CONFLICT DO NOTHING)."""
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
    """Upsert strategy metrics (DO UPDATE refreshes existing rows)."""
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


def load_metrics(symbol: str, short_ema: int, long_ema: int):
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
        print(f"⚠️ load_metrics error: {e}")
        return None


def signals_exist(symbol, short_ema, long_ema):
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT 1 FROM signals
            WHERE symbol=%s AND ema_short=%s AND ema_long=%s
            LIMIT 1
        """, (symbol, short_ema, long_ema))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row is not None
    except Exception as e:
        print(f"⚠️ signals_exist error: {e}")
        return False


def trades_exist(symbol):
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT 1 FROM trades WHERE symbol=%s LIMIT 1
        """, (symbol,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row is not None
    except Exception as e:
        print(f"⚠️ trades_exist error: {e}")
        return False


# ─────────────────────────────────────────────
# EMAIL ALERTS  (new)
# ─────────────────────────────────────────────

def save_alert(email: str, symbol: str, short_ema: int, long_ema: int):
    """
    Store an EMA convergence alert subscription.
    Uses ON CONFLICT DO UPDATE to reactivate a previously deleted alert.
    Requires the alerts table — run this SQL in Supabase once:

        CREATE TABLE alerts (
            id        SERIAL PRIMARY KEY,
            email     TEXT    NOT NULL,
            symbol    TEXT    NOT NULL,
            short_ema INTEGER NOT NULL,
            long_ema  INTEGER NOT NULL,
            active    BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE (email, symbol, short_ema, long_ema)
        );
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO alerts (email, symbol, short_ema, long_ema, active)
            VALUES (%s, %s, %s, %s, TRUE)
            ON CONFLICT (email, symbol, short_ema, long_ema)
            DO UPDATE SET active = TRUE, created_at = NOW();
        """, (email, symbol, short_ema, long_ema))
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Alert saved: {email} → {symbol} EMA {short_ema}/{long_ema}")
    except Exception as e:
        print(f"⚠️ save_alert error: {e}")
        raise


def delete_alert(email: str, symbol: str, short_ema: int, long_ema: int):
    """Deactivate (soft-delete) an alert."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            UPDATE alerts SET active = FALSE
            WHERE  email     = %s
              AND  symbol    = %s
              AND  short_ema = %s
              AND  long_ema  = %s;
        """, (email, symbol, short_ema, long_ema))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ delete_alert error: {e}")
        raise


def get_alerts_for_user(email: str) -> list:
    """Return all active alerts for a given email address."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT symbol, short_ema, long_ema, created_at
            FROM   alerts
            WHERE  email = %s AND active = TRUE
            ORDER  BY created_at DESC;
        """, (email,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {
                "symbol":    r[0],
                "short_ema": r[1],
                "long_ema":  r[2],
                "created_at": str(r[3]),
            }
            for r in rows
        ]
    except Exception as e:
        print(f"⚠️ get_alerts_for_user error: {e}")
        return []


def get_all_active_alerts() -> list:
    """Return all active alert subscriptions (used by Celery task)."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT email, symbol, short_ema, long_ema
            FROM   alerts
            WHERE  active = TRUE
            ORDER  BY symbol;
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {
                "email":     r[0],
                "symbol":    r[1],
                "short_ema": r[2],
                "long_ema":  r[3],
            }
            for r in rows
        ]
    except Exception as e:
        print(f"⚠️ get_all_active_alerts error: {e}")
        return []