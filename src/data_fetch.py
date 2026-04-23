import yfinance as yf
from src.db import get_connection
import pandas as pd
from datetime import datetime, timedelta
import time


def _period_to_dates(period: str):
    """
    Convert a period string (e.g. '2y', '6mo') to an explicit (start, end)
    date pair as 'YYYY-MM-DD' strings.

    Why this matters:
      yf.download(period='2y')  -> Yahoo Finance query param ?period=2y
      yf.download(start=..., end=...) -> Yahoo Finance v8 chart API with
                                         Unix timestamp params

    The second URL path is less aggressive on Yahoo's rate-limit quota,
    so switching to explicit dates makes the primary fetch succeed the
    first time rather than hitting a rate limit and falling through to
    the max fallback.

    Returns (None, None) for 'max' so the caller uses period='max' directly.
    """
    if not period or period == "max":
        return None, None

    period = period.lower().strip()
    end    = datetime.now()

    try:
        if period.endswith("y"):
            years = int(period[:-1])
            start = end - timedelta(days=int(365.25 * years))
        elif period.endswith("mo"):
            months = int(period[:-2])
            start  = end - timedelta(days=30 * months)
        else:
            return None, None

        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    except (ValueError, AttributeError):
        return None, None


def get_stock_data(symbol, period):
    try:
        print(f"Fetching: {symbol} | Period: {period}")

        # Primary fetch: use explicit start/end dates
        # This hits Yahoo's v8 chart API directly (timestamp params) which
        # is far less prone to rate limiting than the period= query path.
        # threads=False avoids parallel connections that trigger throttling
        # on shared-IP cloud hosts (Render, Railway, Fly.io, etc.).
        start_date, end_date = _period_to_dates(period)

        if start_date:
            data = yf.download(
                symbol,
                start   = start_date,
                end     = end_date,
                progress= False,
                threads = False,
            )
        else:
            # period == "max" - no date math, just pass it through
            data = yf.download(symbol, period="max", progress=False, threads=False)

        # Retry once with a short pause if Yahoo still rate-limits
        if data is None or data.empty:
            print(f"Warning: Empty response on first attempt, retrying in 2s...")
            time.sleep(2)

            if start_date:
                data = yf.download(
                    symbol,
                    start   = start_date,
                    end     = end_date,
                    progress= False,
                    threads = False,
                )
            else:
                data = yf.download(symbol, period="max", progress=False, threads=False)

        # Last-resort fallback: max period + trim
        # Only reached if both attempts above failed (very rare).
        # We trim back to the requested period so we never cache 10-year
        # data under a '2y' Redis key.
        if (data is None or data.empty) and period != "max":
            print("Warning: Both attempts failed, falling back to MAX period...")
            data = yf.download(symbol, period="max", progress=False, threads=False)

            if data is not None and not data.empty and start_date:
                data = data[data.index >= pd.Timestamp(start_date)]
                print(f"Trimmed max fallback to {period} ({len(data)} rows)")

        if data is None or data.empty:
            print("No data found even after fallback")
            return None

        # HANDLE MULTI-INDEX (important for some stocks)
        if hasattr(data.columns, "levels"):
            data.columns = data.columns.get_level_values(0)

        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                print(f"Missing column: {col}")
                return None

        # Clean data
        data = data[required_cols].dropna()

        if data.empty:
            print("Data empty after cleaning")
            return None

        return data

    except Exception as e:
        print("Error fetching data:", e)
        return None


def store_price_data(symbol, df):
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Prepare batch data
        data_to_insert = []

        for index, row in df.iterrows():

            # always convert safely
            index = pd.to_datetime(index)

            data_to_insert.append((
                symbol,
                index.date(),
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                int(row['Volume'])
            ))

        # Batch insert (MUCH faster)
        cur.executemany("""
            INSERT INTO prices (symbol, date, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date) DO NOTHING;
        """, data_to_insert)

        conn.commit()
        cur.close()
        conn.close()

        print(f"Stored data for {symbol}")

    except Exception as e:
        print("Error storing data:", e)