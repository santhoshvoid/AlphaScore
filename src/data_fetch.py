import yfinance as yf
from src.db import get_connection
import pandas as pd


def get_stock_data(symbol, period):
    try:
        print(f"Fetching: {symbol} | Period: {period}")

        data = yf.download(symbol, period=period, progress=False, threads=True)

        # 🔥 Fallback if empty
        if data is None or data.empty:
            print("⚠️ No data, retrying with MAX period...")
            data = yf.download(symbol, period="max", progress=False)

        if data is None or data.empty:
            print("❌ No data found even after fallback")
            return None

        # ✅ HANDLE MULTI-INDEX (important for some stocks)
        if hasattr(data.columns, "levels"):
            data.columns = data.columns.get_level_values(0)

        # ✅ Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                print(f"❌ Missing column: {col}")
                return None

        # ✅ Clean data
        data = data[required_cols].dropna()

        if data.empty:
            print("❌ Data empty after cleaning")
            return None

        return data

    except Exception as e:
        print("❌ Error fetching data:", e)
        return None


def store_price_data(symbol, df):
    try:
        conn = get_connection()
        cur = conn.cursor()

        # 🔥 Prepare batch data
        data_to_insert = []

        for index, row in df.iterrows():

            # 🔥 FIX: always convert safely
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

        # 🔥 Batch insert (MUCH faster)
        cur.executemany("""
            INSERT INTO prices (symbol, date, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date) DO NOTHING;
        """, data_to_insert)

        conn.commit()
        cur.close()
        conn.close()

        print(f"✅ Stored data for {symbol}")

    except Exception as e:
        print("❌ Error storing data:", e)