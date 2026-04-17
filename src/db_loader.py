from src.db import get_connection
import pandas as pd

def load_price_data(symbol):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT date, open, high, low, close, volume
        FROM prices
        WHERE symbol = %s
        ORDER BY date ASC;
    """, (symbol,))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    if not rows:
        return None

    import pandas as pd

    df = pd.DataFrame(rows, columns=[
        "Date", "Open", "High", "Low", "Close", "Volume"
    ])
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    return df