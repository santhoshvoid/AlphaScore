import pandas as pd
from src.db_loader import load_price_data
from src.indicators import calculate_ema


def create_features(df):
    # EMAs
    df["EMA20"] = calculate_ema(df["Close"], 20)
    df["EMA50"] = calculate_ema(df["Close"], 50)

    # EMA difference
    df["EMA_diff"] = df["EMA20"] - df["EMA50"]

    # Returns
    df["Return_1D"] = df["Close"].pct_change(1)
    df["Return_3D"] = df["Close"].pct_change(3)

    # Volume change
    df["Volume_change"] = df["Volume"].pct_change()

    # Extra features (good for ML)
    df["Price_vs_EMA20"] = df["Close"] - df["EMA20"]
    df["Price_vs_EMA50"] = df["Close"] - df["EMA50"]

    return df


def create_target(df, days_ahead=3):
    df["Future_Close"] = df["Close"].shift(-days_ahead)
    df["Target"] = (df["Future_Close"] > df["Close"]).astype(int)
    return df


def build_dataset(symbols):
    all_data = []

    for symbol in symbols:
        print(f"Processing {symbol}...")

        df = load_price_data(symbol)

        if df is None or df.empty or len(df) < 100:
            continue

        df = df.copy()
        df = create_features(df)
        df = create_target(df)

        df["symbol"] = symbol

        all_data.append(df)

    if not all_data:
        raise ValueError("No valid data found")

    final_df = pd.concat(all_data)

    # Drop NaN rows
    final_df = final_df.dropna()

    # 🔥 DROP THIS HERE (correct place)
    final_df = final_df.drop(columns=["Future_Close"])

    return final_df


def get_all_symbols():
    from src.db import get_connection

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT symbol FROM stocks;")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [r[0] for r in rows]


if __name__ == "__main__":
    symbols = get_all_symbols()

    print(f"Found {len(symbols)} stocks in DB")

    dataset = build_dataset(symbols)

    dataset.to_csv("dataset.csv", index=False)

    print(f"✅ Dataset saved. Rows: {len(dataset)}")