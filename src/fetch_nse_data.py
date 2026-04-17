import os
import pandas as pd

def fetch_and_save_nse_data():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

    print("Fetching NSE stock list...")
    df = pd.read_csv(url)

    df = df[["SYMBOL", "NAME OF COMPANY"]]

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/nse_stocks.csv", index=False)

    print("✅ Saved as data/nse_stocks.csv")

if __name__ == "__main__":
    fetch_and_save_nse_data()