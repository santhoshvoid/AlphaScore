import pandas as pd
from difflib import get_close_matches

# Load NSE dataset once
df = pd.read_csv("data/nse_stocks.csv")

# Preprocess for faster matching
df["symbol_lower"] = df["SYMBOL"].str.lower()
df["name_lower"] = df["NAME OF COMPANY"].str.lower()

ALL_NAMES = df["name_lower"].tolist()
ALL_SYMBOLS = df["symbol_lower"].tolist()


def resolve_symbol(user_input):
    user_input = user_input.lower().strip()

    # 🔹 1. Direct symbol input
    if user_input.endswith(".ns"):
        return user_input.upper()

    if user_input in ALL_SYMBOLS:
        return user_input.upper() + ".NS"

    # 🔹 2. Exact company name
    exact = df[df["name_lower"] == user_input]
    if not exact.empty:
        return exact.iloc[0]["SYMBOL"] + ".NS"

    # 🔹 3. Strong partial match (word-based)
    words = user_input.split()

    def score(name):
        return sum(word in name for word in words)

    df["score"] = df["name_lower"].apply(score)
    best = df[df["score"] > 0].sort_values(by="score", ascending=False)

    if not best.empty:
        return best.iloc[0]["SYMBOL"] + ".NS"

    # 🔹 4. Fuzzy fallback
    matches = get_close_matches(user_input, ALL_NAMES, n=1, cutoff=0.5)
    if matches:
        row = df[df["name_lower"] == matches[0]].iloc[0]
        return row["SYMBOL"] + ".NS"

    # 🔹 5. Final fallback
    return user_input.upper() + ".NS"
