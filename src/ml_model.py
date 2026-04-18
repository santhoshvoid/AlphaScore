import joblib
import pandas as pd
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
model = joblib.load(model_path)


def predict_latest(df):
    df = df.copy()

    # Features
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA_diff"] = df["EMA20"] - df["EMA50"]

    df["Return_1D"] = df["Close"].pct_change(1)
    df["Return_3D"] = df["Close"].pct_change(3)
    df["Volume_change"] = df["Volume"].pct_change()

    df["Price_vs_EMA20"] = df["Close"] - df["EMA20"]
    df["Price_vs_EMA50"] = df["Close"] - df["EMA50"]

    df["Volatility"] = df["Close"].rolling(5).std()
    df["Momentum"] = df["Close"] - df["Close"].shift(5)
    df["Trend"] = df["EMA20"] - df["EMA50"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    

    if df.empty:
        return None

    latest = df.iloc[-1]

    X = latest.drop(["Target"], errors="ignore")

    FEATURE_COLUMNS = [
        "Open", "High", "Low", "Close", "Volume",
        "EMA20", "EMA50", "EMA_diff",
        "Return_1D", "Return_3D", "Volume_change",
        "Price_vs_EMA20", "Price_vs_EMA50",
        "Volatility", "Momentum", "Trend"
    ]

    X = X[FEATURE_COLUMNS]
    X = X.to_frame().T

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][pred]

    return {
        "prediction": "Bullish 📈" if pred == 1 else "Bearish 📉",
        "confidence": round(prob * 100, 2)
    }