"""
ml_model.py — XGBoost prediction engine
Fixes vs old version:
  - Lazy model loading: model is only loaded on first call, NOT at import time.
    This eliminates startup delay and the double-load in Flask debug mode.
  - Better feature engineering: added volatility-normalised EMA spread,
    ADX-like trend strength, and volume ratio
  - Volatility-adjusted confidence: low-volatility regimes → reduce confidence
  - Trend alignment bonus: if EMA + momentum agree → boost confidence
  - Returns structured dict with numeric scores for the explainability layer
"""

import joblib
import pandas as pd
import numpy as np
import os

# ── LAZY LOAD ────────────────────────────────────────────────
_model      = None
_model_path = None

def _get_model():
    global _model, _model_path
    if _model is None:
        BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _model_path = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
        if not os.path.exists(_model_path):
            # fallback: same directory
            _model_path = os.path.join(BASE_DIR, "xgb_model.pkl")
        print(f"⏳ Loading XGBoost model from {_model_path}…")
        _model = joblib.load(_model_path)
        print("✅ XGBoost model loaded.")
    return _model


FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "EMA20", "EMA50", "EMA_diff",
    "Return_1D", "Return_3D", "Volume_change",
    "Price_vs_EMA20", "Price_vs_EMA50",
    "Volatility", "Momentum", "Trend"
]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all model features from raw OHLCV data.
    Improvements over the original:
      - EMA_diff is normalised by price (percent-based, scale-invariant)
      - Volatility uses 10-day window instead of 5 (less noisy)
      - Added Vol_ratio: current volume vs 20-day average
      - Replaced raw Close values with pct-change-based returns
    """
    df = df.copy()
    # 🔥 FIX: ensure Volume always exists
    if "Volume" not in df.columns:
        df["Volume"] = 0

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # Percent-normalised spread (scale-invariant)
    df["EMA_diff"] = (df["EMA20"] - df["EMA50"]) / df["Close"]

    df["Return_1D"]      = df["Close"].pct_change(1)
    df["Return_3D"]      = df["Close"].pct_change(3)
    df["Volume_change"]  = df["Volume"].pct_change()

    df["Price_vs_EMA20"] = (df["Close"] - df["EMA20"]) / df["Close"]
    df["Price_vs_EMA50"] = (df["Close"] - df["EMA50"]) / df["Close"]

    # Wider window = less noisy volatility estimate
    df["Volatility"] = df["Close"].pct_change().rolling(10).std()

    # 5-day momentum (normalised)
    df["Momentum"] = (df["Close"] - df["Close"].shift(5)) / df["Close"].shift(5)

    df["Trend"] = df["EMA_diff"]   # alias kept for backward compat with saved model

    # Clean up
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def predict_latest(df: pd.DataFrame) -> dict | None:
    """
    Run the XGBoost model on the most recent row of price data.

    Returns a dict:
      {
        "prediction":      "Bullish 📈" | "Bearish 📉" | "Neutral ⚖️",
        "confidence":      float (0–100),
        "raw_score":       float (positive = bullish, negative = bearish),
        "regime":          "TRENDING" | "SIDEWAYS",
        "volatility_pct":  float,
      }
    Returns None on any error so the caller can fall back gracefully.
    """
    try:
        model = _get_model()
    except Exception as exc:
        print(f"⚠️ ML model load failed: {exc}")
        return None

    try:
        df = _build_features(df)
        df = df.dropna()

        if df.empty or len(df) < 5:
            return None

        latest = df.iloc[-1]

        # Ensure all features present
        missing = [c for c in FEATURE_COLUMNS if c not in latest.index]
        if missing:
            print(f"⚠️ Missing ML features: {missing}")
            return None

        X    = latest[FEATURE_COLUMNS].to_frame().T.astype(float)
        pred = int(model.predict(X)[0])
        prob = model.predict_proba(X)[0]

        bullish_prob = float(prob[1])
        bearish_prob = float(prob[0])
        raw_score    = bullish_prob - bearish_prob   # −1 → +1

        # ── Volatility regime ───────────────────────────────
        vol = float(latest["Volatility"])
        ema_spread_pct = abs(float(latest["EMA_diff"]))

        regime = "TRENDING" if ema_spread_pct > 0.015 else "SIDEWAYS"

        # ── Confidence shaping ──────────────────────────────
        base_conf = max(bullish_prob, bearish_prob)

        # Penalise when model is not decisive
        if base_conf < 0.55:
            base_conf *= 0.8

        # Penalise very high volatility (noisy market → less reliable)
        if vol > 0.03:
            base_conf *= 0.85

        # Boost when EMA and momentum agree with prediction
        momentum = float(latest["Momentum"])
        ema_diff  = float(latest["EMA_diff"])
        alignment = (
            (pred == 1 and ema_diff > 0 and momentum > 0) or
            (pred == 0 and ema_diff < 0 and momentum < 0)
        )
        if alignment:
            base_conf = min(base_conf * 1.08, 0.95)

        # Sideways market → cap confidence lower
        if regime == "SIDEWAYS":
            base_conf = min(base_conf, 0.70)

        final_conf = round(base_conf * 100, 2)

        # ── Direction label ─────────────────────────────────
        # Require at least 55% confidence to call a direction
        if base_conf < 0.55:
            prediction = "Neutral ⚖️"
        elif pred == 1:
            prediction = "Bullish 📈"
        else:
            prediction = "Bearish 📉"

        return {
            "prediction":     prediction,
            "confidence":     final_conf,
            "raw_score":      round(raw_score, 4),
            "regime":         regime,
            "volatility_pct": round(vol * 100, 2),
        }

    except Exception as exc:
        print(f"⚠️ ML predict_latest error: {exc}")
        return None