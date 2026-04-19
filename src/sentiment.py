"""
sentiment.py — FinBERT-based news sentiment analysis
Fixes vs old version:
  - Lazy model loading: FinBERT is only loaded on first call, NOT at import time.
    This eliminates the 8–12 second startup delay when running python app.py.
  - Minimum data check: if < 3 headlines, confidence is scaled down to 50%
  - Article quality scoring: penalizes junk/irrelevant articles
  - Confidence scaling: low article count → penalize final confidence
  - Neutral fallback: weak sentiment doesn't swing the final decision
  - Capped confidence at 85% (no headline set is 100% reliable)
"""

import requests
import torch
import numpy as np
from dotenv import load_dotenv
import os
import time

load_dotenv()

# ── LAZY LOAD: model is None until first call ─────────────────
_tokenizer = None
_model     = None

def _load_model():
    global _tokenizer, _model

    if _tokenizer is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        MODEL_NAME = "ProsusAI/finbert"
        HF_TOKEN = os.getenv("HF_TOKEN")

        print("⏳ Loading FinBERT model...")

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN
        )

        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN
        )

        print("✅ FinBERT loaded.")


# ── CACHE: symbol → (result_dict, unix_timestamp) ────────────
_sentiment_cache: dict = {}
_CACHE_TTL = 600   # 10 minutes


# ─────────────────────────────────────────────
# NEWS FETCH
# ─────────────────────────────────────────────

API_KEY = os.getenv("NEWS_API_KEY", "")

def get_news(symbol: str) -> list:
    """
    Fetch recent news headlines for a stock symbol.
    Returns a list of (text, quality_score) tuples.
    """
    if not API_KEY:
        return []

    clean_symbol = symbol.replace(".NS", "").replace(".BSE", "")

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={clean_symbol} stock OR {clean_symbol} company OR {clean_symbol} earnings"
        f"&language=en&sortBy=publishedAt&pageSize=15&apiKey={API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return []

        articles = response.json().get("articles", [])

        headlines = []
        FINANCIAL_KEYWORDS = [
            "earnings", "revenue", "profit", "loss", "guidance", "deal",
            "order", "growth", "quarterly", "annual", "acquisition",
            "merger", "dividend", "buyback", "upgrade", "downgrade",
            "target price", "analyst", "results", "beat", "miss"
        ]
        JUNK_KEYWORDS = ["rumor", "tweet", "reddit", "viral", "meme", "gossip"]

        for a in articles:
            title = (a.get("title") or "").strip()
            desc  = (a.get("description") or "").strip()

            if not title or len(title) < 20:
                continue

            text = title + (". " + desc if desc else "")
            text_lower = text.lower()

            # Hard filter: skip junk
            if any(w in text_lower for w in JUNK_KEYWORDS):
                continue

            # Quality score
            quality = sum(1 for w in FINANCIAL_KEYWORDS if w in text_lower)
            headlines.append((text, quality))

        # Sort by quality descending, take top 10
        headlines.sort(key=lambda x: x[1], reverse=True)
        return headlines[:10]

    except Exception:
        return []


# ─────────────────────────────────────────────
# SENTIMENT ANALYSIS
# ─────────────────────────────────────────────

def analyze_sentiment(headlines_with_scores: list) -> dict:
    """
    Run FinBERT on a list of (text, quality_score) tuples.

    Validation rules:
      - Fewer than 3 articles → LOW_DATA flag, confidence scaled to 50%
      - Weak aggregate score (|avg| < 0.08) → returns Neutral regardless
      - Maximum confidence capped at 85% (headline sentiment is noisy)
      - Quality-weighted averaging: financial news counts more
    """
    if not headlines_with_scores:
        return {"prediction": "Neutral ⚖️", "confidence": 0, "article_count": 0, "data_quality": "insufficient"}

    _load_model()

    texts  = [h[0] for h in headlines_with_scores]
    weights = [max(h[1], 0.5) for h in headlines_with_scores]  # min weight 0.5

    inputs  = _tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = _model(**inputs)

    # FinBERT labels: 0=negative, 1=neutral, 2=positive
    probs  = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
    # Score = positive_prob - negative_prob  →  [-1, +1]
    scores = [float(p[2] - p[0]) for p in probs]

    # Weighted average
    total_weight = sum(weights)
    avg_score    = sum(s * w for s, w in zip(scores, weights)) / total_weight

    article_count = len(headlines_with_scores)
    data_quality  = "good" if article_count >= 6 else ("moderate" if article_count >= 3 else "low")

    # ── Confidence calculation ──────────────────────────────
    raw_confidence = min(abs(avg_score) * 1.2, 1.0)

    # Penalize low article count
    if article_count < 3:
        raw_confidence *= 0.5
        data_quality    = "insufficient"
    elif article_count < 6:
        raw_confidence *= 0.75

    # Hard cap: sentiment from headlines is noisy — never trust it >85%
    raw_confidence = min(raw_confidence, 0.85)

    # ── Direction ──────────────────────────────────────────
    # Weak signal → Neutral (don't swing the decision on weak sentiment)
    if abs(avg_score) < 0.08:
        prediction = "Neutral ⚖️"
        raw_confidence = min(raw_confidence, 0.35)   # low confidence on neutral
    elif avg_score > 0:
        prediction = "Bullish 📈"
    else:
        prediction = "Bearish 📉"

    return {
        "prediction":    prediction,
        "confidence":    round(raw_confidence * 100, 2),
        "article_count": article_count,
        "data_quality":  data_quality,
        "raw_score":     round(avg_score, 4),
    }


# ─────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────

def get_sentiment(symbol: str) -> dict:
    """
    Returns sentiment dict with caching (10-minute TTL).
    Safe fallback: returns Neutral/0 if API key missing or error.
    """
    if not API_KEY:
        return {
            "prediction": "Neutral ⚖️",
            "confidence": 0,
            "article_count": 0,
            "data_quality": "no_api_key",
        }

    now = time.time()

    if symbol in _sentiment_cache:
        cached_result, cached_time = _sentiment_cache[symbol]
        if now - cached_time < _CACHE_TTL:
            return cached_result

    try:
        headlines = get_news(symbol)
        result    = analyze_sentiment(headlines)
    except Exception as exc:
        print(f"⚠️ Sentiment error for {symbol}: {exc}")
        result = {
            "prediction":    "Neutral ⚖️",
            "confidence":    0,
            "article_count": 0,
            "data_quality":  "error",
        }

    _sentiment_cache[symbol] = (result, now)
    return result