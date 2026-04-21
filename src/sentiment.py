"""
sentiment.py — News sentiment analysis for AlphaCross

Production path  (USE_HF_INFERENCE_API=true  OR  just default now):
  Uses VADER (Valence Aware Dictionary and sEntiment Reasoner).
  - 100% offline, zero API calls, no rate limits, instant
  - Ships as a tiny pip package: vaderSentiment
  - Specifically designed for news / social-media text
  - Compound score: -1 (very negative) → +1 (very positive)

Development path  (USE_HF_INFERENCE_API=false):
  Uses local FinBERT model (original behaviour, unchanged).

Why we moved away from HuggingFace Inference API:
  ProsusAI/finbert is not hosted on HF's free serverless endpoint —
  it returns 404 regardless of token validity.
"""

import requests
import numpy as np
from dotenv import load_dotenv
import os
import time

load_dotenv()

USE_VADER = os.getenv("USE_VADER", "true").lower() == "true"

_tokenizer = None
_model     = None
_vader     = None


def _get_vader():
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
            print("✅ VADER sentiment analyser loaded.")
        except ImportError:
            print("⚠️ vaderSentiment not installed. Run: pip install vaderSentiment")
            _vader = None
    return _vader


def _load_finbert():
    global _tokenizer, _model
    if _tokenizer is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        MODEL_NAME = "ProsusAI/finbert"
        HF_TOKEN   = os.getenv("HF_TOKEN")
        print("⏳ Loading FinBERT model locally…")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        _model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        print("✅ FinBERT loaded locally.")


_sentiment_cache: dict = {}
_CACHE_TTL = 600


API_KEY = os.getenv("NEWS_API_KEY", "")


def get_news(symbol: str) -> list:
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
        FINANCIAL_KEYWORDS = [
            "earnings", "revenue", "profit", "loss", "guidance", "deal",
            "order", "growth", "quarterly", "annual", "acquisition",
            "merger", "dividend", "buyback", "upgrade", "downgrade",
            "target price", "analyst", "results", "beat", "miss",
        ]
        JUNK_KEYWORDS = ["rumor", "tweet", "reddit", "viral", "meme", "gossip"]

        headlines = []
        for a in articles:
            title = (a.get("title") or "").strip()
            desc  = (a.get("description") or "").strip()
            if not title or len(title) < 20:
                continue
            text       = title + (". " + desc if desc else "")
            text_lower = text.lower()
            if any(w in text_lower for w in JUNK_KEYWORDS):
                continue
            quality = sum(1 for w in FINANCIAL_KEYWORDS if w in text_lower)
            headlines.append((text, quality))

        headlines.sort(key=lambda x: x[1], reverse=True)
        return headlines[:10]

    except Exception:
        return []


def _scores_via_vader(texts: list) -> list:
    vader = _get_vader()
    if vader is None:
        return [0.0] * len(texts)
    scores = []
    for text in texts:
        try:
            vs = vader.polarity_scores(text)
            scores.append(float(vs["compound"]))
        except Exception:
            scores.append(0.0)
    return scores


def analyze_sentiment(headlines_with_scores: list) -> dict:
    if not headlines_with_scores:
        return {"prediction": "Neutral ⚖️", "confidence": 0, "article_count": 0, "data_quality": "insufficient"}

    texts   = [h[0] for h in headlines_with_scores]
    weights = [max(h[1], 0.5) for h in headlines_with_scores]

    if USE_VADER:
        scores = _scores_via_vader(texts)
    else:
        import torch
        _load_finbert()
        inputs = _tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = _model(**inputs)
        probs  = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
        scores = [float(p[2] - p[0]) for p in probs]

    while len(scores) < len(texts):
        scores.append(0.0)

    total_weight = sum(weights)
    avg_score    = sum(s * w for s, w in zip(scores, weights)) / total_weight

    article_count = len(headlines_with_scores)
    data_quality  = "good" if article_count >= 6 else ("moderate" if article_count >= 3 else "low")

    raw_confidence = min(abs(avg_score) * 1.2, 1.0)
    if article_count < 3:
        raw_confidence *= 0.5
        data_quality    = "insufficient"
    elif article_count < 6:
        raw_confidence *= 0.75
    raw_confidence = min(raw_confidence, 0.85)

    if abs(avg_score) < 0.05:
        prediction     = "Neutral ⚖️"
        raw_confidence = min(raw_confidence, 0.35)
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


def get_sentiment(symbol: str) -> dict:
    if not API_KEY:
        return {"prediction": "Neutral ⚖️", "confidence": 0, "article_count": 0, "data_quality": "no_api_key"}

    now = time.time()
    if symbol in _sentiment_cache:
        cached_result, cached_time = _sentiment_cache[symbol]
        if now - cached_time < _CACHE_TTL:
            return cached_result

    try:
        headlines = get_news(symbol)
        result    = analyze_sentiment(headlines)
        print(f"📰 [{symbol}] Sentiment: {result['prediction']} ({result['confidence']}% conf, {result['article_count']} articles, score: {result.get('raw_score','N/A')})")
    except Exception as exc:
        print(f"⚠️ Sentiment error for {symbol}: {exc}")
        result = {"prediction": "Neutral ⚖️", "confidence": 0, "article_count": 0, "data_quality": "error"}

    _sentiment_cache[symbol] = (result, now)
    return result