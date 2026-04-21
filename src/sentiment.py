"""
sentiment.py — FinBERT-based news sentiment analysis

Deployment fix:
  - Added USE_HF_INFERENCE_API env var (set to "true" on Render).
    When true, FinBERT runs on HuggingFace's servers via their free
    Inference API — no local model loaded, zero RAM usage on your server.
    Free tier: 1 000 requests/day (plenty for a demo project).
    Cold start: ~10–20 s on very first call (HF warms up the model).
  - When false (local development) the existing lazy-load behaviour is kept.
"""

import requests
import numpy as np
from dotenv import load_dotenv
import os
import time

load_dotenv()

# ── PRODUCTION vs LOCAL flag ──────────────────────────────────
# Set USE_HF_INFERENCE_API=true in your Render environment variables.
# Leave it unset / false for local development.
USE_INFERENCE_API = os.getenv("USE_HF_INFERENCE_API", "false").lower() == "true"
HF_INFERENCE_URL  = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

# ── LAZY LOAD: model is None until first call (local dev only) ─
_tokenizer = None
_model     = None


def _load_model():
    """Load FinBERT locally (only used when USE_INFERENCE_API is false)."""
    global _tokenizer, _model

    if _tokenizer is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        MODEL_NAME = "ProsusAI/finbert"
        HF_TOKEN   = os.getenv("HF_TOKEN")

        print("⏳ Loading FinBERT model locally…")

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        _model     = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, token=HF_TOKEN
        )

        print("✅ FinBERT loaded locally.")


# ── CACHE: symbol → (result_dict, unix_timestamp) ────────────
_sentiment_cache: dict = {}
_CACHE_TTL = 600   # 10 minutes


# ─────────────────────────────────────────────
# HF INFERENCE API  (production path)
# ─────────────────────────────────────────────

def _scores_via_api(texts: list) -> list:
    """
    Call HuggingFace Inference API for FinBERT sentiment scoring.
    Returns a list of float scores (positive_prob − negative_prob) per text.

    HF response format:
      [[{"label": "positive", "score": 0.9}, {"label": "negative", ...}, ...], ...]
    """
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    print("HF TOKEN (first 10 chars):", HF_TOKEN[:10])
    headers  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    scores = []

    for text in texts:
        try:
            resp = requests.post(
                HF_INFERENCE_URL,
                headers = headers,
                json    = {"inputs": text[:512]},
                timeout = 20,
            )

            if resp.status_code == 200:
                data = resp.json()

                # Normalise to flat list of {label, score} dicts
                if isinstance(data, list) and data:
                    items = data[0] if isinstance(data[0], list) else data
                else:
                    scores.append(0.0)
                    continue

                label_map = {
                    item["label"].lower(): item["score"]
                    for item in items
                    if isinstance(item, dict) and "label" in item
                }

                # FinBERT labels on HF: "positive", "negative", "neutral"
                score = label_map.get("positive", 0.0) - label_map.get("negative", 0.0)
                scores.append(float(score))

            elif resp.status_code == 503:
                # Model still loading on HF's side (cold start) — return neutral.
                # The in-memory _sentiment_cache means the next call within 10 min
                # skips this path entirely; no repeated cold starts per symbol.
                print("⏳ HF Inference API cold-starting, returning neutral for now.")
                scores.append(0.0)

            else:
                print(f"⚠️ HF API returned {resp.status_code}")
                scores.append(0.0)

        except Exception as exc:
            print(f"⚠️ HF Inference API error: {exc}")
            scores.append(0.0)

    return scores


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
            "target price", "analyst", "results", "beat", "miss",
        ]
        JUNK_KEYWORDS = ["rumor", "tweet", "reddit", "viral", "meme", "gossip"]

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


# ─────────────────────────────────────────────
# SENTIMENT ANALYSIS
# ─────────────────────────────────────────────

def analyze_sentiment(headlines_with_scores: list) -> dict:
    """
    Run FinBERT on a list of (text, quality_score) tuples.
    Automatically uses HF Inference API or local model based on env var.
    """
    if not headlines_with_scores:
        return {
            "prediction":    "Neutral ⚖️",
            "confidence":    0,
            "article_count": 0,
            "data_quality":  "insufficient",
        }

    texts   = [h[0] for h in headlines_with_scores]
    weights = [max(h[1], 0.5) for h in headlines_with_scores]

    # ── Score texts via API or local model ──────────────────────
    if USE_INFERENCE_API:
        scores = _scores_via_api(texts)
    else:
        # Local path (development)
        import torch
        _load_model()
        inputs = _tokenizer(
            texts,
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = 512,
        )
        with torch.no_grad():
            outputs = _model(**inputs)
        probs  = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
        # FinBERT labels locally: 0=negative, 1=neutral, 2=positive
        scores = [float(p[2] - p[0]) for p in probs]

    # ── Guard: if API returned fewer scores than texts, pad with 0 ──
    while len(scores) < len(texts):
        scores.append(0.0)

    # ── Weighted average ─────────────────────────────────────────
    total_weight = sum(weights)
    avg_score    = sum(s * w for s, w in zip(scores, weights)) / total_weight

    article_count = len(headlines_with_scores)
    data_quality  = (
        "good"     if article_count >= 6 else
        "moderate" if article_count >= 3 else
        "low"
    )

    # ── Confidence calculation ────────────────────────────────────
    raw_confidence = min(abs(avg_score) * 1.2, 1.0)

    if article_count < 3:
        raw_confidence *= 0.5
        data_quality    = "insufficient"
    elif article_count < 6:
        raw_confidence *= 0.75

    raw_confidence = min(raw_confidence, 0.85)

    # ── Direction ─────────────────────────────────────────────────
    if abs(avg_score) < 0.08:
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
            "prediction":    "Neutral ⚖️",
            "confidence":    0,
            "article_count": 0,
            "data_quality":  "no_api_key",
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