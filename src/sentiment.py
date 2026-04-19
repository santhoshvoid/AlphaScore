import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
sentiment_cache = {}
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# Load FinBERT once (important)
MODEL_NAME = "ProsusAI/finbert"
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=HF_TOKEN)

API_KEY = os.getenv("NEWS_API_KEY")
if not API_KEY:
    raise ValueError("NEWS_API_KEY not found in .env")


def get_news(symbol):
    clean_symbol = symbol.replace(".NS", "")

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={clean_symbol} stock OR {clean_symbol} company OR {clean_symbol} earnings"
        f"&language=en&sortBy=publishedAt&pageSize=10&apiKey={API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return []

        data = response.json()
        articles = data.get("articles", [])

        headlines = []

        for a in articles:
            title = a.get("title")
            desc = a.get("description")

            if not title:
                continue

            text = title
            if desc:
                text += ". " + desc

            # filter weak junk headlines
            if len(title) < 20:
                continue

            # filter low-quality / irrelevant news
            bad_keywords = ["rumor", "tweet", "reddit", "viral", "meme"]
            if any(word in title.lower() for word in bad_keywords):
                continue

            
            important_keywords = ["earnings", "revenue", "profit", "guidance", "deal", "order"]

            score = sum(1 for w in important_keywords if w in text.lower())

            # keep only strong news OR enough articles
            if score == 0 and len(headlines) > 5:
                continue

            headlines.append(text)

        return headlines

    except Exception:
        return []


def analyze_sentiment(headlines):
    if not headlines:
        return {
            "prediction": "Neutral 😐",
            "confidence": 0
        }

    inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()

    scores = [p[2] - p[0] for p in probs]

    avg_score = np.mean(scores)

    confidence = min(abs(avg_score) * 1.2, 1)

    if avg_score > 0.1:
        pred = "Bullish 📈"
    elif avg_score < -0.1:
        pred = "Bearish 📉"
    else:
        pred = "Neutral ⚖️"

    return {
        "prediction": pred,
        "confidence": round(confidence * 100, 2)
    }


import time

def get_sentiment(symbol):
    now = time.time()

    if symbol in sentiment_cache:
        data, timestamp = sentiment_cache[symbol]

        if now - timestamp < 600:  # 10 min cache
            return data

    headlines = get_news(symbol)
    result = analyze_sentiment(headlines)

    sentiment_cache[symbol] = (result, now)

    return result