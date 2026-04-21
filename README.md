# AlphaCross — Stock Intelligence & Strategy Platform

> EMA Crossover Backtesting · XGBoost ML Prediction · NLP Sentiment Fusion · Real-Time Alerts  
> Built for Indian equity markets (NSE) · Deployed on Render · Mobile-responsive interface

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)](https://flask.palletsprojects.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)](https://xgboost.ai)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supabase-blue?logo=postgresql)](https://supabase.com)
[![Redis](https://img.shields.io/badge/Redis-Cache-red?logo=redis)](https://redis.io)
[![Celery](https://img.shields.io/badge/Celery-Async-brightgreen)](https://docs.celeryq.dev)
[![Deployed on Render](https://img.shields.io/badge/Deployed-Render-46E3B7?logo=render)](https://render.com)

---

## 🔗 Live Demo

**[https://alphascore-1.onrender.com](https://alphascore-1.onrender.com/)**

> ⚠️ **Cold Start Notice:** The demo runs on Render's free tier, which spins down after inactivity. First request may take 15–30 seconds to wake up. After that it's fast — Redis keeps things warm.

---

## ⚡ TL;DR

Full-stack stock analysis system combining:
- EMA-based backtesting engine
- XGBoost ML prediction
- NLP sentiment (FinBERT/VADER)
- Redis caching + Celery async processing

→ Outputs explainable Buy/Sell signals with confidence and performance metrics

---

## The Problem — Why This Exists

If you've spent any time around stock trading, you've probably heard of EMA crossovers. The Golden Cross, the Death Cross — these aren't just fancy names. They're among the most widely-used technical signals in the world, used by retail traders, hedge funds, and algorithmic systems alike.

But here's what bothered me: **there's no single place where a trader can actually test whether this strategy works for a specific stock, in a specific market, over a specific time period — and understand *why* the signal is firing.**

Most charting tools just draw the lines. They don't tell you:
- Has this EMA pair actually made money on this stock historically?
- What's the win rate? What's the drawdown? Is the current signal trustworthy?
- What does the market sentiment look like right now? Is the broader context supporting this trade?
- And critically — is the current market trending or ranging? Because a strategy that crushes it in a trending market can bleed you dry in a sideways one.

That gap is what AlphaCross is trying to close. It's not trying to replace a broker or tell you what to buy. It's trying to give traders — especially retail traders in India who don't have access to expensive platforms — a proper, explainable framework for evaluating an EMA-based strategy on any NSE-listed stock.

---

## What AlphaCross Actually Does

Given a stock ticker and a time range, the platform:

1. Detects EMA crossover signals (Golden Cross / Death Cross) with configurable periods
2. Backtests the strategy end-to-end — simulating real buys and sells with capital tracking
3. Runs an XGBoost model over engineered features to predict near-term price direction
4. Scrapes and scores recent financial news via NLP sentiment analysis
5. Fuses all three signals with adaptive weighting based on the detected market regime (trending vs sideways)
6. Outputs a final Bullish / Bearish / Neutral prediction with a confidence score and a full breakdown of how each component contributed

The last part is the bit I'm most proud of. You don't just get a prediction — you get *why*. Every component is shown: what the EMA says, what the ML model thinks, what the news sentiment looks like, and how they were weighted together given the current market conditions.

---

## Features

### Technical Analysis & Backtesting

- Custom EMA crossover detection with configurable short and long periods (9/21, 20/50, 50/200, etc.)
- Golden Cross / Death Cross signals plotted on an interactive price chart
- Full backtesting engine — simulates buy-at-GC / sell-at-DC from historical OHLCV data
- Performance metrics: Total Return %, Win Rate, Trade Count, CAGR, Max Drawdown, Sharpe Ratio, Best/Worst Trade
- Portfolio growth simulation — tracks compound capital across every trade
- Per-signal confidence scoring with EMA strength and 1D/3D/1W/1M outcome analysis
- Strategy comparison — benchmark up to 4 EMA pairs side-by-side (9/21, 12/26, 20/50, 50/200)
- Market bias detection (Bullish / Bearish / Neutral) and a composite Health Score

### ML Prediction Layer

- XGBoost binary classifier trained on engineered features: EMA spreads, volatility-normalised momentum, volume ratios, price-relative EMAs
- Market regime classification (TRENDING vs SIDEWAYS) based on EMA spread percentage
- Volatility-adjusted confidence — penalised in high-volatility or low-conviction environments
- Trend alignment bonus — confidence boosted when EMA direction, momentum, and model all agree
- Lazy model loading — loaded on first inference, not at startup, to avoid double-load in Flask debug mode

### NLP Sentiment Analysis

- NewsAPI integration — fetches up to 15 recent articles per symbol
- Financial keyword filtering to strip out noise before scoring
- Quality-weighted sentiment — articles with more finance-relevant keywords carry higher weight
- Dual-mode engine: FinBERT transformer for local dev, VADER for production (see Design Decisions)

### 3-Signal Adaptive Fusion Engine

- Combines ML prediction, EMA trend score, and news sentiment into one final signal
- Adaptive weighting based on regime:
  - Trending market → ML 40%, EMA 40%, Sentiment 20%
  - Sideways market → ML 60%, EMA 20%, Sentiment 20%
- Disagreement penalty applied when signals conflict
- Full explainability panel — each component's label, score, and weight contribution is shown

### Data & Storage

- yfinance for live OHLCV data with multi-index flattening and max-period fallback
- PostgreSQL (Supabase) for persistent storage: prices, stocks, signals, trades, metrics, alerts
- Redis caching with 5-minute TTL on full analysis results — cache-first on every request
- DB-first loading — always checks the database before hitting yfinance externally
- Idempotent batch inserts via psycopg2 with `ON CONFLICT DO NOTHING`

### Async Background Processing (Celery)

- `save_to_db_task` — async PostgreSQL write for signals/trades/metrics after each analysis
- `save_price_task` — async OHLCV storage after a live yfinance fetch
- `refresh_recent_stocks` — scheduled every 15 min during IST market hours (9:15–15:30)
- `check_ema_alerts` — scheduled every 15 min to evaluate all active alert subscriptions
- Graceful fallback to a daemon thread if Celery broker is unreachable — response is never blocked

### Email Alert System

- Subscribe to alerts for any stock + EMA pair combination
- Three alert types:
  - 🟢 Golden Cross confirmation — short EMA crosses above long
  - 🔴 Death Cross confirmation — short EMA crosses below long
  - ⚠️ Convergence warning — EMAs within 1% of each other (crossover imminent)
- Gmail SMTP delivery via App Password
- Alerts are soft-deleted (deactivated) rather than permanently removed

### Frontend & UX

- Fully mobile-responsive — works on phones without any horizontal scroll or zoom
- Interactive price chart with EMA overlays and BUY/SELL markers (Chart.js)
- Portfolio growth chart visualising compound capital across the backtest period
- Fuzzy NSE symbol resolver — supports company names, partial names, and symbols across 1,800+ listed stocks using word-score and difflib matching
- Cookie-based watchlist — remembers your last 8 searches, no login required
- CSV export for the full signal table and trade history

### Testing

Unit test suite using pytest:
- `test_signals.py` — crossover detection correctness
- `test_backtest.py` — backtest returns valid final capital
- `test_indicators.py` — EMA output length matches input
- `test_api.py` — Flask home route returns HTTP 200
- `conftest.py` handles path injection so tests run from the `tests/` subfolder

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask 3.1, Gunicorn |
| Frontend | HTML5, CSS3, JavaScript, Chart.js |
| Machine Learning | XGBoost, scikit-learn, joblib |
| NLP / Sentiment | FinBERT (HuggingFace) · VADER (production) |
| Market Data | yfinance, NewsAPI |
| Database | PostgreSQL (Supabase), psycopg2 |
| Caching | Redis (Upstash cloud / Docker local) |
| Task Queue | Celery, Celery Beat |
| Email | Gmail SMTP (App Password) |
| Deployment | Render |
| Testing | pytest |

---

## Project Structure

```
alphacross/
├── app.py                      # Flask app — routes, fusion engine, Redis logic
├── celery_app.py               # Celery config + beat schedule
├── Procfile                    # Render/Heroku process definition
├── runtime.txt                 # Python version pin
├── requirements.txt
├── dataset.csv                 # Training dataset for XGBoost model
├── env.example                 # Template for environment variables
├── .gitignore
│
├── data/
│   └── nse_stocks.csv          # NSE stock name → symbol mapping (~1,800 stocks)
│
├── models/
│   └── xgb_model.pkl           # Trained XGBoost model
│
├── src/
│   ├── advanced_metrics.py     # CAGR, Sharpe, Max Drawdown, portfolio growth
│   ├── analysis.py             # Signal outcome analysis & confidence scoring
│   ├── backtest.py             # Backtesting engine
│   ├── data_fetch.py           # yfinance wrapper + PostgreSQL price storage
│   ├── db.py                   # DB connection helper
│   ├── db_loader.py            # All PostgreSQL read/write helpers
│   ├── fetch_nse_data.py       # NSE data fetching utilities
│   ├── indicators.py           # EMA calculation (Series + list input)
│   ├── main.py                 # Entry point / CLI runner
│   ├── ml_dataset.py           # Feature engineering for model training
│   ├── ml_model.py             # XGBoost inference engine (lazy loading)
│   ├── redis_client.py         # Redis connection (local + TLS/Upstash)
│   ├── sentiment.py            # NewsAPI fetch + FinBERT/VADER scoring
│   ├── signals.py              # EMA crossover detection
│   ├── symbol_resolver.py      # Fuzzy NSE symbol lookup
│   ├── tasks.py                # Celery async tasks
│   └── visualization.py        # Matplotlib chart (standalone CLI)
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html              # Main UI (Jinja2 template)
│
└── tests/
    ├── conftest.py
    ├── test_api.py
    ├── test_backtest.py
    ├── test_indicators.py
    └── test_signals.py
```

---

## Why This Project Is Different

There are plenty of charting tools. There are also plenty of ML stock prediction demos floating around GitHub that train a model on closing prices and call it done. AlphaCross isn't trying to be either of those.

A few things that I think make it actually interesting:

**It's explainable by design.** The fusion engine doesn't just output a direction — it tells you exactly why. The ML score, the EMA trend score, and the sentiment score are all visible, along with their weights. A trader can look at the breakdown and decide for themselves whether they trust the signal.

**It adapts to the market regime.** This is the part that took the most thought. An EMA signal in a strongly trending market is much more reliable than the same signal in a choppy, sideways market. The fusion engine detects the regime first and adjusts component weights accordingly — leaning more on the ML model when the market is sideways and can't be trusted from a trend-following perspective.

**It actually backtests the strategy, not just the model.** A lot of ML finance projects train a model and report accuracy. What AlphaCross does is simulate actual trades based on EMA signals, track real P&L, compute proper financial metrics (Sharpe, max drawdown, CAGR), and let you see whether the strategy would have worked historically on that specific stock. That's the difference between academic output and something a trader might actually use.

**It's built as a system, not just a script.** Redis caching, async Celery tasks, PostgreSQL persistence, email alerts — these aren't there to show off. They're there because a tool that freezes every time you run it, or that loses all its data on restart, isn't useful. Every architectural decision was made with real usage in mind.

---

## System Design — How It's Built

This project was built with a few core principles in mind, and I think they're worth explaining because they affected a lot of the decisions.

**Separation of concerns throughout.** Each component in `src/` does exactly one thing. The backtest engine doesn't touch the database. The ML model doesn't care about the frontend. The sentiment module doesn't know about EMA signals. This made it much easier to iterate on individual parts without breaking everything else, and it's why the test suite can cover each module independently.

**Cache-first, external-last.** Every analysis request checks Redis first, then the database, and only hits yfinance or NewsAPI if neither has the data. This keeps the app fast and reduces dependency on external APIs that have rate limits or can be unreliable. The 5-minute TTL on Redis keeps data fresh during active sessions without hammering external sources.

**Async where it matters.** The Celery workers handle things that don't need to block the user — writing to PostgreSQL, refreshing data during market hours, checking alerts. The app responds immediately while the background work happens in parallel. And if Celery isn't available, it falls back to a daemon thread so the core functionality never breaks.

**Lazy loading for heavy resources.** The XGBoost model and the sentiment pipeline are loaded on first use, not at startup. In Flask's debug mode, the app can reload twice — loading a 1GB transformer model on startup would mean loading it twice. Lazy loading avoids this entirely and keeps cold start times acceptable.

**Practical deployment constraints shaped real decisions.** FinBERT is more accurate than VADER for financial text, but it's also nearly 1GB and inference is slow on CPU. For a free-tier deployment with limited RAM, VADER is the right tradeoff. The codebase supports both with a single environment variable switch — so if and when the infrastructure scales, upgrading is trivial.

---

## What's Next

A few things I'd like to add when there's time:

- **Live trading integration** — connecting to broker APIs (Zerodha Kite, Angel One SmartAPI) to move from analysis to actual execution
- **LSTM / Transformer models** — the XGBoost model works well on tabular features, but sequence models might capture temporal patterns better
- **Portfolio-level optimisation** — right now everything is per-stock; building a multi-stock allocation engine on top would make it genuinely useful for portfolio management
- **User accounts and dashboards** — cookie-based watchlists work, but persistent user accounts would enable better alert management and historical tracking
- **Real-time streaming** — WebSocket-based live price updates so the analysis refreshes without page reloads during market hours

---

## Local Setup

### Prerequisites

- Python 3.10+
- PostgreSQL database (or a free [Supabase](https://supabase.com) project)
- Redis — easiest via Docker: `docker run -d -p 6379:6379 redis`
- A [NewsAPI](https://newsapi.org) key (free tier, 100 req/day)
- A Gmail account with an [App Password](https://support.google.com/accounts/answer/185833) enabled (for email alerts)

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/alphacross.git
cd alphacross
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional: for local FinBERT sentiment (heavy, ~1 GB)
pip install transformers torch
```

### 2. Environment Variables

Copy `env.example` to `.env` and fill in your values:

```env
# Database
DATABASE_URL=postgresql://user:password@host:5432/dbname

# Redis
REDIS_URL=redis://localhost:6379/0

# NewsAPI
NEWS_API_KEY=your_newsapi_key_here

# Email Alerts (Gmail SMTP)
SMTP_EMAIL=youremail@gmail.com
SMTP_PASSWORD=your_gmail_app_password

# Sentiment Mode
USE_VADER=true
# Set to false to use local FinBERT (requires transformers + torch)
```

### 3. Initialise the Database

Run this SQL in your PostgreSQL instance (or Supabase SQL editor):

```sql
CREATE TABLE prices (
    symbol TEXT, date DATE, open FLOAT, high FLOAT,
    low FLOAT, close FLOAT, volume BIGINT,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE stocks (
    symbol TEXT PRIMARY KEY,
    last_updated TIMESTAMP
);

CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    symbol TEXT, date DATE, signal TEXT,
    ema_short INT, ema_long INT,
    UNIQUE (symbol, date, ema_short, ema_long)
);

CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    symbol TEXT, buy_date DATE, sell_date DATE,
    buy_price FLOAT, sell_price FLOAT, return FLOAT,
    UNIQUE (symbol, buy_date, sell_date)
);

CREATE TABLE metrics (
    symbol TEXT, short_ema INT, long_ema INT,
    total_return FLOAT, win_rate FLOAT, sharpe FLOAT, max_drawdown FLOAT,
    PRIMARY KEY (symbol, short_ema, long_ema)
);

CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    email TEXT, symbol TEXT, short_ema INT, long_ema INT,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (email, symbol, short_ema, long_ema)
);
```

### 4. Run the App

```bash
# Terminal 1 — Flask dev server
python app.py

# Terminal 2 — Celery worker (background DB saves)
celery -A celery_app worker --pool=solo --loglevel=info

# Terminal 3 — Celery Beat (alerts + auto-refresh scheduler)
celery -A celery_app beat --loglevel=info
```

Visit: `http://localhost:5000`

### 5. Run Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_api.py::test_home_route              PASSED
tests/test_backtest.py::test_backtest_runs      PASSED
tests/test_indicators.py::test_ema_output_length PASSED
tests/test_signals.py::test_detect_crossovers   PASSED
```

---

## Environment Variables Reference

| Variable | Required | Description | Where to get |
|---|---|---|---|
| `DATABASE_URL` | ✅ | PostgreSQL connection string | [Supabase](https://supabase.com) (free) |
| `REDIS_URL` | ✅ | Redis connection string | Docker local or [Upstash](https://upstash.com) (free) |
| `NEWS_API_KEY` | ✅ | NewsAPI key for sentiment | [newsapi.org](https://newsapi.org) (free tier) |
| `SMTP_EMAIL` | Optional | Gmail address for alerts | Your Gmail |
| `SMTP_PASSWORD` | Optional | Gmail App Password | Google account settings |
| `USE_VADER` | Optional | `true` = VADER (default), `false` = FinBERT | — |
| `HF_TOKEN` | Optional | HuggingFace token (only if USE_VADER=false) | [huggingface.co](https://huggingface.co) |

---

## Demo vs. Local Deployment

| Feature | Local | Demo (Render Free Tier) |
|---|---|---|
| Sentiment model | FinBERT (transformer, accurate) | VADER (lightweight, offline) |
| Celery workers | Running | Not running (daemon thread fallback) |
| Email alerts | Active (if SMTP configured) | Active (if SMTP configured) |
| Auto market refresh | Every 15 min via Celery Beat | Manual requests only |
| Redis | Local Docker | Upstash (cloud, TLS) |
| PostgreSQL | Local or Supabase | Supabase |
| Cold start | None | ~15–30 seconds after inactivity |

---

## License

MIT — free to use, fork, or build on top of.
