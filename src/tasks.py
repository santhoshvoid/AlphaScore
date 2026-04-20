"""
tasks.py — Celery tasks for AlphaCross
Located at: src/tasks.py

Tasks:
  save_to_db_task       — save signals/trades/metrics after analysis
  save_price_task       — save OHLCV price data to DB
  refresh_recent_stocks — scheduled: refresh price data every 15 min (market hours)
  check_ema_alerts      — scheduled: check EMA convergence + send email alerts

To run the worker:
  celery -A celery_app worker --loglevel=info

To run the beat scheduler (needed for scheduled tasks):
  celery -A celery_app beat --loglevel=info
"""

from celery_app import celery

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime


# ─────────────────────────────────────────────
# DB SAVE TASK
# ─────────────────────────────────────────────

@celery.task(name="src.tasks.save_to_db_task")
def save_to_db_task(symbol, sig_copy, trd_copy, met_db, short_ema, long_ema):
    """
    Save signals, trades, and metrics to PostgreSQL.
    Note: sig_copy contains (signal_str, date_str) tuples — already serialized.
    The DB functions handle string dates just fine via psycopg2.
    """
    from src.db_loader import (
        save_stock, save_signals, save_trades, save_metrics,
        load_metrics, signals_exist, trades_exist,
    )

    try:
        print(f"🔥 [CELERY] Saving {symbol}…")

        save_stock(symbol)

        if not signals_exist(symbol, short_ema, long_ema):
            save_signals(symbol, sig_copy, short_ema, long_ema)
        else:
            print(f"⚡ [{symbol}] Signals already exist, skipping.")

        if not trades_exist(symbol):
            save_trades(symbol, trd_copy)
        else:
            print(f"⚡ [{symbol}] Trades already exist, skipping.")

        if not load_metrics(symbol, short_ema, long_ema):
            save_metrics(symbol, met_db, short_ema, long_ema)
        else:
            print(f"⚡ [{symbol}] Metrics already exist, skipping.")

        print(f"✅ [CELERY] DB save complete for {symbol}")

    except Exception as e:
        print(f"❌ [CELERY] save_to_db_task error: {e}")


# ─────────────────────────────────────────────
# PRICE SAVE TASK
# ─────────────────────────────────────────────

@celery.task(name="src.tasks.save_price_task")
def save_price_task(symbol, data_dict):
    """
    Save OHLCV price data to PostgreSQL.
    data_dict is a {column: {index: value}} dict with string indexes.
    """
    import pandas as pd
    from src.data_fetch import store_price_data

    try:
        df = pd.DataFrame(data_dict)
        df.index = pd.to_datetime(df.index)
        print(f"📦 [CELERY] Saving price data for {symbol} ({len(df)} rows)…")
        store_price_data(symbol, df)
        print(f"✅ [CELERY] Price data saved for {symbol}")

    except Exception as e:
        print(f"❌ [CELERY] save_price_task error: {e}")


# ─────────────────────────────────────────────
# SCHEDULED: 15-MIN MARKET HOURS REFRESH
# ─────────────────────────────────────────────

@celery.task(name="src.tasks.refresh_recent_stocks")
def refresh_recent_stocks():
    """
    Scheduled task: runs every 15 minutes during IST market hours.
    Refreshes price data for recently analysed stocks.
    This keeps the DB current without any user action.
    Beat schedule configured in celery_app.py.
    """
    from src.db_loader import get_watchlist
    from src.data_fetch import get_stock_data, store_price_data

    now_utc = datetime.utcnow()
    # IST = UTC+5:30  |  Market: 9:15–15:30 IST = 3:45–10:00 UTC
    # We use a slightly wider window (3:00–10:30 UTC) as buffer
    if not (3 <= now_utc.hour <= 10):
        print("⏰ [CELERY] Outside market hours, skipping refresh.")
        return

    symbols = get_watchlist(limit=20)   # top 20 recently analysed
    if not symbols:
        print("⚠️ [CELERY] No symbols in watchlist to refresh.")
        return

    print(f"🔄 [CELERY] Refreshing {len(symbols)} stocks…")

    for item in symbols:
        symbol = item["symbol"]
        try:
            data = get_stock_data(symbol, "1y")
            if data is not None and not data.empty:
                store_price_data(symbol, data)
                print(f"  ✅ Refreshed {symbol}")
            else:
                print(f"  ⚠️ No data for {symbol}")
        except Exception as e:
            print(f"  ❌ Refresh failed for {symbol}: {e}")

    print("✅ [CELERY] Market refresh complete.")


# ─────────────────────────────────────────────
# SCHEDULED: EMA ALERT CHECK
# ─────────────────────────────────────────────

@celery.task(name="src.tasks.check_ema_alerts")
def check_ema_alerts():
    """
    Scheduled task: runs every 15 minutes during market hours.
    Checks if any monitored stock's EMAs are converging.
    If |EMA_short - EMA_long| / price < 1% → send Stage 1 (warning) email.
    If actual crossover just happened → send Stage 2 (confirmation) email.
    """
    from src.db_loader import get_all_active_alerts
    from src.data_fetch import get_stock_data
    from src.indicators import calculate_ema

    now_utc = datetime.utcnow()
    if not (3 <= now_utc.hour <= 10):
        return   # outside market hours

    alerts = get_all_active_alerts()
    if not alerts:
        return

    print(f"🔔 [CELERY] Checking {len(alerts)} alerts…")

    processed = set()

    for alert in alerts:
        email     = alert["email"]
        symbol    = alert["symbol"]
        short_ema = alert["short_ema"]
        long_ema  = alert["long_ema"]

        # Only process each symbol+EMA combo once even if multiple emails
        combo = f"{symbol}_{short_ema}_{long_ema}"
        if combo in processed:
            continue
        processed.add(combo)

        try:
            data = get_stock_data(symbol, "3mo")
            if data is None or data.empty or len(data) < long_ema + 5:
                continue

            short_col = f"EMA{short_ema}"
            long_col  = f"EMA{long_ema}"
            data[short_col] = calculate_ema(data["Close"], short_ema)
            data[long_col]  = calculate_ema(data["Close"], long_ema)

            latest   = data.iloc[-1]
            prev     = data.iloc[-2]
            price    = float(latest["Close"])
            s_now    = float(latest[short_col])
            l_now    = float(latest[long_col])
            s_prev   = float(prev[short_col])
            l_prev   = float(prev[long_col])

            spread_pct = abs(s_now - l_now) / price

            # ── Crossover detection ──────────────────────────────
            crossed_up   = (s_prev <= l_prev) and (s_now > l_now)   # Golden Cross
            crossed_down = (s_prev >= l_prev) and (s_now < l_now)   # Death Cross

            # ── Convergence check ────────────────────────────────
            converging = spread_pct < 0.01   # EMAs within 1% of price

            # Re-query all emails for this alert
            alert_emails = [a["email"] for a in alerts
                            if a["symbol"] == symbol
                            and a["short_ema"] == short_ema
                            and a["long_ema"] == long_ema]

            if crossed_up:
                subject = f"🟢 AlphaCross: Golden Cross on {symbol}!"
                body    = (
                    f"Hello,\n\n"
                    f"A Golden Cross just occurred on {symbol}.\n\n"
                    f"EMA {short_ema} ({s_now:.2f}) crossed ABOVE EMA {long_ema} ({l_now:.2f}).\n"
                    f"Current Price: ₹{price:.2f}\n\n"
                    f"This is a BULLISH signal.\n\n"
                    f"— AlphaCross\n"
                    f"⚠️ Not financial advice. For educational purposes only."
                )
                for em in alert_emails:
                    _send_email(em, subject, body)
                print(f"  📬 Golden Cross alert sent for {symbol}")

            elif crossed_down:
                subject = f"🔴 AlphaCross: Death Cross on {symbol}!"
                body    = (
                    f"Hello,\n\n"
                    f"A Death Cross just occurred on {symbol}.\n\n"
                    f"EMA {short_ema} ({s_now:.2f}) crossed BELOW EMA {long_ema} ({l_now:.2f}).\n"
                    f"Current Price: ₹{price:.2f}\n\n"
                    f"This is a BEARISH signal.\n\n"
                    f"— AlphaCross\n"
                    f"⚠️ Not financial advice. For educational purposes only."
                )
                for em in alert_emails:
                    _send_email(em, subject, body)
                print(f"  📬 Death Cross alert sent for {symbol}")

            elif converging:
                subject = f"⚠️ AlphaCross: EMA Convergence on {symbol}"
                body    = (
                    f"Hello,\n\n"
                    f"The EMAs on {symbol} are converging — a crossover may be coming soon!\n\n"
                    f"EMA {short_ema}: {s_now:.2f}\n"
                    f"EMA {long_ema}:  {l_now:.2f}\n"
                    f"Spread: {spread_pct*100:.2f}% of price\n"
                    f"Current Price: ₹{price:.2f}\n\n"
                    f"Keep watching this one.\n\n"
                    f"— AlphaCross\n"
                    f"⚠️ Not financial advice. For educational purposes only."
                )
                for em in alert_emails:
                    _send_email(em, subject, body)
                print(f"  📬 Convergence alert sent for {symbol}")

        except Exception as e:
            print(f"  ❌ Alert check failed for {symbol}: {e}")

    print("✅ [CELERY] Alert check complete.")


# ─────────────────────────────────────────────
# EMAIL SENDER
# ─────────────────────────────────────────────

def _send_email(to_email: str, subject: str, body: str):
    """
    Send an email using Gmail SMTP.
    Requires SMTP_EMAIL and SMTP_PASSWORD in .env
    (use a Gmail App Password, NOT your real password).
    """
    smtp_email    = os.getenv("SMTP_EMAIL", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")

    if not smtp_email or not smtp_password:
        print(f"⚠️ Email not configured — skipping send to {to_email}")
        return

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = smtp_email
        msg["To"]      = to_email
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, to_email, msg.as_string())

        print(f"  ✉️ Email sent to {to_email}")

    except Exception as e:
        print(f"  ❌ Email send failed to {to_email}: {e}")