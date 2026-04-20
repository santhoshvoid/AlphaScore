"""
celery_app.py — Celery application + beat schedule for AlphaCross
Located at: project root (same level as app.py)

To run the worker:
  celery -A celery_app worker --loglevel=info

To run the beat scheduler (needed for scheduled tasks):
  celery -A celery_app beat --loglevel=info

To run both worker + beat together (development only):
  celery -A celery_app worker --beat --loglevel=info
"""

from celery import Celery
from celery.schedules import crontab

celery = Celery(
    "alpha_cross",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
    include=["src.tasks"]
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
)

# ─────────────────────────────────────────────
# BEAT SCHEDULE — runs automatically every 15 min during market hours
# ─────────────────────────────────────────────
# IST 9:15–15:30 = UTC 3:45–10:00
# crontab uses UTC since enable_utc=True

celery.conf.beat_schedule = {

    # Refresh price data for recently analysed stocks
    # Every 15 min, Mon–Fri, 3:45–10:00 UTC (9:15–15:30 IST)
    "refresh-stocks-15min": {
        "task":     "src.tasks.refresh_recent_stocks",
        "schedule": crontab(
            minute  = "*/15",
            hour    = "3-10",
            day_of_week = "1-5",   # Mon–Fri
        ),
    },

    # Check EMA convergence / crossovers and send email alerts
    # Same schedule
    "check-ema-alerts-15min": {
        "task":     "src.tasks.check_ema_alerts",
        "schedule": crontab(
            minute  = "*/15",
            hour    = "3-10",
            day_of_week = "1-5",
        ),
    },
}