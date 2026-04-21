"""
celery_app.py — Celery application + beat schedule for AlphaCross
Located at: project root (same level as app.py)

Local development:
  Requires Redis running via Docker:
    docker run -d -p 6379:6379 redis

  Terminal 1 — worker:
    celery -A celery_app worker --pool=solo --loglevel=info

  Terminal 2 — beat scheduler:
    celery -A celery_app beat --loglevel=info

Production (Render / Fly.io):
  REDIS_URL env var points to Upstash (rediss://) — SSL handled automatically.
"""

from celery import Celery
from celery.schedules import crontab
import os

# Fallback to local Redis if REDIS_URL not set (local development)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery = Celery(
    "alpha_cross",
    broker  = REDIS_URL,
    backend = REDIS_URL,
    include = ["src.tasks"],
)

# ── SSL config: only needed for rediss:// (Upstash / cloud Redis) ──────
# Local Redis uses redis:// (no TLS) — applying SSL config to it breaks things.
if REDIS_URL.startswith("rediss://"):
    celery.conf.broker_use_ssl       = {"ssl_cert_reqs": "none"}
    celery.conf.redis_backend_use_ssl = {"ssl_cert_reqs": "none"}

celery.conf.update(
    task_serializer   = "json",
    accept_content    = ["json"],
    result_serializer = "json",
    timezone          = "Asia/Kolkata",
    enable_utc        = True,
)

# ─────────────────────────────────────────────
# BEAT SCHEDULE
# Runs automatically every 15 min during IST market hours.
# IST 9:15–15:30 = UTC 3:45–10:00
# ─────────────────────────────────────────────

celery.conf.beat_schedule = {

    "refresh-stocks-15min": {
        "task":     "src.tasks.refresh_recent_stocks",
        "schedule": crontab(
            minute      = "*/15",
            hour        = "3-10",
            day_of_week = "1-5",
        ),
    },

    "check-ema-alerts-15min": {
        "task":     "src.tasks.check_ema_alerts",
        "schedule": crontab(
            minute      = "*/15",
            hour        = "3-10",
            day_of_week = "1-5",
        ),
    },
}