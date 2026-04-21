import redis
import os


def get_redis():
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # If using TLS (Render / Upstash)
    if url.startswith("rediss://"):
        return redis.Redis.from_url(
            url,
            decode_responses=True,
            ssl_cert_reqs=None
        )

    # Local Redis (no TLS)
    return redis.Redis.from_url(
        url,
        decode_responses=True
    )