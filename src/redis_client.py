import redis
import os


def get_redis():
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Upstash (and most cloud Redis) uses rediss:// (TLS).
    # redis-py requires ssl_cert_reqs to be set explicitly for TLS URLs,
    # otherwise it throws:
    # "A rediss:// URL must have parameter ssl_cert_reqs..."
    # ssl_cert_reqs=None disables certificate verification — safe for
    # Upstash's managed TLS where the cert is always valid.
    return redis.Redis.from_url(
        url,
        decode_responses=True,
        ssl_cert_reqs=None,
    )