import redis
import os

def get_redis():
    return redis.Redis.from_url(
        os.getenv("REDIS_URL"),
        decode_responses=True
    )