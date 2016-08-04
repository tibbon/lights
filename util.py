import redis
import os

def redis_conn():
    host = os.environ.get('REDISHOST', '192.168.1.101')
    return redis.Redis(host=host)
