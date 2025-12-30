# test_redis.py
from redis_client import get_stm

# redis_client.set("test_key", "hello")
# print(redis_client.get("test_key"))
session_id = "121"
print(get_stm(session_id))
