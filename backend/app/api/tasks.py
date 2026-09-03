import uuid
import time
from typing import Dict

# In-memory task store with TTL (ISRO fix: prevents leak on 10k batch)
TASKS: Dict[str, dict] = {}
_TTL_SEC = 3600  # 1h
_MAX_TASKS = 200


def _evict():
    now = time.time()
    # TTL eviction
    for k, v in list(TASKS.items()):
        if now - v.get("ts", 0) > _TTL_SEC:
            TASKS.pop(k, None)
    # Cap size: drop oldest
    if len(TASKS) > _MAX_TASKS:
        oldest = sorted(TASKS.items(), key=lambda x: x[1].get("ts", 0))
        for k, _ in oldest[: len(TASKS) - _MAX_TASKS]:
            TASKS.pop(k, None)


def create_task(result: dict) -> str:
    _evict()
    tid = str(uuid.uuid4())
    # Strip large base64 from stored result to bound memory (keep shape only)
    stored = dict(result)
    if stored.get("warped_b64") and len(stored["warped_b64"]) > 500:
        stored = {**stored, "warped_b64": stored["warped_b64"][:500] + "...[truncated in store]"}
    TASKS[tid] = {"status": "completed", "result": stored, "ts": time.time()}
    return tid


def get_task(tid: str):
    v = TASKS.get(tid)
    if v and time.time() - v.get("ts", 0) > _TTL_SEC:
        TASKS.pop(tid, None)
        return None
    return v
