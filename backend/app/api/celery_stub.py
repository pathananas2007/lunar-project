"""
Celery + Redis async stub for NFR-03 (10k batch). Designed but not wired for single-pair MVP.
To enable: pip install celery[redis] && set REDIS_URL, then replace in-memory TASKS with Celery tasks.

Architecture:
- Broker: Redis (localhost:6379)
- Queue: lunar_align
- Task: run_pipeline -> stores result in Redis hash with TTL 24h
- Frontend polls GET /api/v1/status/{task_id} (Celery AsyncResult)

This stub documents the intended production wiring without adding heavy deps to MVP.
"""
from typing import Dict

# MVP in-memory store lives in tasks.py
# Production would be:
#
# from celery import Celery
# celery = Celery("lunar", broker="redis://redis:6379/0", backend="redis://redis:6379/0")
#
# @celery.task(bind=True, max_retries=2)
# def align_task(self, img1_path, img2_path, params):
#     img1, meta1 = PDS4DataIngestor().read(img1_path)
#     img2, meta2 = PDS4DataIngestor().read(img2_path)
#     result = run_pipeline(img1, img2, **params)
#     return result
#
# Endpoint POST /api/v1/align would then:
#   task = align_task.delay(...)
#   return {"task_id": task.id, "status": "accepted"}
#
# Endpoint GET /api/v1/status/{id}:
#   res = celery.AsyncResult(task_id)
#   if res.ready(): return {"status": "completed", **res.result}
#   else: return {"status": res.status}

CELERY_DESIGN = {
    "broker": "redis://redis:6379/0",
    "backend": "redis://redis:6379/0",
    "queue": "lunar_align",
    "concurrency": 4,
    "task_timeout": 300,
    "result_ttl": 86400,
    "note": "Single-pair MVP uses in-memory TASKS for zero-infra demo; swap to Celery for 10k batch NFR-03",
}
