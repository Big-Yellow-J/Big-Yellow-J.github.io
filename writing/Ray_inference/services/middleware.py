"""HTTP 中间件:request_id 注入 + 并发限流 + inflight 计数 + 慢请求日志 + 请求计数 + 时延直方图。"""
import asyncio
import time
import uuid
from collections import defaultdict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config import MAX_INFLIGHT_REQUESTS, SLOW_REQUEST_MS
from utils.logging_setup import setup_logger

log = setup_logger("api")
_global_semaphore = asyncio.Semaphore(MAX_INFLIGHT_REQUESTS)

# 按 (endpoint_path, status_code) 累计的进程内计数器;/metrics 渲染时读
_request_counter: dict = defaultdict(int)

# 时延直方图桶边界(秒),与 Prometheus 默认桶一致
HISTOGRAM_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

# {path: {"buckets": [count_per_bucket], "sum": float, "count": int}}
_request_hist: dict = defaultdict(
    lambda: {"buckets": [0] * len(HISTOGRAM_BUCKETS), "sum": 0.0, "count": 0}
)

# inflight 计数:优雅退出时等到归零才真正杀 actor
_inflight: int = 0
_inflight_lock = asyncio.Lock()


def request_counter_snapshot() -> dict:
    """返回 {(path, status): count} 的快照。"""
    return dict(_request_counter)


def request_hist_snapshot() -> dict:
    """返回 {path: {buckets, sum, count}} 的快照(浅拷贝足够,渲染只读)。"""
    return {p: dict(v) for p, v in _request_hist.items()}


def _observe_duration(path: str, seconds: float):
    """把一次请求耗时落入对应直方图桶,sum/count 同步累加。"""
    h = _request_hist[path]
    h["count"] += 1
    h["sum"] += seconds
    for i, le in enumerate(HISTOGRAM_BUCKETS):
        if seconds <= le:
            h["buckets"][i] += 1


def inflight_count() -> int:
    """当前正在处理的请求数(优雅退出用)。"""
    return _inflight


class RequestIDMiddleware(BaseHTTPMiddleware):
    """生成/透传 X-Request-ID,记录访问日志(慢请求自动升级为 WARNING),累计请求计数 + 时延直方图。"""

    async def dispatch(self, request: Request, call_next):
        global _inflight
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        t0 = time.time()
        response = None
        async with _inflight_lock:
            _inflight += 1
        try:
            response = await call_next(request)
            return response
        finally:
            async with _inflight_lock:
                _inflight -= 1
            elapsed_sec = time.time() - t0
            elapsed_ms = elapsed_sec * 1000.0
            status = getattr(response, "status_code", -1)
            path = request.url.path
            _request_counter[(path, status)] += 1
            _observe_duration(path, elapsed_sec)
            level = log.warning if elapsed_ms > SLOW_REQUEST_MS else log.info
            level(
                "http_request rid=%s path=%s status=%s ms=%.1f",
                rid, path, status, elapsed_ms,
                extra={
                    "event": "http_request",
                    "rid": rid,
                    "path": path,
                    "status": status,
                    "ms": round(elapsed_ms, 1),
                },
            )
            if response is not None:
                response.headers["X-Request-ID"] = rid


class ConcurrencyLimitMiddleware(BaseHTTPMiddleware):
    """入口全局并发上限:正在处理的请求 > MAX_INFLIGHT_REQUESTS 立即 429,不排队。"""

    async def dispatch(self, request: Request, call_next):
        try:
            await asyncio.wait_for(_global_semaphore.acquire(), timeout=0.001)
        except asyncio.TimeoutError:
            return JSONResponse({"error": "too many requests"}, status_code=429)
        try:
            return await call_next(request)
        finally:
            _global_semaphore.release()
