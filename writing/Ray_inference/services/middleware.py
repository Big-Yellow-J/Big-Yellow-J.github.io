"""HTTP 中间件:request_id 注入 + 并发限流 + 慢请求日志 + 按端点/状态码请求计数。"""
import asyncio
import time
import uuid
from collections import defaultdict
from typing import Tuple

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config import MAX_INFLIGHT_REQUESTS, SLOW_REQUEST_MS
from utils.logging_setup import setup_logger

log = setup_logger("api")
_global_semaphore = asyncio.Semaphore(MAX_INFLIGHT_REQUESTS)

# 按 (endpoint_path, status_code) 累计的进程内计数器;/metrics 渲染时读
_request_counter: dict = defaultdict(int)


def request_counter_snapshot() -> dict:
    """返回 {(path, status): count} 的快照。"""
    return dict(_request_counter)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """生成/透传 X-Request-ID,记录访问日志(慢请求自动升级为 WARNING),累计请求计数。"""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        t0 = time.time()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed_ms = (time.time() - t0) * 1000.0
            status = getattr(response, "status_code", -1)
            _request_counter[(request.url.path, status)] += 1
            level = log.warning if elapsed_ms > SLOW_REQUEST_MS else log.info
            level(
                "rid=%s path=%s status=%s ms=%.1f",
                rid, request.url.path, status, elapsed_ms,
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
