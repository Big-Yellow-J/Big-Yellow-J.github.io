"""HTTP 中间件:request_id 注入 + 全局并发限流。"""
import asyncio
import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config import MAX_INFLIGHT_REQUESTS

log = logging.getLogger("ray_inference.access")

_global_semaphore = asyncio.Semaphore(MAX_INFLIGHT_REQUESTS)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """为每个请求生成或透传 X-Request-ID,挂到 request.state.request_id 并写入访问日志。"""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        t0 = time.time()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            log.info(
                "rid=%s path=%s status=%s ms=%.1f",
                rid, request.url.path,
                getattr(response, "status_code", "-"),
                (time.time() - t0) * 1000.0,
            )
            if response is not None:
                response.headers["X-Request-ID"] = rid


class ConcurrencyLimitMiddleware(BaseHTTPMiddleware):
    """入口全局并发上限:正在处理的请求超过 MAX_INFLIGHT_REQUESTS 立即 429。"""

    async def dispatch(self, request: Request, call_next):
        try:
            # acquire 超时设为极小值 → 等价于 try_acquire,排队即视为过载
            await asyncio.wait_for(_global_semaphore.acquire(), timeout=0.001)
        except asyncio.TimeoutError:
            return JSONResponse({"error": "too many requests"}, status_code=429)
        try:
            return await call_next(request)
        finally:
            _global_semaphore.release()
