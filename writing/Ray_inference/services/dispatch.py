"""Ray actor 调用层:句柄缓存 + 超时 + 自愈重试 + 熔断器,被各路由共享。

抽出来是为了避免 `services/routers/*` 反向 import `services/online_api.py` 造成循环。
"""
import asyncio
import time
from collections import deque

import ray
from fastapi import HTTPException
from ray.exceptions import RayActorError

from config import (
    CIRCUIT_FAIL_THRESHOLD,
    CIRCUIT_OPEN_SEC,
    CIRCUIT_WINDOW_SEC,
    INFER_TIMEOUT_SEC,
    RAY_NAMESPACE,
)
from utils.logging_setup import setup_logger

log = setup_logger("api")

_actors: dict = {}

# logical_key → Ray actor 注册名(与 ray_deploy.ACTOR_SPECS 保持一致)
_RAY_NAMES = {
    "clip": "clip",
    "yolo": "yolo",
    "oneformer": "oneformer",
    "qwen_embed": "qwen_embed",
}

# 熔断器状态:{key: {"open_until": ts, "fails": deque[ts]}}
_circuit: dict = {}


def _circuit_check(key: str) -> None:
    """打开状态直接 503,关闭状态过路。"""
    state = _circuit.get(key)
    if state and state.get("open_until", 0) > time.time():
        raise HTTPException(503, f"circuit open for '{key}', retry later")


def _circuit_record_fail(key: str) -> None:
    """记一次失败,达到阈值就开熔断 CIRCUIT_OPEN_SEC 秒。"""
    now = time.time()
    state = _circuit.setdefault(key, {"open_until": 0.0, "fails": deque()})
    fails: deque = state["fails"]
    fails.append(now)
    while fails and now - fails[0] > CIRCUIT_WINDOW_SEC:
        fails.popleft()
    if len(fails) >= CIRCUIT_FAIL_THRESHOLD:
        state["open_until"] = now + CIRCUIT_OPEN_SEC
        fails.clear()
        log.warning("circuit OPENED for '%s' for %ds", key, CIRCUIT_OPEN_SEC)


def _circuit_record_success(key: str) -> None:
    """成功就清空失败窗口(半开转关闭)。"""
    if key in _circuit:
        _circuit[key]["fails"].clear()


def circuit_snapshot() -> dict:
    """{key: {"open": bool, "fails": int, "open_until": ts}};metrics / 调试用。"""
    now = time.time()
    return {
        k: {
            "open": v["open_until"] > now,
            "fails": len(v["fails"]),
            "open_until": v["open_until"],
        }
        for k, v in _circuit.items()
    }


def set_actors(actors: dict) -> None:
    """ray_deploy 启动钩子注入 actor handle。"""
    global _actors
    _actors = actors


def get_actors() -> dict:
    """metrics / 健康检查等只读访问。"""
    return _actors


def _refresh_actor(key: str) -> None:
    """从 Ray 重新拿句柄并覆盖缓存;找不到 → 503。"""
    if key not in _RAY_NAMES:
        raise HTTPException(400, f"unknown actor key '{key}'")
    try:
        _actors[key] = ray.get_actor(_RAY_NAMES[key], namespace=RAY_NAMESPACE)
    except ValueError as e:
        raise HTTPException(503, f"model '{key}' not available") from e


def actor_handle(key: str):
    """按 key 取 actor 句柄,缓存未命中则 lazy attach。"""
    if key not in _actors:
        _refresh_actor(key)
    return _actors[key]


async def _await_with_timeout(ref, rid: str) -> dict:
    """异步等 Ray 任务,超时 → ray.cancel + 504。"""
    fut = asyncio.wrap_future(ref.future())
    try:
        return await asyncio.wait_for(fut, timeout=INFER_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        ray.cancel(ref, force=False)
        raise HTTPException(504, f"inference timeout > {INFER_TIMEOUT_SEC}s (rid={rid})")


async def actor_call(key: str, method: str, source, kwargs: dict, rid: str) -> dict:
    """通用 actor 方法调用:熔断保护 + 超时 + RayActorError 自动 refresh + 重试一次。"""
    _circuit_check(key)
    kwargs = dict(kwargs)
    kwargs["_rid"] = rid                           # 透传到 actor 错误日志
    for attempt in (1, 2):
        ref = getattr(actor_handle(key), method).remote(source, **kwargs)
        try:
            result = await _await_with_timeout(ref, rid)
            _circuit_record_success(key)
            return result
        except RayActorError as e:
            _circuit_record_fail(key)
            if attempt == 2:
                raise HTTPException(503, f"actor '{key}' died: {e} (rid={rid})") from e
            log.warning("actor '%s' died, refreshing handle (rid=%s)", key, rid)
            _actors.pop(key, None)
        except HTTPException as e:
            if e.status_code == 504:               # 超时也算熔断信号
                _circuit_record_fail(key)
            raise
