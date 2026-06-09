"""FastAPI 在线推理服务。

端点:
    GET  /healthz       Liveness:进程活就 200
    GET  /readyz        Readiness:三 actor 都 alive 才 200,否则 503
    GET  /health        详细健康(每次 ray.get_actor 刷新业务缓存)
    GET  /version       版本号 + git commit + 各模型 repo
    GET  /metrics       Prometheus 文本格式指标
    POST /classify      CLIP zero-shot 分类
    POST /detect        YOLOv8 目标检测
    POST /segment       OneFormer 分割
"""
import asyncio
import subprocess
from functools import lru_cache
from pathlib import Path

import ray
from fastapi import FastAPI, HTTPException, Request, Response
from ray.exceptions import RayActorError

from config import (
    APP_VERSION, CLIP_MODEL, INFER_TIMEOUT_SEC, ONEFORMER_MODEL,
    RAY_NAMESPACE, YOLO_MODEL,
)
from services.metrics import render_metrics
from services.middleware import ConcurrencyLimitMiddleware, RequestIDMiddleware
from services.schemas import ClassifyBody, DetectBody, SegmentBody
from utils.logging_setup import setup_logger

log = setup_logger("api")
_actors: dict = {}

# logical_key → ray actor 注册名(与 ray_deploy.ACTOR_SPECS 保持一致)
_RAY_NAMES = {"clip": "clip", "yolo": "yolo", "oneformer": "oneformer"}


def set_actors(actors: dict):
    """由 ray_deploy 在 startup 钩子里注入 actor handle。"""
    global _actors
    _actors = actors


def _refresh_actor(key: str):
    """从 Ray 重新拿 actor 句柄并覆盖缓存;找不到则 503。"""
    try:
        _actors[key] = ray.get_actor(_RAY_NAMES[key], namespace=RAY_NAMESPACE)
    except ValueError as e:
        raise HTTPException(503, f"model '{key}' not available") from e


def _actor(key: str):
    """按 key 取 actor,缓存未命中则 lazy attach。"""
    if key not in _actors:
        _refresh_actor(key)
    return _actors[key]


async def _await_with_timeout(ref, rid: str) -> dict:
    """异步等待 Ray 任务,超时 → ray.cancel + 504。"""
    fut = asyncio.wrap_future(ref.future())
    try:
        return await asyncio.wait_for(fut, timeout=INFER_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        ray.cancel(ref, force=False)
        raise HTTPException(504, f"inference timeout > {INFER_TIMEOUT_SEC}s (rid={rid})")


async def _infer_call(key: str, source, kwargs: dict, rid: str) -> dict:
    """调 actor.infer.remote(source, _rid=rid, **kwargs);遇 RayActorError 自动 refresh + 重试一次。"""
    kwargs = dict(kwargs)
    kwargs["_rid"] = rid                           # 透传到 actor 内的错误日志
    for attempt in (1, 2):
        ref = _actor(key).infer.remote(source, **kwargs)
        try:
            return await _await_with_timeout(ref, rid)
        except RayActorError as e:
            if attempt == 2:
                raise HTTPException(503, f"actor '{key}' died: {e} (rid={rid})") from e
            log.warning("actor '%s' died, refreshing handle (rid=%s)", key, rid)
            _actors.pop(key, None)


@lru_cache(maxsize=1)
def _git_commit() -> str:
    """读取当前仓库 short commit,失败返回 'unknown'。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


app = FastAPI(title="Ray Inference (CLIP + YOLO + OneFormer)", version=APP_VERSION)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(ConcurrencyLimitMiddleware)


@app.get("/healthz")
async def healthz():
    """Liveness probe:进程活就 200。"""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """Readiness probe:三个 actor 都能拿到且 alive 才 200,否则 503。"""
    for key, ray_name in _RAY_NAMES.items():
        try:
            actor = ray.get_actor(ray_name, namespace=RAY_NAMESPACE)
            h = await asyncio.wrap_future(actor.health_check.remote().future())
            if not h.get("alive"):
                raise HTTPException(503, f"actor '{key}' not alive")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(503, f"actor '{key}' unavailable: {e}")
    return {"status": "ready"}


@app.get("/health")
async def health():
    """详细健康:每次 ray.get_actor 重新拿句柄,顺带刷新业务端点缓存。"""
    results, refs = {}, {}
    for key, ray_name in _RAY_NAMES.items():
        try:
            actor = ray.get_actor(ray_name, namespace=RAY_NAMESPACE)
            _actors[key] = actor
            refs[key] = actor.health_check.remote()
        except ValueError as e:
            _actors.pop(key, None)
            results[key] = {"alive": False, "error": f"not found: {e}"}
    for key, ref in refs.items():
        try:
            results[key] = await asyncio.wrap_future(ref.future())
        except Exception as e:
            results[key] = {"alive": False, "error": str(e)}
    ok = all(r.get("alive") for r in results.values())
    return {"status": "ok" if ok else "degraded", "models": results}


@app.get("/version")
async def version():
    """返回应用版本、git commit、各模型 repo。"""
    return {
        "version": APP_VERSION,
        "git_commit": _git_commit(),
        "models": {"clip": CLIP_MODEL, "yolo": YOLO_MODEL, "oneformer": ONEFORMER_MODEL},
    }


@app.get("/metrics")
async def metrics():
    """Prometheus 抓取入口。"""
    return Response(await render_metrics(_actors), media_type="text/plain; version=0.0.4")


@app.post("/classify")
async def classify(body: ClassifyBody, request: Request):
    """CLIP zero-shot 分类。"""
    kwargs = body.model_dump(exclude={"source"})
    return await _infer_call("clip", body.source, kwargs, rid=request.state.request_id)


@app.post("/detect")
async def detect(body: DetectBody, request: Request):
    """YOLOv8 目标检测。"""
    kwargs = body.model_dump(exclude={"source"})
    return await _infer_call("yolo", body.source, kwargs, rid=request.state.request_id)


@app.post("/segment")
async def segment(body: SegmentBody, request: Request):
    """OneFormer 分割(instance/semantic/panoptic)。"""
    kwargs = body.model_dump(exclude={"source"})
    return await _infer_call("oneformer", body.source, kwargs, rid=request.state.request_id)
