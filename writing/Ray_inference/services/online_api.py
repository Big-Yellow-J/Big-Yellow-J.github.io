"""FastAPI 在线推理服务。

端点:
    GET  /health        全模型健康状态(每次都 ray.get_actor,顺带刷新业务缓存)
    GET  /metrics       Prometheus 文本格式指标
    POST /classify      CLIP zero-shot 分类
    POST /detect        YOLOv8 目标检测
    POST /segment       OneFormer 分割(默认 instance)

请求体统一为 JSON,source 字段支持:本地路径 / http(s) URL / base64 / data URI。
大文件用 base64 编码后放在 source 字段。
"""
import asyncio
import logging
from typing import List, Optional

import ray
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from ray.exceptions import RayActorError

from config import INFER_TIMEOUT_SEC, RAY_NAMESPACE
from services.metrics import render_metrics
from services.middleware import ConcurrencyLimitMiddleware, RequestIDMiddleware

log = logging.getLogger("ray_inference.api")
_actors: dict = {}

# logical_key → ray actor 注册名(与 ray_deploy.ACTOR_SPECS 保持一致)
_RAY_NAMES = {"clip": "clip", "yolo": "yolo", "oneformer": "oneformer"}


def set_actors(actors: dict):
    """由 ray_deploy 在 startup 钩子里注入 actor handle。"""
    global _actors
    _actors = actors


def _refresh_actor(key: str):
    """从 Ray 重新拿 actor 句柄并覆盖缓存;集群中找不到则 503。"""
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


async def _infer_call(key: str, args: tuple, rid: str) -> dict:
    """调 actor.infer.remote(*args);遇 RayActorError(句柄指向的实例已死)自动 refresh + 重试一次。"""
    for attempt in (1, 2):
        ref = _actor(key).infer.remote(*args)
        try:
            return await _await_with_timeout(ref, rid)
        except RayActorError as e:
            if attempt == 2:
                raise HTTPException(503, f"actor '{key}' died: {e} (rid={rid})") from e
            log.warning("actor '%s' died, refreshing handle (rid=%s)", key, rid)
            _actors.pop(key, None)   # 清缓存,下一轮 _actor() 会触发 ray.get_actor


class ClassifyBody(BaseModel):
    source: str
    labels: List[str] = Field(..., min_items=1)
    top_k: int = 5


class DetectBody(BaseModel):
    source: str
    conf: float = 0.25


class SegmentBody(BaseModel):
    source: str
    task: str = "instance"          # "instance" | "semantic" | "panoptic"
    return_mask: bool = False


app = FastAPI(title="Ray Inference (CLIP + YOLO + OneFormer)", version="3.0.0")
app.add_middleware(RequestIDMiddleware)
app.add_middleware(ConcurrencyLimitMiddleware)


@app.get("/health")
async def health():
    """每次都 ray.get_actor 重新拿句柄,顺带刷新业务端点缓存。"""
    results = {}
    refs = {}
    for key, ray_name in _RAY_NAMES.items():
        try:
            actor = ray.get_actor(ray_name, namespace=RAY_NAMESPACE)
            _actors[key] = actor                          # 同步刷新缓存
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


@app.get("/metrics")
async def metrics():
    """Prometheus 抓取入口。"""
    return Response(await render_metrics(_actors), media_type="text/plain; version=0.0.4")


@app.post("/classify")
async def classify(body: ClassifyBody, request: Request):
    """CLIP zero-shot 分类。"""
    return await _infer_call(
        "clip", (body.source, body.labels, body.top_k), rid=request.state.request_id,
    )


@app.post("/detect")
async def detect(body: DetectBody, request: Request):
    """YOLOv8 目标检测。"""
    return await _infer_call(
        "yolo", (body.source, body.conf), rid=request.state.request_id,
    )


@app.post("/segment")
async def segment(body: SegmentBody, request: Request):
    """OneFormer 分割(instance/semantic/panoptic)。"""
    return await _infer_call(
        "oneformer", (body.source, body.task, body.return_mask), rid=request.state.request_id,
    )
