"""FastAPI 在线推理服务。

端点:
    GET  /healthz       Liveness:进程活就 200
    GET  /readyz        Readiness:所有 actor 都 alive 才 200,否则 503
    GET  /health        详细健康(每次 ray.get_actor 刷新业务缓存)
    GET  /version       版本号 + git commit + 各模型 repo + milvus 元信息
    GET  /metrics       Prometheus 文本格式指标
    POST /classify      CLIP zero-shot 分类
    POST /detect        YOLOv8 目标检测
    POST /segment       OneFormer 分割
    POST /v1/embed      图像 embedding(CLIP 默认 / Qwen)+ 自动写 milvus
    POST /v1/search     以图搜图(走 milvus 向量检索)
"""
import asyncio
import subprocess
from functools import lru_cache
from pathlib import Path

import ray
from fastapi import FastAPI, HTTPException, Request, Response

from config import (
    APP_VERSION,
    CLIP_MODEL,
    CLIP_REVISION,
    MILVUS_COLLECTION_PREFIX,
    MILVUS_URI,
    ONEFORMER_MODEL,
    ONEFORMER_REVISION,
    QWEN_EMBED_MODEL,
    QWEN_EMBED_REVISION,
    RAY_NAMESPACE,
    YOLO_MODEL,
)
from services.db.milvus import close_milvus, ping as milvus_ping
from services.dispatch import (
    _RAY_NAMES,
    actor_call,
    circuit_snapshot,
    get_actors,
)
from services.metrics import render_metrics
from services.middleware import (
    ConcurrencyLimitMiddleware,
    RequestIDMiddleware,
)
from services.routers.embed import cache_stats, router as embed_router
from services.schemas import ClassifyBody, DetectBody, SegmentBody
from utils.logging_setup import setup_logger

log = setup_logger("api")


async def _infer_call(key: str, source, kwargs: dict, rid: str) -> dict:
    """薄包装:复用 dispatch.actor_call 调 actor.infer 方法。"""
    return await actor_call(key, "infer", source, kwargs, rid)


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


app = FastAPI(title="Ray Inference (CLIP + YOLO + OneFormer + Qwen-Embed)", version=APP_VERSION)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(ConcurrencyLimitMiddleware)
app.include_router(embed_router)


@app.on_event("shutdown")
async def _on_shutdown():
    """优雅退出:关闭 milvus 连接。"""
    close_milvus()


@app.get("/healthz")
async def healthz():
    """Liveness probe:进程活就 200。"""
    return {"status": "ok"}


HEALTH_PROBE_TIMEOUT_SEC = 3.0   # 单个 actor health_check 超时;防止某 actor __init__ 卡住拖死整个探针


async def _probe_actor(ref) -> dict:
    """对单个 actor health_check ref 做有超时的 await,超时记为 timeout 而非阻塞整个探针。"""
    try:
        return await asyncio.wait_for(
            asyncio.wrap_future(ref.future()),
            timeout=HEALTH_PROBE_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        return {"alive": False, "error": f"timeout >{HEALTH_PROBE_TIMEOUT_SEC}s (actor loading or stuck)"}


@app.get("/readyz")
async def readyz():
    """Readiness probe:所有 actor + milvus 都就绪才 200,否则 503。单 actor 卡住 3s 超时。"""
    for key, ray_name in _RAY_NAMES.items():
        try:
            actor = ray.get_actor(ray_name, namespace=RAY_NAMESPACE)
        except Exception as e:
            raise HTTPException(503, f"actor '{key}' unavailable: {e}")
        h = await _probe_actor(actor.health_check.remote())
        if not h.get("alive"):
            raise HTTPException(503, f"actor '{key}' not alive: {h.get('error', '')}")

    try:
        milvus_ping()
    except Exception as e:
        raise HTTPException(503, f"milvus unavailable: {e}")
    return {"status": "ready"}


@app.get("/health")
async def health():
    """详细健康:每个 actor 独立超时,慢的不拖累快的。"""
    results, refs = {}, {}
    actors = get_actors()
    for key, ray_name in _RAY_NAMES.items():
        try:
            actor = ray.get_actor(ray_name, namespace=RAY_NAMESPACE)
            actors[key] = actor
            refs[key] = actor.health_check.remote()
        except ValueError as e:
            actors.pop(key, None)
            results[key] = {"alive": False, "error": f"not found: {e}"}

    # 并发等待所有 actor,每个独立超时,慢的会标 timeout 而不阻塞其他
    keys = list(refs.keys())
    probes = await asyncio.gather(*[_probe_actor(refs[k]) for k in keys])
    for k, h in zip(keys, probes):
        results[k] = h

    ok = all(r.get("alive") for r in results.values())
    return {"status": "ok" if ok else "degraded", "models": results}


@app.get("/version")
async def version():
    """返回应用版本、git commit、各模型 repo + revision sha、milvus 元信息、运行时状态。"""
    return {
        "version": APP_VERSION,
        "git_commit": _git_commit(),
        "models": {
            "clip": {"path": CLIP_MODEL, "revision": CLIP_REVISION or "HEAD"},
            "yolo": {"path": YOLO_MODEL, "revision": "n/a"},
            "oneformer": {"path": ONEFORMER_MODEL, "revision": ONEFORMER_REVISION or "HEAD"},
            "qwen_embed": {"path": QWEN_EMBED_MODEL, "revision": QWEN_EMBED_REVISION or "HEAD"},
        },
        "milvus": {
            "uri": MILVUS_URI,
            "collection_prefix": MILVUS_COLLECTION_PREFIX,
        },
        "runtime": {
            "circuit": circuit_snapshot(),
            "cache": cache_stats(),
        },
    }


@app.get("/metrics")
async def metrics():
    """Prometheus 抓取入口。"""
    return Response(await render_metrics(get_actors()), media_type="text/plain; version=0.0.4")


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
