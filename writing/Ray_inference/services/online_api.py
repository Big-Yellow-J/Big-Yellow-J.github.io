"""FastAPI 在线推理服务。

端点:
    GET  /health        全模型健康状态
    POST /classify      ResNet50 分类
    POST /detect        YOLO 目标检测

输入支持三种方式:
    1) multipart/form-data 上传文件: file=@cat.jpg
    2) JSON: {"source": "<local path | URL | base64 | data URI>"}
"""
import asyncio
from typing import Optional

import ray
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from config import MAX_IMAGE_BYTES

_actors: dict = {}


def set_actors(actors: dict):
    global _actors
    _actors = actors


def _get_actor(name: str):
    actor = _actors.get(name)
    if actor is None:
        raise HTTPException(503, f"Model '{name}' not available")
    return actor


async def _ray_await(ref) -> dict:
    """异步等待 Ray 任务结果,不阻塞事件循环。"""
    try:
        return await asyncio.wrap_future(ref.future())
    except Exception as e:
        return {"success": False, "error": str(e)}


class InferRequest(BaseModel):
    source: str  # 本地路径 / http(s) URL / base64 / data URI


app = FastAPI(title="Ray Inference (ResNet50 + YOLO)", version="2.0.0")


async def _resolve_source(file: Optional[UploadFile], body: Optional[InferRequest]):
    """从 multipart 或 JSON body 中提取图像源。"""
    if file is not None:
        data = await file.read()
        if not data:
            raise HTTPException(400, "Empty file")
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"File too large (> {MAX_IMAGE_BYTES} bytes)")
        return data
    if body is not None and body.source:
        return body.source
    raise HTTPException(400, "Provide either multipart 'file' or JSON {'source': ...}")


@app.get("/health")
async def health():
    refs = {name: actor.health_check.remote() for name, actor in _actors.items()}
    results = {}
    for name, ref in refs.items():
        try:
            results[name] = await asyncio.wrap_future(ref.future())
        except Exception as e:
            results[name] = {"alive": False, "error": str(e)}
    all_ok = all(v.get("alive") for v in results.values())
    return {"status": "ok" if all_ok else "degraded", "models": results}


@app.post("/classify")
async def classify(
    file: Optional[UploadFile] = File(None),
    body: Optional[InferRequest] = None,
    top_k: int = Query(5, ge=1, le=20),
):
    """ResNet50 图像分类。"""
    source = await _resolve_source(file, body)
    actor = _get_actor("resnet")
    return await _ray_await(actor.infer.remote(source, top_k))


@app.post("/detect")
async def detect(
    file: Optional[UploadFile] = File(None),
    body: Optional[InferRequest] = None,
    conf: float = Query(0.25, ge=0.01, le=1.0),
):
    """YOLO 目标检测。"""
    source = await _resolve_source(file, body)
    actor = _get_actor("yolo")
    return await _ray_await(actor.infer.remote(source, conf))
