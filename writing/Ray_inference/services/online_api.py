"""
FastAPI 在线推理服务：提供 RESTful API 供外部调用。

=== 单图推理 ===
- POST /classify            — ResNet50 图像分类
- POST /detect              — YOLO 目标检测
- POST /segment             — SAM 实体分割
- POST /clip/similarity     — CLIP 图文相似度
- POST /clip/chat           — CLIP 简短对话

=== 批量推理（多图并发）===
- POST /classify/batch      — 多图分类
- POST /detect/batch        — 多图检测
- POST /segment/batch       — 多图分割
- POST /clip/similarity/batch — 多图相似度
- POST /clip/chat/batch     — 多图对话
"""
import traceback
from typing import List
from contextlib import asynccontextmanager

import ray
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse

from config import ONLINE_API_HOST, ONLINE_API_PORT

# ------- 全局 Actor 句柄（由 ray_deploy 注入）-------
_actors: dict = {}


def set_actors(actors: dict):
    """由部署模块注入已创建的 Actor 句柄。"""
    global _actors
    _actors = actors


def get_actor(name: str):
    a = _actors.get(name)
    if a is None:
        raise HTTPException(503, f"Model '{name}' not available")
    return a


# ------- FastAPI 生命周期 -------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时仅打印信息，实际 Actor 由 ray_deploy 管理
    print("[OnlineAPI] FastAPI 启动，等待 Ray Actor 就绪...")
    yield
    print("[OnlineAPI] FastAPI 关闭")


app = FastAPI(title="Ray Multi-Model Inference", version="1.0.0", lifespan=lifespan)


# ------- 辅助 -------
async def _read_image(file: UploadFile) -> bytes:
    if file is None:
        raise HTTPException(400, "No image file provided")
    return await file.read()


def _ray_result(ref) -> dict:
    """安全获取 Ray 异步结果。"""
    try:
        return ray.get(ref)
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ====================== 端点 ======================

@app.get("/health")
async def health():
    """所有模型的健康状态。"""
    results = {}
    for name, actor in _actors.items():
        try:
            results[name] = ray.get(actor.health_check.remote())
        except Exception:
            results[name] = {"alive": False, "error": "unreachable"}
    all_ok = all(v.get("alive", False) for v in results.values())
    return {"status": "ok" if all_ok else "degraded", "models": results}


@app.post("/classify")
async def classify(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=20)):
    """ResNet50 图像分类。"""
    image_bytes = await _read_image(file)
    actor = get_actor("resnet")
    ref = actor.infer.remote(image_bytes, top_k)
    return _ray_result(ref)


@app.post("/detect")
async def detect(file: UploadFile = File(...),
                 conf: float = Query(0.25, ge=0.01, le=1.0)):
    """YOLO 目标检测。"""
    image_bytes = await _read_image(file)
    actor = get_actor("yolo")
    ref = actor.infer.remote(image_bytes, conf)
    return _ray_result(ref)


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """SAM 实体分割。"""
    image_bytes = await _read_image(file)
    actor = get_actor("sam")
    ref = actor.infer.remote(image_bytes)
    return _ray_result(ref)


@app.post("/clip/similarity")
async def clip_similarity(file: UploadFile = File(...),
                          texts: str = Query(..., description="逗号分隔的文本列表")):
    """CLIP 图文相似度：计算图片与给定文本的匹配分数。"""
    image_bytes = await _read_image(file)
    text_list = [t.strip() for t in texts.split(",") if t.strip()]
    if not text_list:
        raise HTTPException(400, "请提供至少一个文本")
    actor = get_actor("clip")
    ref = actor.image_text_similarity.remote(image_bytes, text_list)
    return _ray_result(ref)


@app.post("/clip/chat")
async def clip_chat(file: UploadFile = File(...),
                    message: str = Query(..., description="用户消息")):
    """CLIP 简短对话：基于图片内容进行问答。"""
    image_bytes = await _read_image(file)
    actor = get_actor("clip")
    ref = actor.chat.remote(image_bytes, message)
    return _ray_result(ref)


# ====================== 批量推理端点（多图并发）======================

async def _read_files(files: List[UploadFile]) -> List[tuple[str, bytes]]:
    """读取多个上传文件，返回 [(filename, bytes), ...]。"""
    if not files:
        raise HTTPException(400, "请至少上传一张图片")
    result = []
    for f in files:
        content = await f.read()
        if not content:
            continue
        result.append((f.filename or "unknown", content))
    if not result:
        raise HTTPException(400, "未读取到有效图片数据")
    return result


def _gather_ray_results(refs: list, filenames: list[str]) -> dict:
    """并发等待所有 Ray 任务，按文件名组装结果。"""
    per_image = {}
    for name, ref in zip(filenames, refs):
        try:
            per_image[name] = ray.get(ref)
        except Exception as e:
            per_image[name] = {"success": False, "error": str(e)}
    total = len(per_image)
    success_count = sum(1 for v in per_image.values() if v.get("success"))
    return {"total": total, "success_count": success_count, "results": per_image}


@app.post("/classify/batch")
async def classify_batch(files: List[UploadFile] = File(...),
                         top_k: int = Query(5, ge=1, le=20)):
    """多图 ResNet50 分类 —— 并发处理。"""
    image_list = await _read_files(files)
    actor = get_actor("resnet")
    filenames, refs = [], []
    for fname, data in image_list:
        filenames.append(fname)
        refs.append(actor.infer.remote(data, top_k))
    return _gather_ray_results(refs, filenames)


@app.post("/detect/batch")
async def detect_batch(files: List[UploadFile] = File(...),
                       conf: float = Query(0.25, ge=0.01, le=1.0)):
    """多图 YOLO 目标检测 —— 并发处理。"""
    image_list = await _read_files(files)
    actor = get_actor("yolo")
    filenames, refs = [], []
    for fname, data in image_list:
        filenames.append(fname)
        refs.append(actor.infer.remote(data, conf))
    return _gather_ray_results(refs, filenames)


@app.post("/segment/batch")
async def segment_batch(files: List[UploadFile] = File(...)):
    """多图 SAM 实体分割 —— 并发处理。"""
    image_list = await _read_files(files)
    actor = get_actor("sam")
    filenames, refs = [], []
    for fname, data in image_list:
        filenames.append(fname)
        refs.append(actor.infer.remote(data))
    return _gather_ray_results(refs, filenames)


@app.post("/clip/similarity/batch")
async def clip_similarity_batch(files: List[UploadFile] = File(...),
                                texts: str = Query(..., description="逗号分隔的文本列表")):
    """多图 CLIP 图文相似度 —— 并发处理。"""
    text_list = [t.strip() for t in texts.split(",") if t.strip()]
    if not text_list:
        raise HTTPException(400, "请提供至少一个文本")
    image_list = await _read_files(files)
    actor = get_actor("clip")
    filenames, refs = [], []
    for fname, data in image_list:
        filenames.append(fname)
        refs.append(actor.image_text_similarity.remote(data, text_list))
    return _gather_ray_results(refs, filenames)


@app.post("/clip/chat/batch")
async def clip_chat_batch(files: List[UploadFile] = File(...),
                          message: str = Query(..., description="用户消息")):
    """多图 CLIP 简短对话 —— 并发处理。"""
    image_list = await _read_files(files)
    actor = get_actor("clip")
    filenames, refs = [], []
    for fname, data in image_list:
        filenames.append(fname)
        refs.append(actor.chat.remote(data, message))
    return _gather_ray_results(refs, filenames)


# ====================== 启动入口 ======================
def run():
    """独立启动 FastAPI（仅在特殊调试场景使用，生产由 ray_deploy 管理）。"""
    import uvicorn
    uvicorn.run(app, host=ONLINE_API_HOST, port=ONLINE_API_PORT)


if __name__ == "__main__":
    run()
