"""
离线批处理 API 服务：通过 REST 接口提交/查询/取消批量推理任务。

端点：
- POST /batch/jobs              — 提交批处理任务（上传多图 or JSON 描述）
- GET  /batch/jobs              — 列出所有任务及状态
- GET  /batch/jobs/{job_id}     — 查询任务详情与结果
- DELETE /batch/jobs/{job_id}   — 取消/清理任务

任务状态流转：pending → running → completed / failed
"""
import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import ray
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

# ------- 数据模型 -------
class BatchTaskItem(BaseModel):
    model: str            # resnet / yolo / sam / clip
    image_path: str       # 图片绝对路径
    top_k: int = 5
    conf: float = 0.25
    texts: list[str] = []
    mode: str = "similarity"   # similarity / chat


class BatchJobRequest(BaseModel):
    """通过 JSON 提交批处理任务。"""
    tasks: List[BatchTaskItem]
    output_dir: str = ""       # 结果保存目录，为空则在内存中


class BatchJobStatus(BaseModel):
    job_id: str
    status: str          # pending / running / completed / failed
    total: int
    completed: int
    success_count: int
    fail_count: int
    created_at: str
    finished_at: Optional[str] = None
    elapsed_sec: Optional[float] = None
    results: Optional[list] = None


# ------- 全局状态（注入） -------
_actors: dict = {}
_jobs: dict[str, dict] = {}   # job_id → BatchJobStatus + data
_router: Optional[APIRouter] = None


def set_actors_for_batch(actors: dict):
    global _actors
    _actors = actors


def get_actor(name: str):
    a = _actors.get(name)
    if a is None:
        raise HTTPException(503, f"Model '{name}' not available")
    return a


# ------- Router -------
router = APIRouter(prefix="/batch/jobs", tags=["batch"])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_task_async(job_id: str, tasks: List[BatchTaskItem], output_dir: str = ""):
    """后台协程：执行批量任务并更新 _jobs 状态。"""
    job = _jobs.get(job_id)
    if not job:
        return

    job["status"] = "running"
    total = len(tasks)
    results = [None] * total
    start = time.time()

    # 按模型分组提交 Ray 任务
    futures: List[tuple[int, Optional[ray.ObjectRef]]] = []
    for i, t in enumerate(tasks):
        if not Path(t.image_path).exists():
            results[i] = {"task_index": i, "image": t.image_path,
                          "model": t.model, "success": False,
                          "error": f"File not found: {t.image_path}"}
            futures.append((i, None))
            continue

        try:
            img_bytes = Path(t.image_path).read_bytes()
        except Exception as e:
            results[i] = {"task_index": i, "image": t.image_path,
                          "model": t.model, "success": False, "error": str(e)}
            futures.append((i, None))
            continue

        actor = _actors.get(t.model)
        if actor is None:
            results[i] = {"task_index": i, "image": t.image_path,
                          "model": t.model, "success": False,
                          "error": f"Unknown model: {t.model}"}
            futures.append((i, None))
            continue

        ref = _dispatch_infer(actor, t.model, img_bytes, t)
        futures.append((i, ref))

    # 收集结果
    for i, ref in futures:
        if ref is None:
            continue
        try:
            results[i] = ray.get(ref)
            results[i]["task_index"] = i
            results[i]["image"] = tasks[i].image_path
            results[i]["model"] = tasks[i].model
        except Exception as e:
            results[i] = {"task_index": i, "image": tasks[i].image_path,
                          "model": tasks[i].model, "success": False, "error": str(e)}

        job["completed"] = sum(1 for r in results if r is not None)
        job["success_count"] = sum(1 for r in results if r and r.get("success"))
        job["fail_count"] = job["completed"] - job["success_count"]

    elapsed = time.time() - start
    job["status"] = "completed"
    job["finished_at"] = _now_iso()
    job["elapsed_sec"] = round(elapsed, 1)
    job["success_count"] = sum(1 for r in results if r and r.get("success"))
    job["fail_count"] = len(results) - job["success_count"]
    job["results"] = results

    # 可选：保存到文件
    if output_dir:
        out = Path(output_dir) / f"batch_{job_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "job_id": job_id, "status": job["status"],
            "total": total, "elapsed_sec": job["elapsed_sec"],
            "results": results,
        }, ensure_ascii=False, indent=2))


def _dispatch_infer(actor, model_name: str, img_bytes: bytes, task: BatchTaskItem):
    """根据模型名调用对应推理方法。"""
    if model_name == "resnet":
        return actor.infer.remote(img_bytes, task.top_k)
    elif model_name == "yolo":
        return actor.infer.remote(img_bytes, task.conf)
    elif model_name == "sam":
        return actor.infer.remote(img_bytes)
    elif model_name == "clip":
        if task.mode == "chat" and task.texts:
            return actor.chat.remote(img_bytes, task.texts[0])
        else:
            return actor.image_text_similarity.remote(img_bytes, task.texts or ["default"])
    return None


# ====================== 端点 ======================

@router.post("", response_model=BatchJobStatus)
async def submit_batch_job(req: BatchJobRequest):
    """提交 JSON 描述的批处理任务，返回 job_id。"""
    if not req.tasks:
        raise HTTPException(400, "tasks 不能为空")

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "total": len(req.tasks),
        "completed": 0,
        "success_count": 0,
        "fail_count": 0,
        "created_at": _now_iso(),
        "finished_at": None,
        "elapsed_sec": None,
        "results": None,
    }
    # 启动后台执行
    asyncio.create_task(_run_task_async(job_id, req.tasks, req.output_dir))
    return BatchJobStatus(**_jobs[job_id])


@router.post("/upload", response_model=BatchJobStatus)
async def submit_batch_upload(
    files: List[UploadFile] = File(...),
    model: str = Query(..., description="resnet / yolo / sam / clip"),
    top_k: int = Query(5),
    conf: float = Query(0.25),
    mode: str = Query("similarity"),
    texts: str = Query(""),
    output_dir: str = Query(""),
):
    """通过上传多张图片提交批处理任务（统一用同一模型+参数）。"""
    if not files:
        raise HTTPException(400, "请至少上传一张图片")

    # 将上传文件保存到临时目录
    tmp_dir = Path("/tmp/ray_batch_uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tasks: List[BatchTaskItem] = []
    text_list = [t.strip() for t in texts.split(",") if t.strip()] if texts else []

    for f in files:
        content = await f.read()
        if not content:
            continue
        save_path = tmp_dir / f"{uuid.uuid4().hex[:8]}_{f.filename or 'img'}"
        save_path.write_bytes(content)
        tasks.append(BatchTaskItem(
            model=model,
            image_path=str(save_path),
            top_k=top_k,
            conf=conf,
            mode=mode,
            texts=text_list,
        ))

    if not tasks:
        raise HTTPException(400, "未读取到有效图片")

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "total": len(tasks),
        "completed": 0,
        "success_count": 0,
        "fail_count": 0,
        "created_at": _now_iso(),
        "finished_at": None,
        "elapsed_sec": None,
        "results": None,
    }
    asyncio.create_task(_run_task_async(job_id, tasks, output_dir))
    return BatchJobStatus(**_jobs[job_id])


@router.get("")
async def list_batch_jobs():
    """列出所有批处理任务。"""
    return {
        "total_jobs": len(_jobs),
        "jobs": [
            {k: v for k, v in j.items() if k != "results"}
            for j in _jobs.values()
        ],
    }


@router.get("/{job_id}", response_model=BatchJobStatus)
async def get_batch_job(job_id: str):
    """查询指定批处理任务的状态与结果。"""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return BatchJobStatus(**job)


@router.delete("/{job_id}")
async def cancel_batch_job(job_id: str):
    """取消/删除批处理任务。"""
    job = _jobs.pop(job_id, None)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return {"success": True, "message": f"Job '{job_id}' removed"}
