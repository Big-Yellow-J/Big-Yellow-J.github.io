"""
Ray 部署编排模块：初始化 Ray、创建 Actor、健康监控、服务拼接。
"""
import asyncio
import signal
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import ray
import uvicorn
from fastapi import FastAPI

from config import (
    RAY_ADDRESS, RAY_NAMESPACE, ONLINE_API_HOST, ONLINE_API_PORT,
    HEALTH_CHECK_INTERVAL_SEC,
    GPU_FRACTION_RESNET, GPU_FRACTION_YOLO, GPU_FRACTION_SAM, GPU_FRACTION_CLIP,
)
from services.online_api import app, set_actors
from services.batch_worker import BatchProcessor
from services.batch_api import router as batch_router, set_actors_for_batch

# ---------- Actor 导入 ----------
from models.resnet_actor import ResNetActor
from models.yolo_actor import YOLOActor
# from models.sam_actor import SAMActor
# from models.clip_actor import CLIPActor

# 全局：用于 shutdown
_health_task: Optional[asyncio.Task] = None
_actors: dict = {}
_batch_processor: Optional[BatchProcessor] = None


# ====================== 健康监控 ======================
async def health_monitor_loop(interval: float = HEALTH_CHECK_INTERVAL_SEC):
    """
    后台协程：定期检查所有 Actor 健康状态。
    若 Actor 不可达，Ray 的 max_restarts 会自动重建。
    """
    while True:
        await asyncio.sleep(interval)
        status_lines = []
        for name, actor in _actors.items():
            try:
                h = ray.get(actor.health_check.remote(), timeout=10)
                icon = "✅" if h.get("alive") else "❌"
                status_lines.append(
                    f"  {icon} {name}: alive={h.get('alive')} "
                    f"reqs={h.get('total_requests', '?')} "
                    f"last_infer={h.get('last_inference_sec_ago', '?'):.0f}s ago"
                )
            except Exception:
                status_lines.append(f"  ❌ {name}: unreachable")
        print(f"[Health @ {time.strftime('%H:%M:%S')}]")
        for line in status_lines:
            print(line)


# ====================== Ray 初始化 ======================
def init_ray():
    """初始化 Ray 运行时。"""
    if ray.is_initialized():
        print("[Deploy] Ray 已初始化")
        return

    ray.init(
        address=RAY_ADDRESS,
        namespace=RAY_NAMESPACE,
        ignore_reinit_error=True,
        logging_level="info",
        dashboard_host="0.0.0.0",
    )
    print(f"[Deploy] Ray 初始化完成, 节点: {len(ray.nodes())}")


# ====================== 创建所有模型 Actor ======================
def create_actors() -> dict:
    """
    按 GPU 比例创建所有模型 Actor（GPU 常驻）。
    返回 {"resnet": ActorHandle, ...}
    """
    print("[Deploy] 正在创建模型 Actor（GPU 常驻）...")
    actors = {}

    # ResNet50 — 0.15 GPU
    print("  → ResNet50 ...")
    actors["resnet"] = ResNetActor.options(
        name="ResNet50",
        num_gpus=GPU_FRACTION_RESNET,
        lifetime="detached",          # Actor 独立于创建者进程存活
        namespace=RAY_NAMESPACE,
        max_restarts=3,
    ).remote()

    # YOLO — 0.30 GPU
    print("  → YOLOv8 ...")
    actors["yolo"] = YOLOActor.options(
        name="YOLOv8",
        num_gpus=GPU_FRACTION_YOLO,
        lifetime="detached",
        namespace=RAY_NAMESPACE,
        max_restarts=3,
    ).remote()

    # SAM — 0.40 GPU
    # print("  → SAM ...")
    # actors["sam"] = SAMActor.options(
    #     name="SAM",
    #     num_gpus=GPU_FRACTION_SAM,
    #     lifetime="detached",
    #     namespace=RAY_NAMESPACE,
    #     max_restarts=3,
    # ).remote()

    # CLIP — 0.15 GPU
    # print("  → CLIP ...")
    # actors["clip"] = CLIPActor.options(
    #     name="CLIP",
    #     num_gpus=GPU_FRACTION_CLIP,
    #     lifetime="detached",
    #     namespace=RAY_NAMESPACE,
    #     max_restarts=3,
    # ).remote()

    # 等待所有 Actor 就绪
    print("[Deploy] 等待 Actor 就绪...")
    for name, actor in actors.items():
        try:
            ray.get(actor.health_check.remote(), timeout=120)
            print(f"  ✅ {name} 就绪")
        except Exception as e:
            print(f"  ❌ {name} 启动失败: {e}")
            raise

    print("[Deploy] 全部 Actor 就绪 ✅")
    return actors


# ====================== 健康检查端点注入 FastAPI ======================
@app.get("/actors")
async def list_actors():
    """列出所有 Actor 及详细信息。"""
    results = {}
    for name, actor in _actors.items():
        try:
            results[name] = ray.get(actor.health_check.remote())
        except Exception:
            results[name] = {"alive": False, "error": "unreachable"}
    return {"actors": results}


# ====================== 主启动流程 ======================
async def main(port: Optional[int] = None, host: Optional[str] = None):
    global _actors, _batch_processor, _health_task

    bind_host = host or ONLINE_API_HOST
    bind_port = port or ONLINE_API_PORT

    # 1. 初始化 Ray
    init_ray()

    # 2. 创建所有 Actor（GPU 常驻）
    _actors = create_actors()

    # 3. 注入 FastAPI
    set_actors(_actors)
    set_actors_for_batch(_actors)
    _batch_processor = BatchProcessor(_actors)
    app.include_router(batch_router)

    # 4. 启动健康监控后台任务（使用 asyncio + uvicorn 共享事件循环）
    loop = asyncio.get_running_loop()
    _health_task = loop.create_task(health_monitor_loop())

    # 5. 启动 FastAPI
    config = uvicorn.Config(
        app, host=bind_host, port=bind_port, log_level="info"
    )
    server = uvicorn.Server(config)
    print(f"\n{'='*60}")
    print(f"  🚀 Ray 多模型推理服务已启动")
    print(f"  📡 在线 API:    http://{bind_host}:{bind_port}/docs")
    print(f"  📦 离线批处理:  POST /batch/jobs  |  GET /batch/jobs/{{job_id}}")
    print(f"  📊 Ray Dashboard: http://localhost:8265")
    print(f"  🔗 单图: /classify /detect /segment /clip/similarity /clip/chat")
    print(f"  🔗 多图: /classify/batch /detect/batch /segment/batch ...")
    print(f"{'='*60}\n")

    await server.serve()

    # 清理
    if _health_task:
        _health_task.cancel()


def shutdown(signum=None, frame=None):
    """优雅关闭。"""
    print("\n[Deploy] 收到退出信号，正在清理...")
    for name, actor in _actors.items():
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass
    ray.shutdown()
    print("[Deploy] 已退出")
    sys.exit(0)


# 注册信号
signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

if __name__ == "__main__":
    asyncio.run(main())
