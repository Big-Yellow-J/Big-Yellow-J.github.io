"""Ray 部署编排:初始化集群、创建 Actor、健康监控、启动 FastAPI。"""
import asyncio
import signal
import time
from typing import Optional

import ray
import uvicorn

from config import (
    ACTOR_MAX_RESTARTS,
    GPU_FRACTION_RESNET,
    GPU_FRACTION_YOLO,
    HEALTH_CHECK_INTERVAL_SEC,
    ONLINE_API_HOST,
    ONLINE_API_PORT,
    RAY_ADDRESS,
    RAY_NAMESPACE,
)
from models.resnet_actor import ResNetActor
from models.yolo_actor import YOLOActor
from services.online_api import app, set_actors

# Actor 注册表:name → (class, gpu_fraction, ray_actor_name)
ACTOR_SPECS = [
    ("resnet", ResNetActor, GPU_FRACTION_RESNET, "resnet"),
    ("yolo", YOLOActor, GPU_FRACTION_YOLO, "yolo"),
]

_actors: dict = {}
_health_task: Optional[asyncio.Task] = None


def init_ray():
    if ray.is_initialized():
        return
    ray.init(
        address=RAY_ADDRESS,
        namespace=RAY_NAMESPACE,
        ignore_reinit_error=True,
        logging_level="info",
        dashboard_host="0.0.0.0",
    )
    print(f"[Deploy] Ray initialized, nodes={len(ray.nodes())}")


def create_actors() -> dict:
    """按 ACTOR_SPECS 创建所有模型 Actor 并等待就绪。"""
    actors = {}
    for key, cls, gpu_frac, ray_name in ACTOR_SPECS:
        print(f"[Deploy] launching {ray_name} (gpu={gpu_frac}) ...")
        actors[key] = cls.options(
            name=ray_name,
            num_gpus=gpu_frac,
            lifetime="detached",
            namespace=RAY_NAMESPACE,
            max_restarts=ACTOR_MAX_RESTARTS,
        ).remote()

    for key, actor in actors.items():
        try:
            ray.get(actor.health_check.remote(), timeout=120)
            print(f"  [OK] {key}")
        except Exception as e:
            print(f"  [FAIL] {key}: {e}")
            raise
    return actors


async def _health_loop(interval: float = HEALTH_CHECK_INTERVAL_SEC):
    """后台协程:周期性检查 Actor 健康,使用 asyncio 不阻塞事件循环。"""
    while True:
        await asyncio.sleep(interval)
        lines = []
        for name, actor in _actors.items():
            try:
                h = await asyncio.wait_for(
                    asyncio.wrap_future(actor.health_check.remote().future()),
                    timeout=10,
                )
                lines.append(
                    f"  {name}: alive={h.get('alive')} reqs={h.get('total_requests')}"
                )
            except Exception:
                lines.append(f"  {name}: unreachable")
        print(f"[Health @ {time.strftime('%H:%M:%S')}]\n" + "\n".join(lines))


@app.get("/actors")
async def list_actors():
    results = {}
    for name, actor in _actors.items():
        try:
            results[name] = await asyncio.wrap_future(actor.health_check.remote().future())
        except Exception as e:
            results[name] = {"alive": False, "error": str(e)}
    return {"actors": results}


async def main(host: Optional[str] = None, port: Optional[int] = None):
    global _actors, _health_task

    init_ray()
    _actors = create_actors()
    set_actors(_actors)

    loop = asyncio.get_running_loop()
    _health_task = loop.create_task(_health_loop())
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown()))

    bind_host = host or ONLINE_API_HOST
    bind_port = port or ONLINE_API_PORT
    config = uvicorn.Config(app, host=bind_host, port=bind_port, log_level="info")
    server = uvicorn.Server(config)

    print(f"\n{'=' * 60}")
    print(f"  Ray inference service running")
    print(f"  API:       http://{bind_host}:{bind_port}/docs")
    print(f"  Dashboard: http://localhost:8265")
    print(f"  Endpoints: POST /classify   POST /detect   GET /health")
    print(f"{'=' * 60}\n")

    await server.serve()


async def _shutdown():
    print("\n[Deploy] shutdown signal received")
    if _health_task:
        _health_task.cancel()
    for name, actor in _actors.items():
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass
    ray.shutdown()
    asyncio.get_event_loop().stop()


if __name__ == "__main__":
    asyncio.run(main())
