"""Ray 部署编排:初始化集群、创建 Actor、健康巡检、装配并运行 FastAPI。"""
import asyncio
import logging
import signal
from typing import Optional

import ray
import uvicorn

from config import (
    ACTOR_MAX_RESTARTS,
    GPU_FRACTION_CLIP,
    GPU_FRACTION_ONEFORMER,
    GPU_FRACTION_YOLO,
    HEALTH_CHECK_INTERVAL_SEC,
    ONLINE_API_HOST,
    ONLINE_API_PORT,
    RAY_ADDRESS,
    RAY_DASHBOARD_HOST,
    RAY_NAMESPACE,
)
from models.clip_actor import CLIPActor
from models.oneformer_actor import OneFormerActor
from models.yolo_actor import YOLOActor
from services.online_api import app, set_actors

log = logging.getLogger("ray_inference.deploy")

# (logical_key, actor_class, gpu_fraction, ray_actor_name)
ACTOR_SPECS = [
    ("clip", CLIPActor, GPU_FRACTION_CLIP, "clip"),
    ("yolo", YOLOActor, GPU_FRACTION_YOLO, "yolo"),
    ("oneformer", OneFormerActor, GPU_FRACTION_ONEFORMER, "oneformer"),
]

_actors: dict = {}
_health_task: Optional[asyncio.Task] = None


def init_ray():
    """初始化或附加到 Ray 集群,dashboard 默认绑回环避免公网暴露。"""
    if ray.is_initialized():
        return
    ray.init(
        address=RAY_ADDRESS,
        namespace=RAY_NAMESPACE,
        ignore_reinit_error=True,
        logging_level="info",
        dashboard_host=RAY_DASHBOARD_HOST,
    )
    log.info("ray initialized nodes=%d", len(ray.nodes()))


def create_actors() -> dict:
    """按 ACTOR_SPECS 创建/复用 detached Actor 并等待就绪(幂等)。

    Returns:
        {logical_key: ray_actor_handle}
    """
    actors = {}
    for key, cls, gpu_frac, ray_name in ACTOR_SPECS:
        try:
            actors[key] = ray.get_actor(ray_name, namespace=RAY_NAMESPACE)
            log.info("  [reuse] %s already exists", ray_name)
            continue
        except ValueError:
            pass
        log.info("launching %s (gpu=%s)", ray_name, gpu_frac)
        actors[key] = cls.options(
            name=ray_name,
            num_gpus=gpu_frac,
            lifetime="detached",        # API 重启不重载模型
            namespace=RAY_NAMESPACE,
            max_restarts=ACTOR_MAX_RESTARTS,
        ).remote()

    for key, actor in actors.items():
        ray.get(actor.health_check.remote(), timeout=300)   # OneFormer-large 首次加载较慢
        log.info("  [OK] %s ready", key)
    return actors


def attach_actors() -> dict:
    """从已存在的 Ray 集群拿各 detached actor 句柄;缺失则抛错引导先 bootstrap。"""
    actors = {}
    for key, _, _, ray_name in ACTOR_SPECS:
        try:
            actors[key] = ray.get_actor(ray_name, namespace=RAY_NAMESPACE)
        except ValueError as e:
            raise RuntimeError(
                f"actor '{ray_name}' not found in namespace '{RAY_NAMESPACE}'. "
                f"run `python main.py bootstrap` first (after `ray start --head`)."
            ) from e
    return actors


async def _health_loop(interval: float = HEALTH_CHECK_INTERVAL_SEC):
    """后台周期性巡检,日志输出每个 actor 的累计请求与错误数。"""
    while True:
        await asyncio.sleep(interval)
        for name, actor in _actors.items():
            try:
                h = await asyncio.wait_for(
                    asyncio.wrap_future(actor.health_check.remote().future()),
                    timeout=10,
                )
                log.info(
                    "health model=%s reqs=%d errs=%d avg_ms=%s",
                    name, h["total_requests"], h["total_errors"], h["avg_latency_ms"],
                )
            except Exception as e:
                log.warning("health model=%s unreachable: %s", name, e)


@app.on_event("startup")
async def _startup_attach():
    """uvicorn 模式启动钩子:attach 到已存在的 Ray 集群和 detached actor。

    `python main.py serve` 模式下 main() 已经填充 _actors,此 hook 直接跳过。
    """
    global _actors
    if _actors:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    init_ray()
    _actors = attach_actors()
    set_actors(_actors)
    log.info("uvicorn mode: attached %d actors", len(_actors))


@app.get("/actors")
async def list_actors():
    """完整 actor 状态(比 /health 多 device/gpu_fraction/last_inference 等字段)。"""
    results = {}
    for name, actor in _actors.items():
        try:
            results[name] = await asyncio.wrap_future(actor.health_check.remote().future())
        except Exception as e:
            results[name] = {"alive": False, "error": str(e)}
    return {"actors": results}


async def main(host: Optional[str] = None, port: Optional[int] = None):
    """部署入口:Ray 集群 → Actor → FastAPI,捕获 SIGINT/SIGTERM 优雅退出。"""
    global _actors, _health_task

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

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
    log.info("API   http://%s:%s/docs", bind_host, bind_port)
    log.info("Dash  http://%s:8265", RAY_DASHBOARD_HOST)
    await uvicorn.Server(config).serve()


async def _shutdown():
    """优雅退出:取消巡检 → 杀掉 actor(不重启)→ shutdown Ray。"""
    log.info("shutdown signal received")
    if _health_task:
        _health_task.cancel()
    for actor in _actors.values():
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass
    ray.shutdown()
    asyncio.get_event_loop().stop()


if __name__ == "__main__":
    asyncio.run(main())
