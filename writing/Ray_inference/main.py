#!/usr/bin/env python3
"""Ray 推理服务入口。

用法:
    python main.py serve [--host 0.0.0.0] [--port 8000]   一键启动(Ray + Actor + API)
    python main.py bootstrap                              连接已有 Ray 集群,创建/复用 detached actor
    python main.py teardown                               杀掉全部 detached actor(Ray 集群保留)
    python main.py status                                 查看已注册 actor 健康状态
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def cmd_serve(args):
    from ray_deploy import main
    asyncio.run(main(host=args.host, port=args.port))


def cmd_bootstrap(args):
    """创建/确保 detached actor 就绪,完事退出(供 uvicorn --reload 模式提前准备)。"""
    from ray_deploy import create_actors, init_ray
    init_ray()
    create_actors()
    print("bootstrap OK")


def cmd_teardown(args):
    """杀掉所有 detached actor,Ray 集群本身用 `ray stop` 关闭。"""
    import ray
    from config import RAY_NAMESPACE
    from ray_deploy import ACTOR_SPECS, init_ray

    init_ray()
    for key, _, _, ray_name in ACTOR_SPECS:
        try:
            ray.kill(ray.get_actor(ray_name, namespace=RAY_NAMESPACE), no_restart=True)
            print(f"killed {ray_name}")
        except ValueError:
            print(f"skip {ray_name} (not found)")


def cmd_status(args):
    import ray
    from config import RAY_NAMESPACE
    from ray_deploy import ACTOR_SPECS, init_ray

    init_ray()
    for key, _, _, ray_name in ACTOR_SPECS:
        try:
            actor = ray.get_actor(ray_name, namespace=RAY_NAMESPACE)
            print(f"[OK]   {key}: {ray.get(actor.health_check.remote(), timeout=5)}")
        except Exception as e:
            print(f"[FAIL] {key}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Ray inference service")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="start API service (Ray + Actor + uvicorn)")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=cmd_serve)

    bootstrap = sub.add_parser("bootstrap", help="create/reuse detached actors then exit")
    bootstrap.set_defaults(func=cmd_bootstrap)

    teardown = sub.add_parser("teardown", help="kill all detached actors")
    teardown.set_defaults(func=cmd_teardown)

    status = sub.add_parser("status", help="check actor status")
    status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
