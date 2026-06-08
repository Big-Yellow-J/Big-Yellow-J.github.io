#!/usr/bin/env python3
"""Ray 推理服务入口。

用法:
    python main.py serve [--host 0.0.0.0] [--port 8000]   启动服务
    python main.py status                                  查看已运行 actor 状态
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def cmd_serve(args):
    from ray_deploy import main
    asyncio.run(main(host=args.host, port=args.port))


def cmd_status(args):
    import ray
    from config import RAY_NAMESPACE
    from ray_deploy import ACTOR_SPECS, init_ray

    init_ray()
    for key, _, _, ray_name in ACTOR_SPECS:
        try:
            actor = ray.get_actor(ray_name, namespace=RAY_NAMESPACE)
            health = ray.get(actor.health_check.remote(), timeout=5)
            print(f"[OK]   {key}: {health}")
        except Exception as e:
            print(f"[FAIL] {key}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Ray inference service")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="start API service")
    serve.add_argument("--host", type=str, default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=cmd_serve)

    status = sub.add_parser("status", help="check actor status")
    status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
