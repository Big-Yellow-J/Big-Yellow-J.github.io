#!/usr/bin/env python3
"""
Ray 多模型推理平台 — 一键启动入口。

用法:
    python main.py                    # 启动在线服务 + 离线批处理支持
    python main.py --batch <dir>     # 仅离线批处理（不启动 API）

架构:
    main.py → ray_deploy.py → models/*.py + services/*.py
                                    ↳ online_api.py (FastAPI)
                                    ↳ batch_worker.py (离线批处理)
"""
import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Ray 多模型推理平台")
    sub = parser.add_subparsers(dest="command", help="命令")

    # serve — 启动完整在线服务
    serve_parser = sub.add_parser("serve", help="启动在线 API + 离线批处理（GPU 常驻）")
    serve_parser.add_argument("--port", type=int, default=8000, help="API 服务端口 (默认 8000)")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定地址 (默认 0.0.0.0)")

    # batch — 仅离线批处理
    batch_parser = sub.add_parser("batch", help="仅离线批处理")
    batch_parser.add_argument("--task-file", type=str, help="JSON 任务文件路径")
    batch_parser.add_argument("--image-dir", type=str, help="图片文件夹")
    batch_parser.add_argument("--mode", choices=["classify", "detect", "segment"],
                               default="classify", help="批处理模式")
    batch_parser.add_argument("--output", type=str, default="batch_result.json",
                               help="结果输出文件")

    # status — 查看 Actor 状态（连接已有集群）
    sub.add_parser("status", help="查看已有集群中 Actor 状态")

    args = parser.parse_args()

    if args.command == "serve":
        import asyncio
        from ray_deploy import main as deploy_main
        asyncio.run(deploy_main(port=args.port, host=args.host))

    elif args.command == "batch":
        from ray_deploy import init_ray, create_actors
        from services.batch_worker import BatchProcessor

        init_ray()
        actors = create_actors()
        bp = BatchProcessor(actors)

        if args.task_file:
            result = bp.run_task_file(args.task_file)
        elif args.image_dir:
            if args.mode == "classify":
                result = bp.batch_classify(args.image_dir, args.output)
            elif args.mode == "detect":
                result = bp.batch_detect(args.image_dir, args.output)
            else:
                result = bp.batch_segment(args.image_dir, args.output)
        else:
            print("请指定 --task-file 或 --image-dir")
            sys.exit(1)

        import json
        print(json.dumps({"success": result.get("success"), "total": result.get("total"),
                          "elapsed_sec": result.get("elapsed_sec")}, ensure_ascii=False))

    elif args.command == "status":
        import ray
        from ray_deploy import init_ray
        init_ray()

        # 尝试获取已存在的 detached Actor
        from models.resnet_actor import ResNetActor
        from models.yolo_actor import YOLOActor
        # from models.sam_actor import SAMActor
        # from models.clip_actor import CLIPActor
        from config import RAY_NAMESPACE

        model_classes = {
            "resnet": ResNetActor,
            "yolo": YOLOActor,
            # "sam": SAMActor,
            # "clip": CLIPActor,
        }

        for name, cls in model_classes.items():
            try:
                actor = ray.get_actor(name, namespace=RAY_NAMESPACE)
                h = ray.get(actor.health_check.remote(), timeout=5)
                print(f"✅ {name}: {h}")
            except Exception as e:
                print(f"❌ {name}: {e}")

    else:
        # 默认启动服务
        import asyncio
        from ray_deploy import main as deploy_main
        asyncio.run(deploy_main())


if __name__ == "__main__":
    main()
