#!/usr/bin/env python3
"""Ray 推理服务入口。

用法:
    python main.py prepare [--force]                      下载所有 HF 模型权重到 weights/(首次必须)
    python main.py serve [--host 0.0.0.0] [--port 8000]   一键启动(Ray + Actor + API)
    python main.py bootstrap                              创建/复用 detached actor;失败 exit 1
    python main.py teardown                               杀掉全部 detached actor(Ray 集群保留)
    python main.py status                                 查看已注册 actor 健康状态
    python main.py milvus stats                           列出每个 collection 的行数 / 维度
    python main.py milvus backup                          备份 Milvus Lite db 到 data/backup/
    python main.py milvus list-backups                    列出所有备份
    python main.py milvus drop <model>                    删除某模型的 collection(谨慎)
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def cmd_prepare(args):
    """下载所有 HF 模型权重到项目内 weights/,启动时强制本地加载。"""
    from utils.prepare_models import download_all
    try:
        download_all(force=args.force)
    except Exception as e:
        print(f"prepare FAILED: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_serve(args):
    from ray_deploy import main
    asyncio.run(main(host=args.host, port=args.port))


def cmd_bootstrap(args):
    """创建/确保 detached actor 就绪,失败返回非零退出码。"""
    from ray_deploy import create_actors, init_ray
    try:
        init_ray()
        create_actors()
        print("bootstrap OK")
    except Exception as e:
        print(f"bootstrap FAILED: {e}", file=sys.stderr)
        sys.exit(1)


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


def cmd_milvus(args):
    """milvus 子命令分发:stats / backup / list-backups / drop。"""
    from services.db.milvus import collection_for, get_milvus
    from utils.milvus_backup import backup_lite, list_backups

    if args.action == "stats":
        client = get_milvus()
        collections = client.list_collections()
        if not collections:
            print("(no collections)")
            return
        for name in collections:
            try:
                stats = client.get_collection_stats(name)
                print(f"  {name}: {stats}")
            except Exception as e:
                print(f"  {name}: FAILED ({e})")

    elif args.action == "backup":
        r = backup_lite()
        print(r)

    elif args.action == "list-backups":
        backups = list_backups()
        if not backups:
            print("(no backups)")
            return
        for b in backups:
            print(f"  {b['mtime']}  {b['size_mb']:>7.2f} MB  {b['path']}")

    elif args.action == "drop":
        name = collection_for(args.model)
        confirm = input(f"DROP collection '{name}'? type 'YES' to confirm: ")
        if confirm != "YES":
            print("aborted")
            return
        get_milvus().drop_collection(name)
        print(f"dropped {name}")


def main():
    parser = argparse.ArgumentParser(description="Ray inference service")
    sub = parser.add_subparsers(dest="command")

    prepare = sub.add_parser("prepare", help="download all HF weights to weights/")
    prepare.add_argument("--force", action="store_true", help="redownload even if local exists")
    prepare.set_defaults(func=cmd_prepare)

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

    # milvus 子命令组
    milvus = sub.add_parser("milvus", help="milvus admin tools")
    milvus_sub = milvus.add_subparsers(dest="action", required=True)
    milvus_sub.add_parser("stats", help="show collection row counts")
    milvus_sub.add_parser("backup", help="backup Milvus Lite db to data/backup/")
    milvus_sub.add_parser("list-backups", help="list all backups")
    drop = milvus_sub.add_parser("drop", help="drop one collection by model name")
    drop.add_argument("model", help="e.g. clip / qwen_vl")
    milvus.set_defaults(func=cmd_milvus)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
