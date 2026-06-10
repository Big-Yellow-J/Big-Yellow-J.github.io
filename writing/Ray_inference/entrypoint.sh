#!/usr/bin/env bash
# 容器入口:首次启动如缺权重则下载,再 exec 给 main.py serve 接管。
# SIGTERM 由 tini → python → ray_deploy._shutdown 链路传递,等 inflight 归零再退出。
set -euo pipefail

API_PORT="${ONLINE_API_PORT:-7890}"
API_HOST="${ONLINE_API_HOST:-0.0.0.0}"

# 权重缺失自动 prepare,避免首次跑容器时 actor 加载失败
if [ ! -d "/app/weights/clip-vit-base-patch32" ] || \
   [ ! -d "/app/weights/oneformer_ade20k_swin_large" ] || \
   [ ! -d "/app/weights/Qwen3-VL-Embedding-2B" ]; then
    echo "[entrypoint] weights missing, running prepare (~6GB, 数分钟)"
    python main.py prepare
fi

echo "[entrypoint] starting service on ${API_HOST}:${API_PORT}"
exec python main.py serve --host "${API_HOST}" --port "${API_PORT}"
