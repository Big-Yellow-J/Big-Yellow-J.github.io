"""
全局配置：模型路径、GPU 分配、服务端口、健康检查参数。
"""
from pathlib import Path

# ---------- 路径 ----------
PROJECT_ROOT = Path(__file__).parent
MODEL_CACHE_DIR = Path.home() / ".cache" / "ray_inference_models"

# ---------- GPU 分配（单卡 40G）----------
# 每个 Actor 申请的 GPU 比例，总和 ≤ 1.0
GPU_FRACTION_RESNET = 0.15   # ResNet50 ~100MB
GPU_FRACTION_YOLO   = 0.30   # YOLO       ~2GB
GPU_FRACTION_SAM    = 0.40   # SAM        ~2.5GB
GPU_FRACTION_CLIP   = 0.15   # CLIP       ~1.5GB

# ---------- 模型预训练权重 ----------
RESNET_MODEL_NAME = "resnet50"
YOLO_MODEL_NAME   = "yolov8n.pt"          # nano 版本，适合 40G
SAM_MODEL_TYPE    = "vit_b"               # base 版本
SAM_CHECKPOINT    = "sam_vit_b_01ec64.pth"
CLIP_MODEL_NAME   = "openai/clip-vit-base-patch32"

# ---------- 服务端口 ----------
ONLINE_API_HOST = "0.0.0.0"
ONLINE_API_PORT = 8000

# ---------- Ray 运行时 ----------
RAY_ADDRESS            = None             # None=启动本地集群, "auto"=连接已有集群
RAY_NAMESPACE          = "inference"
ACTOR_MAX_RESTARTS     = 3                # 崩溃自动重启次数
ACTOR_MAX_TASK_RETRIES = 2

# ---------- 健康检查 ----------
HEALTH_CHECK_INTERVAL_SEC = 30
TASK_TIMEOUT_SEC          = 120

# ---------- 推理参数 ----------
DEFAULT_BATCH_SIZE     = 8
DEFAULT_IMAGE_SIZE     = 224
