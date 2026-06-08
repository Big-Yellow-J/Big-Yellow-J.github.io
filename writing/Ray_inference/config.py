"""全局配置:模型、GPU 分配、服务端口、运行时参数。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODEL_CACHE_DIR = Path.home() / ".cache" / "ray_inference_models"

# GPU 分配(单卡,Ray 通过 num_gpus=fraction 调度)
GPU_FRACTION_RESNET = 0.15
GPU_FRACTION_YOLO = 0.30

# 模型权重
RESNET_MODEL_NAME = "resnet50"
YOLO_MODEL_NAME = "yolov8n.pt"

# 服务
ONLINE_API_HOST = "0.0.0.0"
ONLINE_API_PORT = 8000

# Ray 运行时
RAY_ADDRESS = None
RAY_NAMESPACE = "inference"
ACTOR_MAX_RESTARTS = 3
ACTOR_MAX_TASK_RETRIES = 2
ACTOR_MAX_CONCURRENCY = 4

# 健康检查 & 超时
HEALTH_CHECK_INTERVAL_SEC = 30
TASK_TIMEOUT_SEC = 120

# 输入限制
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB
URL_FETCH_TIMEOUT_SEC = 10
