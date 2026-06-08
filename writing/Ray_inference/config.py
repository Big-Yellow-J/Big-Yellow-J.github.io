"""全局配置:模型、GPU 分配、服务端口、安全/限流参数。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODEL_CACHE_DIR = Path.home() / ".cache" / "ray_inference_models"

# GPU 分配(num_gpus 是 Ray 调度配额,不是物理隔离)
GPU_FRACTION_CLIP = 0.20
GPU_FRACTION_YOLO = 0.20
GPU_FRACTION_ONEFORMER = 0.50

# 模型权重(HuggingFace repo id 或 本地路径)
CLIP_MODEL = "openai/clip-vit-base-patch32"
YOLO_MODEL = "yolov8n.pt"
ONEFORMER_MODEL = "shi-labs/oneformer_ade20k_swin_large"

# 服务
ONLINE_API_HOST = "0.0.0.0"
ONLINE_API_PORT = 8000

# Ray 运行时
RAY_ADDRESS = None
RAY_NAMESPACE = "inference"
RAY_DASHBOARD_HOST = "127.0.0.1"   # 安全:dashboard 不要暴露公网
ACTOR_MAX_RESTARTS = 3
ACTOR_MAX_TASK_RETRIES = 2
ACTOR_MAX_CONCURRENCY = 4

# SLO:入口并发上限 + 单请求硬超时
MAX_INFLIGHT_REQUESTS = 32
INFER_TIMEOUT_SEC = 30

# 健康巡检
HEALTH_CHECK_INTERVAL_SEC = 30

# 输入限制 + SSRF 防护
MAX_IMAGE_BYTES = 20 * 1024 * 1024
URL_FETCH_TIMEOUT_SEC = 10
URL_ALLOWED_SCHEMES = ("http", "https")
URL_ALLOW_PRIVATE_NETWORK = False   # 生产保持 False,禁止访问内网/回环
