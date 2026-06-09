"""全局配置:模型、GPU 分配、服务端口、限流、tmp 路径。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODEL_CACHE_DIR = Path.home() / ".cache" / "ray_inference_models"

# tmp 目录:统一在项目内,按日期分子目录
TMP_DIR = PROJECT_ROOT / "tmp"
TMP_LOG_DIR = TMP_DIR / "ray_log"      # tmp/ray_log/<YYYYMMDD>/<name>.log
TMP_IMAGE_DIR = TMP_DIR / "image"      # tmp/image/<YYYYMMDD>/<HHMMSS>_<hash>.<ext>
TMP_CLEANUP_DAYS = 7                   # 启动时删除 N 天前的日期子目录

# 应用版本
APP_VERSION = "3.2.0"

# GPU 分配(num_gpus 是 Ray 调度配额,不是物理隔离)
GPU_FRACTION_CLIP = 0.20
GPU_FRACTION_YOLO = 0.20
GPU_FRACTION_ONEFORMER = 0.50

# 模型权重:HF 类模型一律本地化,放项目内 weights/,启动跳过 hub HEAD 验证。
# 首次必须 `python main.py prepare` 把权重 snapshot 到本地;之后 actor 一律本地加载。
WEIGHTS_DIR = PROJECT_ROOT / "weights"

CLIP_REPO = "openai/clip-vit-base-patch32"
CLIP_LOCAL_DIR = WEIGHTS_DIR / "clip-vit-base-patch32"
CLIP_MODEL = str(CLIP_LOCAL_DIR)

ONEFORMER_REPO = "shi-labs/oneformer_ade20k_swin_large"
ONEFORMER_LOCAL_DIR = WEIGHTS_DIR / "oneformer_ade20k_swin_large"
ONEFORMER_MODEL = str(ONEFORMER_LOCAL_DIR)

# YOLO 走 ultralytics 单文件(GitHub release,不经 HF Hub),首次跑会自动下到当前目录
YOLO_MODEL = "yolov8n.pt"

# 服务
ONLINE_API_HOST = "0.0.0.0"
ONLINE_API_PORT = 8000

# Ray 运行时
RAY_ADDRESS = None
RAY_NAMESPACE = "inference"
RAY_DASHBOARD_HOST = "127.0.0.1"
ACTOR_MAX_RESTARTS = 3
ACTOR_MAX_TASK_RETRIES = 2
ACTOR_MAX_CONCURRENCY = 4

# SLO:入口并发上限 + 单请求硬超时 + 慢请求阈值
MAX_INFLIGHT_REQUESTS = 32
INFER_TIMEOUT_SEC = 30
SLOW_REQUEST_MS = 1000                 # 超过这个延迟在 access log 里以 WARNING 级别打

# 健康巡检
HEALTH_CHECK_INTERVAL_SEC = 30

# 输入限制 + URL 下载超时
MAX_IMAGE_BYTES = 20 * 1024 * 1024
URL_FETCH_TIMEOUT_SEC = 10
