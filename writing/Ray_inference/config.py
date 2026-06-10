"""全局配置:模型、GPU 分配、服务端口、限流、tmp 路径、向量库。"""
import os
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
GPU_FRACTION_QWEN_EMBED = 0.30

# 模型权重:HF 类模型一律本地化,放项目内 weights/,启动跳过 hub HEAD 验证。
# 首次必须 `python main.py prepare` 把权重 snapshot 到本地;之后 actor 一律本地加载。
WEIGHTS_DIR = PROJECT_ROOT / "weights"

CLIP_REPO = "openai/clip-vit-base-patch32"
CLIP_LOCAL_DIR = WEIGHTS_DIR / "clip-vit-base-patch32"
CLIP_MODEL = str(CLIP_LOCAL_DIR)

ONEFORMER_REPO = "shi-labs/oneformer_ade20k_swin_large"
ONEFORMER_LOCAL_DIR = WEIGHTS_DIR / "oneformer_ade20k_swin_large"
ONEFORMER_MODEL = str(ONEFORMER_LOCAL_DIR)

# Qwen VL Embedding(图像 → 向量,与 CLIP 并列的可选 embedder)
QWEN_EMBED_REPO = "Qwen/Qwen3-VL-Embedding-2B"
QWEN_EMBED_DOWNLOAD_FROM = "hf" # 直接从 ModelScope 下载,不走 HF Hub;同样放在 weights/ 下管理 注意hf和modelscope上名称一致性
QWEN_EMBED_LOCAL_DIR = WEIGHTS_DIR / "Qwen3-VL-Embedding-2B"
QWEN_EMBED_MODEL = str(QWEN_EMBED_LOCAL_DIR)

# YOLO 走 ultralytics 单文件(GitHub release,不经 HF Hub),首次跑会自动下到当前目录
YOLO_MODEL = "yolov8n.pt"

# Data 目录(Milvus Lite 文件 + 其他持久数据,与 tmp/ 区分:data 不清理,tmp 7 天)
DATA_DIR = PROJECT_ROOT / "data"

# Milvus 向量库:默认走 Lite(本地文件,无需 server),环境变量切到 standalone
MILVUS_URI = os.getenv("MILVUS_URI", str(DATA_DIR / "milvus_lite.db"))
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_COLLECTION_PREFIX = "embeddings"   # 实际 collection = embeddings_<model>
MILVUS_METRIC = "IP"                       # 归一化向量上的内积 = 余弦相似度

# Embedding 默认模型
EMBEDDER_DEFAULT = "clip"

# 服务(容器化场景 ONLINE_API_PORT 通过 env 注入,与 docker-compose 端口映射对齐)
ONLINE_API_HOST = os.getenv("ONLINE_API_HOST", "0.0.0.0")
ONLINE_API_PORT = int(os.getenv("ONLINE_API_PORT", "8000"))

# Ray 运行时(dashboard host 容器内需绑 0.0.0.0 才能从宿主机访问)
RAY_ADDRESS = os.getenv("RAY_ADDRESS") or None
RAY_NAMESPACE = "inference"
RAY_DASHBOARD_HOST = os.getenv("RAY_DASHBOARD_HOST", "127.0.0.1")
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
MAX_IMAGE_PIXELS = 8192 * 8192          # PIL 解压炸弹防护:超过抛 DecompressionBombError
URL_FETCH_TIMEOUT_SEC = 10

# 模型权重 revision sha 锁定(留空 = 跟主分支,生产请填具体 commit)
# 从 HuggingFace 模型页 "Files and versions" 拷 commit hash
CLIP_REVISION = os.getenv("CLIP_REVISION", "")
ONEFORMER_REVISION = os.getenv("ONEFORMER_REVISION", "")
QWEN_EMBED_REVISION = os.getenv("QWEN_EMBED_REVISION", "")

# 优雅退出:SIGTERM 后等待 inflight 完成的最长时间(秒)
SHUTDOWN_GRACE_SEC = 30

# 熔断器:N 秒内连续 M 次失败 → 开熔断 K 秒
CIRCUIT_WINDOW_SEC = 10
CIRCUIT_FAIL_THRESHOLD = 5
CIRCUIT_OPEN_SEC = 30

# 请求级缓存(进程内 LRU,同 source+model 命中跳过 actor)
EMBED_CACHE_SIZE = 1024

# Milvus 自动备份:启动时备份并保留 N 个最新副本(0 = 不自动备份)
MILVUS_BACKUP_KEEP = 7
