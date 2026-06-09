# Ray 推理服务(CLIP + YOLOv8 + OneFormer)

单卡常驻三个模型,FastAPI 暴露同步 HTTP 接口。覆盖工业化要点:**资源切分、限流/超时、自愈/常驻、可观测(request_id/指标/慢请求)、可靠性(actor 句柄自动重连)**。容器化/K8s/CI 暂不涉及。

## 架构

```
main.py                入口(serve / bootstrap / teardown / status)
├── ray_deploy.py      Ray 初始化、Actor 创建/attach、健康巡检、装配 FastAPI
├── config.py          GPU/限流/超时/tmp 路径/版本号全部参数
├── utils/
│   ├── image_loader.py    path/URL/base64 三分支加载,URL 下载落盘 tmp/image/<日期>/
│   ├── logging_setup.py   setup_logger:按日期建 tmp/ray_log/<日期>/<name>.log
│   └── tmp_cleanup.py     启动时清理 N 天前的 tmp 子目录
├── models/
│   ├── base.py            Actor 基类(指标 + health_check + _error 写日志带 rid)
│   ├── clip_actor.py      CLIP zero-shot 分类(torch.compile)
│   ├── yolo_actor.py      YOLOv8 检测(不 compile,letterbox 与 dynamo 不兼容)
│   └── oneformer_actor.py OneFormer 分割(torch.compile dynamic=True)
├── services/
│   ├── schemas.py         请求体 Pydantic 模型(ClassifyBody/DetectBody/SegmentBody)
│   ├── online_api.py      FastAPI 端点(含 /healthz /readyz /version)
│   ├── middleware.py      request_id + 限流 + 慢请求 WARNING + 请求计数器
│   └── metrics.py         Prometheus 文本指标(含按 path/status 的 http_requests_total)
└── tmp/                   运行时产物,整目录 .gitignore
    ├── ray_log/<YYYYMMDD>/<name>.log   API/部署/三模型 actor 分别独立日志
    └── image/<YYYYMMDD>/<HHMMSS_md5>.<ext>   URL 下载落盘
```

## 启动方式

支持两种姿势:**常用一键启动**(进程绑定)和**热重载开发**(模型不重载,推荐)。

> **首次启动前必做一步**:`python main.py prepare` 把 CLIP + OneFormer 权重 snapshot 到项目内 `weights/`。
> 两个 HF 模型 actor 现在**强制本地加载**(`local_files_only=True`),`weights/` 缺失会直接抛错引导你跑 prepare。
> `prepare` 内部按 etag 去重,已存在的目录直接 `[skip]`,二次运行无害,加 `--force` 才强制重拉。
> YOLO 走 ultralytics 单文件(`yolov8n.pt` 在项目根),首次 actor 加载时自动从 GitHub 拉。

```bash
pip install -r requirements.txt
python main.py prepare                # 一次性,~1.8GB(CLIP ~600MB + OneFormer ~1.2GB)
```

### A. 一键启动(临时跑通 / 生产)

```bash
CUDA_VISIBLE_DEVICES=0 python main.py serve --port 7890
```

`serve` 内部:启动 Ray → 创建三个 detached actor(加载模型权重)→ 装配 FastAPI → uvicorn 运行。Ctrl+C 整套退出。

**这条命令会监听的端口**(单进程同时拉起 Ray 迷你集群 + API):

| 端口 | 谁开的 | 用途 | 暴露面 |
|---|---|---|---|
| **7890** | uvicorn | FastAPI(/classify、/detect、/segment、/metrics…) | `0.0.0.0`(对外) |
| 8265 | Ray dashboard | 浏览器看 actor / 任务 / 资源 | `127.0.0.1`(本机) |
| 6379 | Ray GCS | 集群元数据(Redis 协议),内部用 | 本机 |
| 10001 | Ray Client | 外部 driver 接入 Ray | 本机 |
| 随机端口 | Ray raylet / object manager / worker | 节点间通信(单机模式实际只本地用) | 本机 |

### B. 热重载开发(强烈推荐)

把 Ray 集群和 API 进程解耦:actor 常驻,只重启 uvicorn。OneFormer-large 加载一次 ~30s,这个差距很大。

```bash
# 0) 离线 + 跳过 HF Hub HEAD 验证(可选,显著加快 actor 启动)
export HF_HUB_OFFLINE=1                                # 模型已缓存时使用

# 1) 启 Ray 集群(一次性)
CUDA_VISIBLE_DEVICES=0 ray start --head --dashboard-host=127.0.0.1

# 2) 创建/确保 detached actor 就绪(一次性,模型权重在此加载)
python main.py bootstrap                               # 失败 exit 1,便于脚本检测

# 3) 启 API 进程,改代码自动重载
uvicorn ray_deploy:app --host 0.0.0.0 --port 7890 --reload --reload-dir services --reload-dir models
```

之后改 `services/` 或 `models/` 下的代码,uvicorn 秒级重启 API 进程,actor 不动。API 进程的 startup 钩子会 `ray.init(address="auto")` attach 集群,并 `ray.get_actor(name)` 拿到 bootstrap 阶段创建的 actor 句柄。

**B 方式分三步,各自监听的端口**:

| 命令 | 监听端口 | 说明 |
|---|---|---|
| `ray start --head --dashboard-host=127.0.0.1` | **6379**(GCS)+ **8265**(dashboard)+ **10001**(Ray Client)+ 随机 raylet/worker 端口 | 一次性,后续不动 |
| `python main.py bootstrap` | **不开新端口** | 只是通过已存在的 Ray Client(10001)创建/复用 detached actor;actor 在 Ray worker 进程里有自己的内部端口但不对外 |
| `uvicorn ray_deploy:app --port 7890 --reload` | **7890** | FastAPI 主端口;startup 钩子会自动 `ray.init(address="auto")` 复用已存在的 6379/10001 |

跟 A 相比的区别:Ray 端口是手工 `ray start` 起的,所以即使重启 uvicorn,6379/8265/10001 不会断,**actor 进程也不会被回收**。

**重要**:如果你改了 `models/*_actor.py` 的 `__init__` / `_load_model` / `infer` 签名,actor 进程没动 → 必须 `python main.py teardown && python main.py bootstrap` 才能生效。

收尾:

```bash
python main.py teardown        # 杀掉 detached actor(Ray 集群保留)
ray stop                       # 关闭整个 Ray 集群
```

### 何时选 A、何时选 B

| 场景 | 选 |
|---|---|
| 生产部署 / 单卡跑一遍验证 | A |
| 反复改 FastAPI 端点 / 业务逻辑 | B |
| 改了模型加载逻辑 / actor 签名 | B,但需重新 `teardown` + `bootstrap` |

## HuggingFace 模型加载

| 模型 | 加载方式 | 路径 |
|---|---|---|
| **CLIP** | **强制本地**(`local_files_only=True`,不读 hub) | `weights/clip-vit-base-patch32/`,由 `python main.py prepare` 填充 |
| **OneFormer** | **强制本地**(`local_files_only=True`,不读 hub) | `weights/oneformer_ade20k_swin_large/`,由 `python main.py prepare` 填充 |
| YOLO | ultralytics 单文件 | 项目根 `yolov8n.pt`,首次自动从 GitHub release 下载 |

两个 HF 模型必须本地化是因为 transformers 即使命中本地缓存也会向 hub 发 HEAD 验 etag,被匿名限速会卡住每次启动几秒到几十秒。**强制本地路径 + `local_files_only=True` 彻底绕开**;`weights/` 目录缺失或为空,actor 加载会显式抛错引导你跑 prepare,**不会静默回落到在线下载**。

prepare 重复运行安全:目录非空直接 `[skip]` 早返回;`snapshot_download` 内部也按 etag 校验逐文件去重,加 `--force` 才会强制重拉。

启动日志里的 `UNEXPECTED: relative_position_index` 和 `MISSING: swin.layernorm.*` 是 shi-labs 老 ckpt 与新版 transformers 字段差异,**可忽略**,不影响 ADE20K 推理精度。

## Dashboard 远程访问

启动方式 A 和 B 都会暴露 `127.0.0.1:8265` 的 Ray Dashboard(只绑回环,公网摸不到)。远程服务器看 Dashboard 用 SSH 隧道把本机 8265 透到笔记本:

```bash
ssh -L 8265:127.0.0.1:8265 user@server
```

改默认 dashboard 端口:`ray start --head --dashboard-host=127.0.0.1 --dashboard-port=9265`。

## 日志位置(全部在项目内,按日期分子目录)

| 来源 | 路径 | 内容 |
|---|---|---|
| FastAPI 进程 | `tmp/ray_log/<YYYYMMDD>/api.log` | 每条请求 rid/path/status/ms;>1000ms 自动 WARNING |
| 部署/巡检 | `tmp/ray_log/<YYYYMMDD>/deploy.log` | Ray init、actor 创建、健康循环 |
| 三个 actor | `tmp/ray_log/<YYYYMMDD>/{clip,yolo,oneformer}.log` | 各 actor 进程独立日志,出错带 rid 便于全链路追踪 |
| URL 下载图 | `tmp/image/<YYYYMMDD>/<HHMMSS>_<md5>.<ext>` | 客户端发 URL 时落盘 |
| Ray 系统 | `/tmp/ray/session_latest/logs/` | raylet / gcs / dashboard 等,Ray 自己管 |

启动时(`serve` 或 uvicorn startup)自动调用 `cleanup_tmp()` 删除 7 天前的日期目录,见 `config.TMP_CLEANUP_DAYS`。整个 `tmp/` 由 `tmp/.gitignore` 屏蔽提交。

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET`  | `/healthz`  | liveness probe,永远 200 |
| `GET`  | `/readyz`   | readiness probe,三 actor 全 alive 才 200,否则 503 |
| `GET`  | `/health`   | 详细诊断:每个 actor 完整 health_check |
| `GET`  | `/actors`   | actor 完整字段(device/gpu_fraction/last_inference 等) |
| `GET`  | `/version`  | `APP_VERSION` + git short commit + 三模型 repo |
| `GET`  | `/metrics`  | Prometheus 文本(含 `http_requests_total{path,status}`) |
| `POST` | `/classify` | CLIP zero-shot,需 `labels`(>=2 才有意义) |
| `POST` | `/detect`   | YOLOv8 检测,支持 `conf/iou/max_det/imgsz/classes` |
| `POST` | `/segment`  | OneFormer 分割,`task ∈ {instance,semantic,panoptic}`,`mask_format ∈ {png_b64,rle}` |

请求体见 `services/schemas.py`,`source` 支持本地路径 / URL / base64 / data URI。

```bash
# CLIP zero-shot:至少给 2 个 label,否则 softmax 必然 1.0
curl -X POST http://localhost:7890/classify \
  -H 'Content-Type: application/json' \
  -d '{"source":"https://example.com/cat.jpg","labels":["cat","dog","bird"],"top_k":2}'

# YOLO 检测,只看 person/car
curl -X POST http://localhost:7890/detect \
  -d '{"source":"/data/street.jpg","conf":0.3,"classes":[0,2]}'

# OneFormer 实例分割,返回 COCO RLE(比 PNG 小 5-10 倍)
curl -X POST http://localhost:7890/segment \
  -d '{"source":"/data/scene.jpg","task":"instance","return_mask":true,"mask_format":"rle"}'
```

文档:`http://localhost:7890/docs`。

## 全链路追踪(request_id)

每条请求会从 HTTP header 或自动生成一个 `X-Request-ID`,经中间件 → `request.state.rid` → `_infer_call` 注入 `kwargs["_rid"]` → actor `infer` 的 `_rid` 参数 → 出错时由 `BaseModelActor._error` 写入对应 actor 日志。

排查方法:

```bash
# 1) 客户端拿到的响应 header 里 X-Request-ID: 7a3b...
# 2) 在 API 日志找入口
grep "rid=7a3b" tmp/ray_log/20260609/api.log
# 3) 在对应 actor 日志找具体异常
grep "rid=7a3b" tmp/ray_log/20260609/oneformer.log
```

## 压力测试

```bash
python test/stress.py --endpoint all --concurrency 8 --duration 30
python test/stress.py --endpoint classify --ramp 1,2,4,8,16,32 --duration 10
```

## 工业化要点对照

| 维度 | 实现位置 |
|------|----------|
| GPU 配额 | `config.GPU_FRACTION_*` + `ray.remote(num_gpus=...)` |
| Actor 内并发 | `max_concurrency`(OneFormer 减半) |
| 入口限流 | `services/middleware.ConcurrencyLimitMiddleware`,超额 429 |
| 单请求超时 | `_await_with_timeout` + `ray.cancel` |
| Actor 自愈 | `max_restarts` + `lifetime="detached"` |
| 句柄自愈 | `RayActorError` 自动清缓存 + `ray.get_actor` 重连 + 重试一次 |
| request_id 全链路 | `RequestIDMiddleware` → `_infer_call` → actor `_rid` → `_error` 日志 |
| 慢请求告警 | `> SLOW_REQUEST_MS` 自动 WARNING |
| 指标 | `/metrics`(请求数/错误数/平均延迟/按 path&status 计数) |
| 探针 | `/healthz`(liveness) + `/readyz`(readiness) |
| 版本溯源 | `/version` 返回 git short commit |
| tmp 自清理 | 启动时删 `TMP_CLEANUP_DAYS` 天前的日期目录 |
| 图像大小硬限 | `MAX_IMAGE_BYTES`,URL 下载边下边校验 |
| Dashboard 安全 | `RAY_DASHBOARD_HOST=127.0.0.1` |

## 配置常用项(`config.py`)

| 配置 | 含义 |
|---|---|
| `GPU_FRACTION_{CLIP,YOLO,ONEFORMER}` | 三模型 GPU 配额(逻辑值) |
| `MAX_INFLIGHT_REQUESTS` | 入口并发上限,超过返回 429 |
| `INFER_TIMEOUT_SEC` | 单次推理硬超时,超时 504 + `ray.cancel` |
| `MAX_IMAGE_BYTES` | 输入图最大字节数(默认 20MB) |
| `SLOW_REQUEST_MS` | 慢请求 WARNING 阈值(默认 1000ms) |
| `TMP_CLEANUP_DAYS` | tmp 子目录保留天数(默认 7) |
| `APP_VERSION` | `/version` 端点返回 |
| `RAY_DASHBOARD_HOST` | 默认 `127.0.0.1`,生产严禁改 `0.0.0.0` |
