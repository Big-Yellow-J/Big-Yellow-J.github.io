# Ray 推理服务

CLIP + YOLOv8 + OneFormer + Qwen-Embedding 四模型常驻 GPU,FastAPI 暴露同步 HTTP 接口,Milvus Lite 内置向量库做以图/以文搜图。**仅本地/内网部署**,无公网鉴权层。

> 更详细的内容:
> - 版本变化 → [docs/version.md](docs/version.md)
> - 运维/排错/升级 → [docs/introduction.md](docs/introduction.md)

---

## 项目描述

一个把"GPU 重模型 + 在线 HTTP 服务 + 向量检索"工业化拼装起来的最小可用样本。设计上把**计算重型**和**IO 轻型**分两条路:

- **计算**(模型推理)→ Ray Actor 常驻 GPU,只通过 `ray.remote` 调用
- **IO**(向量库、日志、文件)→ FastAPI 进程内直接做,不经 Actor

这套布局想解决的问题:
- 多模型共用单卡,显存配额可控(`ray.remote(num_gpus=...)`)
- 改业务代码不重载模型权重(`uvicorn --reload` 重启 API,actor 不动)
- Actor 崩溃自动重建 + 业务端 RPC 失效自愈
- 模型 embedding 顺手入库,后续切模型不丢历史数据

---

## 架构

```
ray_inference/
├── main.py                入口:prepare / serve / bootstrap / teardown / status / milvus
├── ray_deploy.py          Ray 初始化、Actor 创建/attach、健康巡检、装配 FastAPI
├── config.py              GPU/限流/超时/tmp/data/向量库/熔断/缓存 全部参数
├── utils/
│   ├── image_loader.py        path/URL/base64 三分支加载,verify + 像素上限防炸弹
│   ├── logging_setup.py       setup_logger:按日期建 tmp/ray_log/<日期>/<name>.log
│   ├── tmp_cleanup.py         启动清理 N 天前 tmp 子目录
│   ├── milvus_backup.py       Lite db 备份到 data/backup/,保留最新 N 个
│   └── prepare_models.py      snapshot HF 权重(支持 revision 锁定)
├── models/
│   ├── base.py                Actor 基类(指标 + health_check + _error 带 rid)
│   ├── clip_actor.py          CLIP 分类 / 图 embed / 文本 embed
│   ├── yolo_actor.py          YOLOv8 检测
│   ├── oneformer_actor.py     OneFormer 分割(instance/semantic/panoptic)
│   └── qwen_embed_actor.py    Qwen3-VL-Embedding 图像 embedding
├── services/
│   ├── schemas.py             所有请求体 Pydantic 模型
│   ├── middleware.py          request_id + 限流 + inflight 计数 + 慢请求 WARNING
│   ├── dispatch.py            actor 调用层(句柄缓存 + 自愈 + 熔断)
│   ├── metrics.py             Prometheus 文本指标
│   ├── online_api.py          FastAPI 端点 + lifecycle
│   ├── db/
│   │   └── milvus.py              Milvus Lite 客户端单例 + collection 管理
│   └── routers/
│       └── embed.py               /v1/embed、/v1/embed_text、/v1/search、/v1/search_text
├── tests/                  pytest 套件(image_loader / dispatch / milvus / schemas)
├── docs/                   版本变化 + 运维文档
├── Dockerfile              CUDA 12.1 runtime 镜像
├── .dockerignore           排除 weights/data/tmp/.git 不打进镜像
├── entrypoint.sh           容器入口:权重缺失自动 prepare 后 exec serve
├── docker-compose.yml      单机编排:GPU + shm + volume + healthcheck
├── .env.example            环境变量样板
├── weights/                HF 模型本地权重(.gitignore)
├── data/                   Milvus Lite db + backup(.gitignore)
└── tmp/                    运行时日志和图片下载缓存(.gitignore)
```

### 关键设计点

| 维度 | 落地 |
|---|---|
| GPU 配额 | `config.GPU_FRACTION_*` + `ray.remote(num_gpus=...)` |
| 模型本地化 | `weights/` 强制 `local_files_only=True`,跳过 hub HEAD 验证 |
| Actor 进程隔离 | 每个 actor 独立进程,独立日志(`tmp/ray_log/<日期>/<name>.log`) |
| 入口限流 | `MAX_INFLIGHT_REQUESTS=32`,超过 429 |
| 单请求超时 | `INFER_TIMEOUT_SEC=30` + `ray.cancel` |
| Actor 自愈 | `max_restarts=3` + `lifetime="detached"` |
| 句柄自愈 | `RayActorError` 自动 `ray.get_actor` 重连 + 重试一次 |
| 熔断器 | 10s 内连续 5 次失败 → 30s 内直接 503 |
| 慢请求 | `> SLOW_REQUEST_MS=1000` 自动 WARNING |
| request_id 全链路 | header → middleware → kwargs["_rid"] → actor 日志 |
| 优雅退出 | `SIGTERM` 等 `SHUTDOWN_GRACE_SEC=30` inflight 完成才杀 actor |
| 请求级缓存 | embed 类调用 LRU 缓存 `(model, source_hash) → vector` |
| 向量库 | Milvus Lite 单文件,每模型独立 collection,启动自动备份 |

---

## 服务启动

### 0. 安装依赖 + 下载模型(首次必做)

```bash
pip install -r requirements.txt
python main.py prepare                 # 一次性,下载 CLIP + OneFormer + Qwen 到 weights/
```

`prepare` 内部按 etag 去重,目录非空直接 `[skip]`;加 `--force` 才重拉。
首次约 **~6GB**(CLIP 0.6GB + OneFormer 1.2GB + Qwen 4GB)。

### 方式 A:一键启动(生产 / 临时跑通)

```bash
CUDA_VISIBLE_DEVICES=0 python main.py serve --port 7890
```

一个进程拉起 Ray 迷你集群 + 4 个 actor + FastAPI。Ctrl+C 整套退出(`SIGTERM` 会等 inflight)。

### 方式 B:热重载开发(强烈推荐)

把 Ray 集群和 API 进程解耦:actor 常驻,只重启 uvicorn。OneFormer-large 加载一次 ~30s,这个差距很大。

```bash
# 1) 启 Ray 集群(一次性)
CUDA_VISIBLE_DEVICES=0 ray start --head --dashboard-host=127.0.0.1 --dashboard-port=8265
# 2) 创建/确保 detached actor 就绪(一次性,模型权重在此加载)
python main.py bootstrap
# 3) 启 API 进程,改代码自动重载
uvicorn ray_deploy:app --host 0.0.0.0 --port 7890 --reload --reload-dir services --reload-dir models
```

之后改 `services/` 或 `models/` 下的代码,uvicorn 秒级重启 API 进程,**模型权重不会重新加载**。

收尾:

```bash
python main.py teardown        # 杀掉 detached actor(Ray 集群保留)
ray stop                       # 关闭整个 Ray 集群
```

> 改了 `models/*_actor.py` 的 `__init__` / `_load_model` / actor 方法签名 → 必须 `teardown && bootstrap`,uvicorn `--reload` 不会重启 actor 进程。

### 方式 C:Docker 部署(单机生产)

需要宿主机已装 NVIDIA Driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。

```bash
# 1) 准备环境变量
cp .env.example .env                   # 按需改 ONLINE_API_PORT / MILVUS_URI / REVISION

# 2) 构建 + 启动(首次会自动 prepare 权重,~6GB 数分钟)
docker compose up -d

# 3) 看日志确认 actor 就绪(看到 "API http://...:7890/docs" 即成功)
docker compose logs -f ray-inference

# 4) 健康检查
curl http://localhost:7890/readyz
```

**关键点**:`weights/`、`data/`、`tmp/` 全部 volume 挂载,容器重建不丢权重也不丢 Milvus 数据;`stop_grace_period: 45s` 配合代码里 30s grace,SIGTERM 后 inflight 不会被截断;`shm_size: 4gb` 给 Ray plasma object store。

收尾:

```bash
docker compose down                    # 停服务(数据全保留)
docker compose down -v                 # 连 volume 一起清(慎用)
```

> Dockerfile 不打权重(`.dockerignore` 排除 `weights/`),镜像保持精简 ~13GB(CUDA 8GB + pip 5GB)。

### 启动后监听的端口

| 端口 | 服务 | 暴露 |
|---|---|---|
| **7890** | FastAPI(可改) | `0.0.0.0` |
| 8265 | Ray Dashboard | `127.0.0.1`(仅本机,远程用 SSH 隧道) |
| 6379 | Ray GCS | 本机 |
| 10001 | Ray Client | 本机 |

`ssh -L 8265:127.0.0.1:8265 user@server` 在笔记本看 dashboard。
Docker 模式下 dashboard 默认不映射到宿主机,需要的话取消 `docker-compose.yml` 里 `127.0.0.1:8265:8265` 那行注释。

---

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET`  | `/healthz`           | liveness probe,永远 200 |
| `GET`  | `/readyz`            | readiness probe,actor + milvus 全 OK 才 200 |
| `GET`  | `/health`            | 详细诊断:每个 actor 完整 health_check |
| `GET`  | `/actors`            | actor 完整字段(device / gpu_fraction / last_inference) |
| `GET`  | `/version`           | 版本号 + git short commit + 各模型 repo&revision + 熔断 / 缓存状态 |
| `GET`  | `/metrics`           | Prometheus 文本(含 `http_requests_total{path,status}`) |
| `POST` | `/classify`          | CLIP zero-shot 分类 |
| `POST` | `/detect`            | YOLOv8 目标检测 |
| `POST` | `/segment`           | OneFormer 分割(instance / semantic / panoptic) |
| `POST` | `/v1/embed`          | 图像 → 向量(默认 CLIP,可选 `qwen_vl`),默认顺手写库 |
| `POST` | `/v1/embed_text`     | 文本 → 向量(CLIP) |
| `POST` | `/v1/search`         | 以图搜图 |
| `POST` | `/v1/search_text`    | 以文搜图(CLIP 文/图共享向量空间) |

请求体定义见 `services/schemas.py`,`source` 字段支持本地路径 / URL / base64 / data URI。

### 调用示例

```bash
# CLIP 分类(labels 至少 2 个)
curl -X POST http://localhost:7890/classify \
  -d '{"source":"/data/cat.jpg","labels":["cat","dog","bird"]}'

# YOLO 检测,只看 person/car
curl -X POST http://localhost:7890/detect \
  -d '{"source":"/data/street.jpg","conf":0.3,"classes":[0,2]}'

# OneFormer 实例分割,RLE mask
curl -X POST http://localhost:7890/segment \
  -d '{"source":"/data/scene.jpg","task":"instance","return_mask":true,"mask_format":"rle"}'

# 图像入库(CLIP)
curl -X POST http://localhost:7890/v1/embed \
  -d '{"source":"/data/cat.jpg","metadata":{"url":"...","tags":["pet"]}}'

# 以文搜图
curl -X POST http://localhost:7890/v1/search_text \
  -d '{"text":"a red sports car","top_k":5}'
```

Swagger 文档:`http://localhost:7890/docs`。

---

## 测试

```bash
pytest tests/ -v
```

覆盖:`image_loader`(4 种输入分支 + 大小/损坏图防护)、`dispatch`(句柄自愈 + 熔断)、`milvus`(用临时 Lite db 跑真实集成)、`schemas`(Pydantic 边界)。
