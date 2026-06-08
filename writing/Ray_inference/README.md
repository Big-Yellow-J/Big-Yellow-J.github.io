# Ray 推理服务(CLIP + YOLOv8 + OneFormer)

单卡常驻三个模型,FastAPI 暴露同步 HTTP 接口。设计点覆盖工业化要点 1–5:**资源切分、SLO(限流/超时)、可靠性(自愈/常驻)、可观测性(request_id/指标)、安全(SSRF/Dashboard 内网)**。部署(容器化/K8s/CI)暂不涉及。

## 架构

```
main.py                入口(serve / bootstrap / teardown / status)
├── ray_deploy.py      Ray 初始化、Actor 创建/attach、健康巡检、FastAPI 装配
├── config.py          GPU/限流/超时/SSRF 全部参数
├── models/
│   ├── base.py            Actor 基类(指标 + health_check)
│   ├── image_loader.py    path/URL/base64 统一加载 + SSRF 防护 + verify
│   ├── clip_actor.py      CLIP zero-shot 分类
│   ├── yolo_actor.py      YOLOv8 检测
│   └── oneformer_actor.py OneFormer 分割(instance/semantic/panoptic)
└── services/
    ├── online_api.py      FastAPI 端点
    ├── middleware.py      request_id + 入口限流
    └── metrics.py         Prometheus 文本指标
```

## 启动方式

支持两种姿势:**常用一键启动**(进程绑定)和**热重载开发**(进程解耦,改代码不重载模型)。

### A. 常用一键启动(生产 / 临时跑通)

```bash
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python main.py serve --port 7890
```

`main.py serve` 内部依次:启动 Ray → 创建三个 detached actor(自动加载模型权重)→ 装配 FastAPI → uvicorn 运行。一个进程包圆,Ctrl+C 整套退出。

### B. 热重载开发(改代码不重载模型,推荐)

把 Ray 集群和 API 进程解耦:Ray actor 常驻不动,只重启 uvicorn。三步:

```bash
# 1) 启 Ray 集群(一次性,后续不动)
CUDA_VISIBLE_DEVICES=0 ray start --head --dashboard-host=127.0.0.1

# 2) 创建/确保 detached actor 就绪(一次性,模型权重在此加载)
python main.py bootstrap

# 3) 启 API 进程,代码改动自动重载(模型不会被重载)
uvicorn ray_deploy:app --host 0.0.0.0 --port 7890 \
    --reload --reload-dir services --reload-dir models
```

之后改 `services/` 或 `models/` 下的代码,uvicorn 秒级重启 API 进程,但 actor 进程不动 —— **模型权重无需重新加载**(OneFormer-large 加载一次 ~30s,这个差距很大)。

API 进程的 FastAPI startup hook 会自动 `ray.init(address="auto")` attach 已存在的集群,并通过 `ray.get_actor(name)` 拿到 bootstrap 阶段创建的 actor 句柄。

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
| 改了 `models/*_actor.py` 的模型加载逻辑 | B 也要 `teardown` + `bootstrap`,因为 actor 进程没动 |

## 端口与监控

| 端口 | 服务 | 说明 |
|---|---|---|
| **7890** | FastAPI API(可改) | `--port` 指定 |
| **8265** | Ray Dashboard | `--dashboard-host=127.0.0.1` 限本机 |
| 6379 | GCS(Redis 协议) | 内部元数据,不要暴露 |
| 10001 | Ray Client | 外部 driver 接入用 |
| 10002–19999 | worker 间通信 | 随机分配,集群内部 |

**远程服务器看 Dashboard**:本地终端建 SSH 隧道,然后开浏览器 `http://localhost:8265`:

```bash
ssh -L 8265:127.0.0.1:8265 user@server
```

**自定义端口**:

```bash
ray start --head --dashboard-host=127.0.0.1 --dashboard-port=9265
```

**日志位置**:

- `python main.py serve` / `uvicorn ...` 的日志 → 启动终端的 stdout/stderr
- Actor 内部日志(每个模型独立) → `/tmp/ray/session_latest/logs/worker-*.out|.err`
- Ray 系统日志(raylet/gcs/dashboard) → `/tmp/ray/session_latest/logs/`
- Dashboard 也能在浏览器看 actor 实时日志:`http://localhost:8265` → Actors

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET`  | `/health`   | 全模型健康摘要 |
| `GET`  | `/actors`   | actor 详细字段 |
| `GET`  | `/metrics`  | Prometheus 文本格式 |
| `POST` | `/classify` | CLIP zero-shot,需要 `labels` |
| `POST` | `/detect`   | YOLOv8 检测 |
| `POST` | `/segment`  | OneFormer 分割,`task ∈ {instance, semantic, panoptic}` |

请求体统一 JSON,`source` 支持本地路径 / URL / base64 / data URI。

```bash
curl -X POST http://localhost:7890/classify \
  -H 'Content-Type: application/json' \
  -d '{"source":"https://example.com/cat.jpg","labels":["cat","dog","bird"],"top_k":2}'

curl -X POST http://localhost:7890/segment \
  -H 'Content-Type: application/json' \
  -d '{"source":"/data/scene.jpg","task":"instance"}'
```

文档:`http://localhost:7890/docs`。

## 压力测试

`test/stress.py` 配合 `test/image/`(本地图)和 `test/test_image.txt`(URL 列表):

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
| request_id | `RequestIDMiddleware`,透传 `X-Request-ID` |
| 指标 | `/metrics`(请求数/错误数/平均延迟) |
| SSRF 防护 | `image_loader._fetch_url`:scheme 白名单 + 显式私网 CIDR + 手动重定向校验 |
| 图像解码校验 | `Image.verify()` 防恶意构造 |
| Dashboard 安全 | `RAY_DASHBOARD_HOST=127.0.0.1` |

## 配置常用项

`config.py`:

- `GPU_FRACTION_CLIP / YOLO / ONEFORMER` —— 三模型 GPU 配额(逻辑值)
- `MAX_INFLIGHT_REQUESTS` —— 入口并发上限,超过返回 429
- `INFER_TIMEOUT_SEC` —— 单次推理硬超时,超时 504 + `ray.cancel`
- `MAX_IMAGE_BYTES` —— 输入图最大字节数
- `URL_ALLOW_PRIVATE_NETWORK` —— 默认 False,生产严禁置 True

## HuggingFace 模型下载

OneFormer-large 首次下载较大(~1 GB),未配 token 会限速。三选一:

```bash
export HF_ENDPOINT=https://hf-mirror.com    # 镜像(国内首选)
# 或
export HF_TOKEN=hf_xxxxxxxx                 # 官方 token
# 或
huggingface-cli download shi-labs/oneformer_ade20k_swin_large
huggingface-cli download openai/clip-vit-base-patch32
export HF_HUB_OFFLINE=1                     # 后续完全离线
```
