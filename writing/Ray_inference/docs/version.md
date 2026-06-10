# 版本变化

记录每个发布版本的新增 / 变更 / 修复。`config.APP_VERSION` 与本文件版本号保持同步。

格式约定:
- 🆕 新增功能 / 端点 / 模型
- ✏️ 改进 / 重构(行为基本不变)
- ⚠️ 破坏性变更(需要重启 actor / 改客户端)
- 🐛 修复
- 🗑️ 移除

---

## v1.2(当前 · 2026-06-09)

性能 + 可观测两块小升级,P0 优先级。

### ✏️ 性能

- **CLIP / OneFormer / Qwen 全部切 fp16**(GPU 上,CPU 自动降回 fp32)
  - 显存:CLIP ~600MB → ~300MB,OneFormer ~3.5GB → ~1.8GB
  - 时延:CLIP / OneFormer 前向快 30~50%
  - 实现:`models.base.BaseModelActor` 加 `self._dtype` 与 `_cast_floats(inputs)` 辅助,actor 内 `from_pretrained(..., torch_dtype=self._dtype)` 并对 `pixel_values` 等浮点输入 cast
- **CLIP softmax 在 fp32 上做**,避免极小值产生 NaN

### 🆕 可观测

- **结构化 JSON 日志**:`LOG_FORMAT=json` 切换,extra={...} 字段会作为顶级 JSON 字段透传
  - 默认仍是 text(零行为变化),Loki / ES 直接抓取 stdout / 文件
  - `utils.logging_setup.JsonFormatter` 实现
- **`/metrics` 补直方图**:`http_request_duration_seconds_{bucket,sum,count}{path}`
  - 11 桶默认 Prometheus 间距(5ms / 10ms / 25ms / 50ms / 0.1s / 0.25s / 0.5s / 1s / 2.5s / 5s / 10s)+ `+Inf`
  - Grafana 可直接画 P50 / P95 / P99
- **`/metrics` 加 GPU 显存指标**:`ray_actor_gpu_memory_mb{model}`,每 actor 独立进程视角
- **`/health` / `/version` 增加 dtype 与 gpu_memory_mb 字段**

### ✏️ 中间件

- `RequestIDMiddleware` 调用日志用 `extra={"event":"http_request","rid","path","status","ms"}` 透传字段,JSON 模式下能字段化检索

---

## v1.1(2026-06-09)

新增单机 Docker 部署能力,代码改动极小(`config.py` 三个端口/地址改成读环境变量),其余通过新增编排文件覆盖。

### 🆕 部署

- **Dockerfile**:基于 `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`,`tini` 做 PID1 转发 SIGTERM
- **docker-compose.yml**:GPU + `shm_size: 4gb` + 三份 volume(weights/data/tmp) + `/readyz` healthcheck + `stop_grace_period: 45s` 配合优雅退出
- **entrypoint.sh**:权重缺失自动 `prepare` 后 `exec python main.py serve`,保留信号链
- **.dockerignore**:排除 weights/data/tmp/.git,镜像不打权重(~13GB 而非 ~20GB)
- **.env.example**:列出所有可配环境变量样板

### ✏️ 配置

- `config.ONLINE_API_HOST / ONLINE_API_PORT / RAY_DASHBOARD_HOST / RAY_ADDRESS` 改读环境变量,默认值不变 → **裸跑零影响**

### 📝 文档

- `README.md` 新增"方式 C:Docker 部署"小节
- `docs/introduction.md` 新增:1.2.1 Docker 常用命令、2.5 Docker 排错(6 个症状)、3.8 镜像构建/升级/切环境

---

## v1.0(2026-06-09)

第一个完整发布版本。把"GPU 多模型推理 + 向量检索 + 在线 HTTP 服务"拼装成可工业化使用的最小样本。

### 🆕 模型与端点

- **CLIP**(`openai/clip-vit-base-patch32`):zero-shot 分类 `/classify`、图 embed `/v1/embed`、文本 embed `/v1/embed_text`
- **YOLOv8**(`yolov8n.pt`):目标检测 `/detect`
- **OneFormer**(`shi-labs/oneformer_ade20k_swin_large`):通用分割 `/segment`,支持 instance/semantic/panoptic + PNG/RLE mask 格式
- **Qwen3-VL-Embedding-2B**:第二编码器,与 CLIP 并列,各自独立 collection
- **向量检索**:`/v1/search`(以图搜图)、`/v1/search_text`(CLIP 跨模态以文搜图)

### 🆕 基础设施

- **Ray Actor 体系**:四模型常驻 GPU,`lifetime="detached"`,`max_restarts=3` 自愈
- **句柄自愈**:`RayActorError` 自动 `ray.get_actor` 重连 + 重试一次
- **熔断器**:10s 内连续 5 次失败 → 30s 内直接 503,避免雪崩
- **入口限流**:`MAX_INFLIGHT_REQUESTS=32`,超过 429
- **单请求超时**:`INFER_TIMEOUT_SEC=30` + `ray.cancel`
- **优雅退出**:`SIGTERM` 等 `SHUTDOWN_GRACE_SEC=30` inflight 完成才杀 actor
- **请求级缓存**:embed 类 `(model, source_hash) → vector` LRU 缓存

### 🆕 模型本地化

- 所有 HF 模型 `local_files_only=True` 强制本地加载
- `python main.py prepare` 一键 snapshot 所有权重到 `weights/`
- **revision 锁定**:`config.CLIP_REVISION / ONEFORMER_REVISION / QWEN_EMBED_REVISION` 支持 pin sha,保证可复现

### 🆕 向量库

- **Milvus Lite**(`data/milvus_lite.db`)零部署,环境变量切到 standalone
- 每个 embedding 模型独立 collection(`embeddings_clip`、`embeddings_qwen_vl`)
- 启动自动备份到 `data/backup/`,保留最新 `MILVUS_BACKUP_KEEP=7` 份
- `python main.py milvus {stats,backup,list-backups,drop}` 运维命令

### 🆕 可观测

- `/healthz`(liveness)、`/readyz`(readiness)、`/health`(详细诊断)三层探针
- `/version` 端点返回版本 / git commit / 各模型 revision / 熔断 / 缓存状态
- `/metrics` Prometheus 文本(含 `http_requests_total{path,status}`)
- request_id 全链路:HTTP header → middleware → kwargs["_rid"] → actor 日志
- 慢请求自动 WARNING(`SLOW_REQUEST_MS=1000`)
- 各 actor 独立日志文件,按日期分子目录:`tmp/ray_log/<YYYYMMDD>/<name>.log`

### 🆕 安全

- 图像炸弹防护:`MAX_IMAGE_PIXELS=8192²`,加载前 `verify()` 二次校验
- 输入大小硬限:`MAX_IMAGE_BYTES=20MB`,URL 边下边校验

### 🆕 测试

- pytest 套件覆盖 `image_loader / dispatch / milvus / schemas` 四块,共 30+ 测试

### 🆕 启动方式

- 方式 A(一键):`python main.py serve --port 7890`
- 方式 B(热重载开发):`ray start --head` → `bootstrap` → `uvicorn --reload`

### ⚠️ 设计取舍(本版本明确不做)

- ❌ batch 端点(actor `max_concurrency` + 客户端并发已能拿到吞吐)
- ❌ API 鉴权(本地/内网使用,无公网风险)
- ❌ Docker / K8s / CI(本阶段聚焦应用层 · **v1.1 已补 Docker**)
- ❌ DNS rebinding / SSRF 防护(requests 直拉,接受风险)

---

## 后续版本规划(草案)

详细 backlog 见 [introduction.md 的"路线图"](introduction.md#路线图)。
