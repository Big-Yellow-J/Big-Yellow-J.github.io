# 运维与扩展手册

按"日常运维 → 排错 runbook → 升级 / 扩展 → 路线图"组织。第一次接手项目从上往下读;真出问题直接跳到对应症状。

---

## 1. 日常运维

### 1.1 目录布局速记

| 目录 | 用途 | 是否入 git |
|---|---|---|
| `weights/` | HF 模型权重(prepare 下载) | 否 |
| `data/` | Milvus Lite db + 自动备份 | 否 |
| `data/backup/` | Lite db 历史副本(自动保留 7 份) | 否 |
| `tmp/ray_log/<YYYYMMDD>/` | API / deploy / 各 actor 独立日志 | 否 |
| `tmp/image/<YYYYMMDD>/` | URL 下载到本地的图像缓存 | 否 |
| `docs/` | 本文档 + version.md | 是 |
| `tests/` | pytest 套件 | 是 |

启动时自动:
- `cleanup_tmp(days=TMP_CLEANUP_DAYS=7)` 删 7 天前的日期子目录
- `backup_lite()` 复制 Lite db 到 `data/backup/`,保留最新 `MILVUS_BACKUP_KEEP=7` 份

### 1.2 常用命令

```bash
# 模型权重
python main.py prepare                  # 首次/补漏(已存在跳过)
python main.py prepare --force          # 强制重拉(revision 改了时用)

# 服务生命周期
python main.py serve --port 7890        # 一键启动(开发简单)
python main.py bootstrap                # 仅创建/复用 actor,失败 exit 1
python main.py teardown                 # 杀掉所有 detached actor
python main.py status                   # 查看每个 actor 健康

# Milvus 运维
python main.py milvus stats             # 各 collection 行数 / 维度
python main.py milvus backup            # 手动触发一次备份
python main.py milvus list-backups      # 列出所有备份
python main.py milvus drop clip         # 删除 embeddings_clip(需输入 YES 确认)

# 测试
pytest tests/ -v                        # 全量(不需要 actor / 模型)
pytest tests/test_milvus.py -v          # 只跑 milvus 集成
```

### 1.2.1 Docker 模式常用命令

```bash
# 生命周期
docker compose up -d                    # 启动(detach)
docker compose down                     # 停止(保留 volume)
docker compose restart ray-inference    # 重启服务

# 观察
docker compose ps                       # 看健康状态(starting / healthy / unhealthy)
docker compose logs -f ray-inference    # 跟 entrypoint + uvicorn + ray 合并日志
docker compose exec ray-inference bash  # 进容器内排查

# 容器内运行 CLI 工具(权重/Milvus 维护)
docker compose exec ray-inference python main.py status
docker compose exec ray-inference python main.py milvus stats
docker compose exec ray-inference python main.py prepare --force

# 镜像
docker compose build --no-cache         # 改了 requirements.txt 之后强制重建
docker images ray-inference             # 看镜像大小/tag
```

### 1.3 日志位置速查

| 找什么 | 在哪 |
|---|---|
| HTTP 请求 / 慢请求 / 鉴权 | `tmp/ray_log/<today>/api.log` |
| Ray init / actor 创建 / 巡检 / 备份 | `tmp/ray_log/<today>/deploy.log` |
| CLIP actor 出错 | `tmp/ray_log/<today>/clip.log` |
| YOLO / OneFormer / Qwen actor 出错 | 同上,改 actor 名 |
| Milvus 客户端 | `tmp/ray_log/<today>/milvus.log` |
| Ray 系统(raylet/gcs/dashboard) | `/tmp/ray/session_latest/logs/` |

### 1.4 全链路追踪 rid

```bash
# 1) 响应 header 拿到 X-Request-ID,例如 "7a3b9c..."
# 2) API 进程入口
grep "rid=7a3b9c" tmp/ray_log/20260609/api.log
# 3) 出错的具体 actor
grep "rid=7a3b9c" tmp/ray_log/20260609/oneformer.log
```

---

## 2. 排错 runbook

按"症状 → 怎么定位 → 怎么修"组织,凌晨被叫起来照着做。

### 2.1 启动失败

#### 症状:`python main.py bootstrap` 报 `OneFormer weights not found at .../weights/oneformer_ade20k_swin_large`

**根因**:首次没跑 prepare 或 weights/ 目录被误删。
**修复**:`python main.py prepare`,再 bootstrap。

#### 症状:Qwen actor 启动报 `trust_remote_code=True` 相关错误

**根因**:HF transformers 版本太老不识别 Qwen3-VL 接口。
**修复**:`pip install -U transformers`(>= 4.40);若 `Qwen3-VL-Embedding-2B` 真实 repo ID 不对,改 `config.QWEN_EMBED_REPO` 为实际名称重新 prepare。

#### 症状:OneFormer 启动耗时 30s+,日志显示 `UNEXPECTED: relative_position_index` / `MISSING: swin.layernorm.*`

**根因 1**(可忽略):shi-labs 老 ckpt 与新版 transformers 字段差异,**不影响精度**。
**根因 2**(真慢):`torch.compile(dynamic=True)` 首次 trace 慢,已在当前版本注释掉 compile;若仍想用,加 fp16 + 异步 warm_up。

#### 症状:`python main.py serve` 报 `Address already in use: 6379` 或 `8265`

**根因**:上次 `ray stop` 没清干净 / 端口被其他进程占。
**修复**:
```bash
ray stop --force
lsof -i :6379 -i :8265 -i :10001       # 看占用进程
kill -9 <pid>                           # 必要时手杀
```

### 2.2 运行时错误

#### 症状:某 actor 接 `503 "actor 'xxx' died"`

**根因**:actor 进程崩了。
**定位**:`tail -50 tmp/ray_log/<today>/xxx.log` 看异常栈。
**修复**:
- 第一次出现,dispatch 已自动 refresh + 重试,通常用户无感
- 反复出现 → `python main.py teardown && python main.py bootstrap`
- 若是 OOM(显存)→ 调小 `GPU_FRACTION_*` 或 `ACTOR_MAX_CONCURRENCY`

#### 症状:接 `503 "circuit open for 'xxx', retry later"`

**根因**:某 actor 10s 内连续 5 次失败,熔断保护打开 30s。
**定位**:`curl /version | jq .runtime.circuit` 看熔断状态;翻 actor 日志找根因。
**修复**:解决 actor 根因后等 30s 自动关闭;紧急可重启服务清空 `_circuit`。

#### 症状:接 `504 "inference timeout > 30s"`

**根因**:单请求超过 `INFER_TIMEOUT_SEC`,Ray 任务已被 cancel。
**定位**:看哪个端点 / 多大的图;OneFormer 大图 + 高分辨率确实可能超 30s。
**修复**:调大 `INFER_TIMEOUT_SEC`,或客户端先 resize。

#### 症状:接 `429 "too many requests"`

**根因**:并发超过 `MAX_INFLIGHT_REQUESTS=32`。
**修复**:客户端加 Semaphore 限并发,或调大常量(注意会增加 GPU 排队)。

#### 症状:接 `400 "image too large"` / `"invalid image"`

**根因**:输入图 > 20MB 或解码失败(可能是 PIL 拒绝的恶意图)。
**修复**:客户端检查源;真合法图 > 20MB 才需要调 `MAX_IMAGE_BYTES`。

### 2.3 Milvus 相关

#### 症状:`/readyz` 返回 503 `"milvus unavailable: ..."`

**根因**:
- Lite db 文件被删 / 权限不对 → `ls -la data/milvus_lite.db`
- 切了远程 milvus 但 server 没起 → `curl $MILVUS_URI`

**修复**:
- Lite 模式:`cp data/backup/milvus_<latest>.db data/milvus_lite.db` 恢复最近备份
- 远程模式:确认 server 正常,`unset MILVUS_URI` 退回 Lite

#### 症状:`/v1/search` 返回空 results

**根因**:
- 对应 collection 没数据 → `python main.py milvus stats` 看行数
- model 字段对不上 → search 用的 model 必须和入库时一致
- filter 表达式语法错误

**修复**:`/v1/embed` 入几张图先,再 search。

#### 症状:Milvus Lite db 损坏 / 误删

**修复**:
```bash
python main.py milvus list-backups      # 看可用备份
cp data/backup/milvus_<timestamp>.db data/milvus_lite.db
python main.py teardown && python main.py bootstrap
```

### 2.4 GPU / 显存

#### 症状:OOM(`CUDA out of memory`)

**根因**:四模型同卡,显存紧张。
**修复**:
- 优先:减小 `ACTOR_MAX_CONCURRENCY`(actor 内并发,默认 4,OneFormer 2)
- 次选:CLIP/OneFormer 加载改 fp16(`from_pretrained(..., torch_dtype=torch.float16)`)
- 最后:暂时禁用某个 actor(注释 `ray_deploy.ACTOR_SPECS` 里的项,bootstrap 时就不创建)

#### 症状:`nvidia-smi` 看显存被占但服务停了

**根因**:`ray stop` 没杀干净 actor 进程。
**修复**:
```bash
ray stop --force
ps aux | grep ray::                     # 看僵尸 actor 进程
kill -9 <pid>
```

### 2.5 Docker 模式相关

#### 症状:`docker compose up` 卡在 `prepare ~6GB` 下载好几分钟

**根因**:首次启动 `entrypoint.sh` 检测 `weights/` 三个子目录,任一缺失就跑 `python main.py prepare`。这是预期行为。
**修复**:
- 等就行,只发生一次
- 想避开容器内下载:宿主机先 `python main.py prepare` 把权重放到 `./weights/`,compose 把目录挂进去就跳过下载

#### 症状:`docker run` 报 `could not select device driver "" with capabilities: [[gpu]]`

**根因**:宿主机没装 `nvidia-container-toolkit`,或装了但没重启 docker daemon。
**修复**:
```bash
# Ubuntu 安装
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi   # 验证
```

#### 症状:容器内偶发 `RuntimeError: shared memory ...` / Ray plasma 报错

**根因**:`/dev/shm` 太小(docker 默认 64MB),Ray object store 装不下。
**修复**:确认 compose 里有 `shm_size: "4gb"`,或 `docker run --shm-size=4g`。负载大可调更大。

#### 症状:容器 HEALTHCHECK 一直 `unhealthy`

**根因**:首次启动权重下载 + actor 加载需要数分钟,而 healthcheck 的 `start_period: 120s` 不够长。
**定位**:`docker compose logs ray-inference | tail` 看是不是仍在 prepare 或 actor loading。
**修复**:
- 一次性问题:权重下完后下次启动就 < 60s 健康
- 永久:把 `docker-compose.yml` 的 `start_period` 调到 300s 或更长

#### 症状:`docker compose down` 后日志被截断 / 业务报 `connection reset`

**根因**:`stop_grace_period` 太短,SIGKILL 在 inflight drain 完之前就发了。
**修复**:确认 compose 里 `stop_grace_period: 45s`(代码侧 `SHUTDOWN_GRACE_SEC=30`,容器侧再给 15s buffer)。

#### 症状:`/version` 返回 `git_commit: "unknown"`

**根因**:`.dockerignore` 排除了 `.git/`,镜像内 `git rev-parse` 失败。
**修复**(非必需):
- 接受不显示 commit(默认),或
- 构建时把 commit 注入 env:`docker build --build-arg GIT_COMMIT=$(git rev-parse --short HEAD)`,Dockerfile 加 `ARG GIT_COMMIT` + `ENV GIT_COMMIT=$GIT_COMMIT`,改 `online_api._git_commit` 优先读 env

#### 症状:dashboard `http://localhost:8265` 访问不到

**根因**:`docker-compose.yml` 默认没暴露 8265 端口。
**修复**:取消注释 `- "127.0.0.1:8265:8265"` 后 `docker compose up -d`,并确认 `.env` 里 `RAY_DASHBOARD_HOST=0.0.0.0`(否则容器内 ray 还是绑回环)。

---

## 3. 升级 / 扩展指南

### 3.1 升级模型 revision(锁定可复现性)

1. 上 HF 模型页 Files and versions 拷新的 commit hash
2. `export CLIP_REVISION=<new_sha>`(或改 `config.py` 默认值)
3. `python main.py prepare --force`(下载锁定版本到 `weights/`)
4. `python main.py teardown && python main.py bootstrap`
5. `curl /version` 确认 revision 字段更新

### 3.2 切换 Qwen embedder 真实型号

如果 `Qwen/Qwen3-VL-Embedding-2B` 这个 repo 在 HF 上找不到,改 `config.py`:

```python
QWEN_EMBED_REPO = "<实际可用的 repo id>"
QWEN_EMBED_LOCAL_DIR = WEIGHTS_DIR / "<对应本地目录名>"
```

`models/qwen_embed_actor.py` 的 `_encode` 有三档回退:
1. `model.get_image_features(**inputs)`
2. `model(**inputs).pooler_output`
3. `model(**inputs).last_hidden_state.mean(dim=1)`

如果三档都不对,直接在 `_encode` 里改成模型的真实接口。

### 3.3 新增一个模型 actor

1. **新建** `models/<name>_actor.py`,继承 `BaseModelActor`,实现 `_load_model / _warm_up / infer`(签名带 `_rid: str = ""`)
2. **config.py** 加 `GPU_FRACTION_<NAME>`、`<NAME>_REPO / LOCAL_DIR / MODEL / REVISION`
3. **utils/prepare_models.py** 在 `download_all` 加一行 `_snapshot(...)`
4. **ray_deploy.py** 在 `ACTOR_SPECS` 加一项 `(key, ActorClass, gpu_frac, ray_name)`
5. **services/dispatch.py** 在 `_RAY_NAMES` 加一项
6. **services/online_api.py**(可选)加新 router 或新端点
7. `prepare && teardown && bootstrap` 让新 actor 就位

### 3.4 切换 Milvus 到生产 standalone

```bash
# 1. 起 standalone(docker compose)
wget https://github.com/milvus-io/milvus/releases/download/v2.4.10/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker compose up -d

# 2. 切环境变量
export MILVUS_URI=http://127.0.0.1:19530
export MILVUS_TOKEN=                    # 开了 auth 才填

# 3. 重启 API,代码零改
python main.py teardown && python main.py bootstrap
```

`/readyz` 探针自动覆盖新 milvus。

### 3.5 备份 / 恢复 Milvus Lite

**手动备份**:`python main.py milvus backup`,产物在 `data/backup/milvus_<时间戳>.db`。
**恢复**:
```bash
python main.py milvus list-backups
cp data/backup/milvus_<想要的>.db data/milvus_lite.db
# 重启服务让客户端拿到新单例
```

### 3.6 给 actor 加新方法(比如批量入库 task)

actor 加方法后**必须** `teardown && bootstrap`,uvicorn `--reload` **不会**重启 actor 进程,旧 actor 仍然没有新方法。

```python
# 例:CLIPActor 加文本批量 embed
def embed_text_batch(self, texts: list, _rid: str = "") -> dict:
    ...
```

router 里通过 `actor_call(key, method_name, ...)` 调:

```python
result = await actor_call("clip", "embed_text_batch", texts, {}, rid)
```

### 3.7 配置环境变量一览

| 变量 | 用途 | 默认 |
|---|---|---|
| `ONLINE_API_HOST` | FastAPI 绑定地址 | `0.0.0.0` |
| `ONLINE_API_PORT` | FastAPI 端口 | `8000`(docker 模式建议 7890) |
| `RAY_DASHBOARD_HOST` | Ray dashboard 绑定地址 | `127.0.0.1`(docker 内设 `0.0.0.0`) |
| `RAY_ADDRESS` | 连接远程 Ray 集群(`ray://host:10001`) | 空 = 本地起头节点 |
| `MILVUS_URI` | Milvus 连接串(`.db` 文件 = Lite) | `data/milvus_lite.db` |
| `MILVUS_TOKEN` | Milvus 鉴权 token | 空 |
| `CLIP_REVISION` | CLIP 权重 commit sha | 空 = HEAD |
| `ONEFORMER_REVISION` | OneFormer 权重 sha | 空 = HEAD |
| `QWEN_EMBED_REVISION` | Qwen 权重 sha | 空 = HEAD |
| `LOG_FORMAT` | `text`(默认,人读)或 `json`(结构化,送 Loki/ES) | `text` |
| `HF_HUB_OFFLINE` | 跳过 hub HEAD 验证 | 未设 |
| `HF_ENDPOINT` | HF 镜像 | 未设 |
| `CUDA_VISIBLE_DEVICES` | 选 GPU(docker 模式由 compose `device_ids` 控制) | 未设 |

### 3.8 切换 JSON 日志(送 Loki / ELK)

```bash
# 裸跑
LOG_FORMAT=json python main.py serve --port 7890

# Docker
# .env 加一行:
LOG_FORMAT=json
docker compose restart ray-inference
```

JSON 模式下每行示例:
```json
{"ts":"2026-06-09T12:34:56","level":"INFO","logger":"api","msg":"http_request rid=abc path=/classify status=200 ms=42.3","event":"http_request","rid":"abc","path":"/classify","status":200,"ms":42.3}
```

Loki / Promtail 配置示例(只贴关键过滤):
```yaml
- json:
    expressions:
      rid: rid
      path: path
      status: status
      ms: ms
      event: event
- labels:
    path:
    status:
    event:
```

回切 text:删除 `LOG_FORMAT` 或设 `LOG_FORMAT=text`。

### 3.9 fp16 精度回退(怀疑模型输出异常时)

v1.2 起 CLIP / OneFormer / Qwen 在 CUDA 上默认 fp16。出现"分类概率全 NaN / 分割空结果"等极端情况时,排查顺序:

1. 看日志 `actor ready device=cuda:0 dtype=torch.float16 ...` 是否真的 fp16
2. 临时回 fp32 验证(确认是 fp16 引起的问题再针对性调):
   ```python
   # models/base.py 改一行
   self._dtype = torch.float32   # 强制 fp32
   ```
3. 再 `teardown && bootstrap` 重建 actor

实测 CLIP-base-32 / OneFormer-Swin-L / Qwen3-VL-2B 三个模型 fp16 推理与 fp32 的结果差异在第 4 位小数以内,业务可忽略。

### 3.10 Docker 镜像构建 / 升级 / 切环境

#### 首次构建

```bash
cp .env.example .env                    # 必做,否则 compose 用 shell 当前环境
docker compose build                    # 构建,~10 分钟(主要是 pip install torch + transformers)
docker compose up -d
```

#### 改了 `requirements.txt`

```bash
docker compose build --no-cache         # 强制重装包
docker compose up -d --force-recreate
```

#### 改了应用代码(没动 requirements)

```bash
docker compose build                    # 用 layer cache,只重建 COPY . . 那层(秒级)
docker compose up -d --force-recreate
```

#### 升模型 revision

```bash
vim .env                                # 改 CLIP_REVISION 等
docker compose exec ray-inference python main.py prepare --force
docker compose restart ray-inference
docker compose exec ray-inference curl -s http://localhost:7890/version
```

#### 切到 Milvus standalone

在 `docker-compose.yml` 同目录新建 `compose.milvus.yml`(用 milvus 官方 compose),用 `docker compose -f docker-compose.yml -f compose.milvus.yml up -d` 同启。`.env` 里把 `MILVUS_URI=http://milvus:19530`,本服务通过 docker 默认网络解析 `milvus` 主机名。详细步骤见 §3.4。

#### 多 GPU / 选指定卡

`docker-compose.yml` 的 `device_ids` 改成 `["1"]` 或 `["0","1"]`;**注意**:Ray actor 用 GPU fraction 调度,多卡需要相应改 `config.GPU_FRACTION_*` 才能利用上。

#### 多实例 / 横向扩展

单容器单 Ray head 设计,不能直接 `docker compose up --scale ray-inference=3`(每实例都会争 GPU 0、写同一份 weights/data)。需要扩展:
- 多卡:每实例独占一张 GPU,挂不同 `weights/`(或软链共享)、不同 `data/` 文件
- 多节点:前面架个 Nginx 做 round-robin,或上 K8s(见路线图)

---

## 4. 路线图

短期可做(按优先级,业务驱动决定取舍):

| 项 | 类别 | 价值 | 估时 |
|---|---|---|---|
| **OpenTelemetry 集成** | 可观测 | rid 升 trace_id,送 Jaeger/Tempo,看完整调用链 | 半天 |
| ~~结构化日志(JSON)~~ | ~~可观测~~ | ~~`LOG_FORMAT=json` 切换,Loki/ES 直接索引~~ | **v1.2 已做** |
| ~~CLIP/OneFormer fp16~~ | ~~性能~~ | ~~推理快 30–50%,显存减半~~ | **v1.2 已做** |
| **OneFormer 输入短边 resize** | 性能 | 大图自动缩到 800px,P99 降一半 | 1 小时 |
| **错误体规范化** | API | `{error:{code,message,rid}}`,客户端解析不靠猜 | 2 小时 |
| **批量入库 task**(@ray.remote 函数) | 性能 | 几万张图入库提速 N×,不污染在线 API | 2 小时 |
| **集成测试 / 故障注入** | 测试 | 跑测试中途 ray.kill(actor),确认自愈 | 半天 |
| **依赖 pin + lock 文件** | 工程 | `uv pip compile` 锁版本,装包可复现 | 半小时 |
| **pre-commit hooks(ruff)** | 工程 | 提交前自动 lint + format | 15 分钟 |
| **OCR / BLIP caption 端点** | 业务 | OCR 与 detect 串联做招牌识别 | 1 天 |
| **CHANGELOG.md** | 文档 | 跟 git tag 对齐的发布日志 | 持续 |

明确不做:

- ❌ batch 端点(actor `max_concurrency` + 客户端并发已够)
- ❌ API 鉴权(本地内网,无公网风险)
- ❌ K8s / CI(单机 docker 已覆盖,集群按需引入)
- ❌ DNS rebinding / SSRF 防护(`requests` 直拉,接受风险)

---

## 5. 设计决策回顾(为什么这么做)

便于后来者读懂"为什么不那样写"。

| 决策 | 替代方案 | 选当前方案的理由 |
|---|---|---|
| Milvus 客户端走 FastAPI 进程,不进 actor | 包成 actor | actor 单实例 = 连接池被串行化,延迟 +1 跳 Ray RPC;DB 客户端轻量,直连更高效 |
| 每模型独立 collection | 共用一个 collection 加 model 字段 | 不同模型维度不同,vector 字段必须固定 dim → 只能拆 collection |
| source md5 做主键 | 自增 ID | 同图重复调用幂等;切模型回填时按 source 对齐 |
| Actor 强制本地加载 | HF Hub 在线 | transformers 即使命中缓存也会发 HEAD,被匿名限速会卡每次启动几秒到几十秒 |
| OneFormer 关掉 `torch.compile(dynamic=True)` | 保留 compile | dynamic trace 首次 30s+,推理收益对 Swin-L 有限,得不偿失 |
| YOLO 不 compile | 加 compile | letterbox 是 numpy,dynamo trace 不兼容,首次推理直接抛 fake tensor 错 |
| 抽出 `services/dispatch.py` | 都堆在 online_api | routers/embed.py 需要 actor_call,反向 import online_api 会循环 |
| 熔断器在 dispatch 层 | 在 middleware 层 | dispatch 知道是哪个 actor 失败,middleware 只看 HTTP 状态;按 actor 维度熔断更准 |
| 请求级缓存在 router 层(LRU) | dispatch / actor 内 | router 知道业务语义("同 source+model = 同结果"),dispatch 不该懂业务 |
| Milvus Lite + 启动备份 | 起 standalone | 单机场景零部署最快;data/backup/ 单文件备份也够用 |
