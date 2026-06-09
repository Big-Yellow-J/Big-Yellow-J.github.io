# Ray 推理服务 · 待处理事项

记录当前服务还差什么、按优先级排序。已完成项见 README。

---

## ✅ 已经完成

**基线**
- 三模型 actor:CLIP / YOLOv8 / OneFormer-large
- FastAPI 端点 + Pydantic schemas 集中(`services/schemas.py`)
- 入口限流(`MAX_INFLIGHT_REQUESTS`)、单请求超时(`INFER_TIMEOUT_SEC` + `ray.cancel`)
- Actor 自愈(`max_restarts` + `lifetime="detached"`)
- Actor 句柄自愈(`RayActorError` 自动 `ray.get_actor` 重连 + 重试一次)
- 两种启动方式:`main.py serve` 一键、`ray start + bootstrap + uvicorn --reload` 热开发
- 压测脚本(`test/stress.py`),支持 path/URL/base64/mixed 输入与并发爬坡
- 各 actor 推理参数化(`prompt_template` / `temperature` / `iou` / `classes` / `score_threshold` 等)
- `torch.compile`:CLIP / OneFormer 开,YOLO 关(numpy preprocess 不兼容)
- 图像加载简化 + URL 自动落盘到 `tmp/image/`

**P0/P1 第二批(本轮全部落地)**
- ✅ #1 `request_id` 全链路:HTTP header → 中间件 → `_infer_call` → actor `_rid` → `_error` 写日志
- ✅ #2 输入大小硬限 `MAX_IMAGE_BYTES`(默认 20MB),URL 边下边累计超大立停
- ✅ #3 探针三层:`/healthz`(liveness)、`/readyz`(readiness 检查三 actor)、`/health`(详细诊断)
- ✅ #5 `tmp/.gitignore`(整目录屏蔽)+ 启动 `cleanup_tmp(days=TMP_CLEANUP_DAYS)`
- ✅ #6 慢请求自动 WARNING(`SLOW_REQUEST_MS=1000`)
- ✅ #7 `/metrics` 加 `http_requests_total{path,status}`
- ✅ #8 `/version` 端点(`APP_VERSION` + git short commit + 三模型 repo)
- ✅ #9 `main.py bootstrap` 失败 `sys.exit(1)`,便于 CI / 部署脚本检测
- ✅ #10 OneFormer `mask_format ∈ {png_b64, rle}`,RLE 体积比 PNG 小 5–10×

**模型本地化(新)**
- ✅ CLIP + OneFormer 强制本地加载(`local_files_only=True`),`weights/` 缺失抛 `RuntimeError` 引导先 prepare
- ✅ `python main.py prepare [--force]` 一次 snapshot 所有 HF 模型;目录非空早返回 `[skip]`,etag 校验逐文件去重
- ✅ `weights/.gitignore` 屏蔽提交

**日志结构化(新)**
- ✅ FastAPI / deploy / 三 actor 各自独立日志文件(`tmp/ray_log/<YYYYMMDD>/{api,deploy,clip,yolo,oneformer}.log`)
- ✅ 按日期分子目录,启动时清理 7 天前

---

## 🔴 P0 优先(未完成)

### #4. API 鉴权
**问题**:`/classify` 等端点裸奔,公网部署任何人都能调,可被刷流量 / 滥用 GPU。
**做法**:`X-API-Key` 头校验中间件(key 走环境变量),或前置 OAuth 网关。**公网上线前必做**。

---

## 🟡 P1 优先(未完成)

### #11. 单元测试
- `utils/image_loader`:path / URL / base64 / data URI 四分支 + `MAX_IMAGE_BYTES` 边界
- `services/online_api._infer_call`:`RayActorError` retry 逻辑(mock actor)
- `services/schemas`:Pydantic 校验边界(空 labels / 非法 task / 超长 source)
- `utils/tmp_cleanup`:过期日期目录识别

### #12. 集成测试 / 故障注入
- 端到端:启服务 → 调三端点 → 检查响应字段 + status code
- 故障注入:跑测试中途 `ray.kill(actor)`,确认下次调用自动重连
- prepare 缺失:断网下 actor 加载抛 `RuntimeError`(已实现,补一个测试覆盖)

---

## 🧠 架构演进 · 接入数据库 / 多服务 / 版本控制

回答"后续要扩成完整服务"时的设计决策。

### A. 数据库接入(Milvus / MySQL / Redis):**不要走 Ray actor**

| 维度 | actor 包一层 | FastAPI 进程内直连 |
|---|---|---|
| 调用链 | API → Ray IPC → actor → DB | API → DB |
| 延迟 | +1 跳 Ray RPC(几 ms) | 直接 |
| 连接池 | actor 单实例 ≈ 池被串行化 | 多 worker 共享真并发 |
| 资源调度 | 不消耗 GPU,actor 化无收益 | 同左 |

**推荐模式**:`services/db/<name>.py` 提供单例工厂,FastAPI `Depends` 注入。

```python
# services/db/milvus.py
from pymilvus import MilvusClient
_client: MilvusClient | None = None
def get_milvus() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=..., token=...)
    return _client

# services/routers/search.py
@router.post("/search")
async def search(body: SearchBody, mv: MilvusClient = Depends(get_milvus)):
    emb = await call_clip_embed(body.source)   # 走 Ray actor(CPU/GPU 重活)
    return mv.search("images", emb, ...)        # 直连 milvus(轻量 IO)
```

**例外**:
- **离线批量 ETL / 大批量 embed 入库** → 用 `@ray.remote` **函数(task)**,不是 actor,Ray 自动起多 worker 并行;
- **跨进程结果缓存** → 直接 Redis,不用 actor 自造缓存。

### B. 非 actor 服务在 `services/` 怎么扩

当前 services/ 文件平铺,适合现在的 4 个端点。**未来端点 > 5 或 `online_api.py` 超 300 行再拆**(不提前过度设计):

```
services/
├── online_api.py     # 仅装配 app + include_router(...)
├── routers/          # 按业务拆 router
│   ├── classify.py
│   ├── detect.py
│   ├── segment.py
│   └── search.py     # 例:向量检索新增
├── deps.py           # Depends 工厂(actor handle / DB / Redis 单例)
├── db/
│   ├── milvus.py
│   └── mysql.py
├── middleware.py
├── metrics.py
└── schemas.py
```

新增端点的判定:
- **纯计算(GPU)** → 走已有 actor 或新增 actor
- **IO(DB / HTTP / 缓存)** → FastAPI 进程内 `Depends`,**不要进 actor**
- **混合(embed + 入库)** → 计算部分进 actor,IO 部分留 router

### C. 版本控制(四层 + 一个补丁)

| 层 | 工具 | 落地点 | 当前 |
|---|---|---|---|
| 代码 | git tag `v3.2.0` + `config.APP_VERSION` 同步 | `/version` 端点 | ✅ |
| **API** | URL 前缀 `/v1/classify` | `APIRouter(prefix="/v1")` | ⚠️ 当前裸 `/classify`,先加 v1 前缀,日后破坏性变更不影响老客户端 |
| **模型** | HF repo + revision sha 锁定 | `snapshot_download(revision="<sha>")` | ⚠️ 当前不锁,上游悄悄更新会拉到新版,生产**必须 pin sha** |
| 依赖 | `requirements.txt` pinned + uv/poetry lock | `==X.Y.Z`,锁文件入库 | ⚠️ 当前未 pin |
| DB schema | mysql:alembic;milvus:collection 字段加 `schema_version` | 引入时一起做 | — 未接 |
| 发布日志 | `CHANGELOG.md` 跟 git tag 对齐 | 项目根 | ⚠️ 暂无 |

**最该立即做的**:`config.py` 给每个 HF repo 加 `*_REVISION`,prepare 用 `revision=` 参数 snapshot,`/version` 返回各模型 sha。否则不同时间跑 prepare 拉到的权重可能不一样,故障难复现。

### D. 进一步优化建议(按维度分类)

**性能**
- **请求级缓存**:`source` 哈希 + 参数为 key,LRU(进程内)或 Redis(跨进程)命中跳过整次推理。重复图查询场景收益巨大
- **微批合并**:用 Ray Serve 或自写 batching adapter,~10ms 窗口聚合请求送进 GPU,吞吐 ×N(同步延迟略增)
- **模型量化**:OneFormer fp16(`model.half()`)或 int8(`bitsandbytes`),显存减半 + 推理快 30–50%
- **预处理下推**:resize/letterbox 拆到 CPU actor(`num_cpus=2`),GPU actor 只跑 forward
- **YOLO TensorRT**:`yolo.export(format="engine")` → 加载 `.engine`,延迟降一半

**可靠性**
- **优雅退出**:`SIGTERM` 不应立即 `ray.kill`,先等 inflight 请求完成(给 30s grace)
- **熔断**:某 actor 连续 N 次 `RayActorError` → 进熔断态直接返回 503,避免反复 retry 拖垮入口
- **模型蓝绿切换**:新版本 actor 起 `name=clip_v2`,router 切流;旧的稳定运行 → 灰度发布零中断
- **依赖回退**:DB / Redis 不可用时降级返回缓存或 fail-open,而不是 5xx

**可观测**
- **OpenTelemetry**:把 `rid` 升格为 trace_id,送 Jaeger/Tempo,看完整调用链(API → Ray RPC → DB)
- **Sentry**:`_error` 上报异常,自动聚合 + 告警
- **结构化日志**(P2-#14):JSON line 直接进 Loki/ES,告别 grep

**安全**
- **TLS**:`uvicorn --ssl-keyfile/--ssl-certfile` 或反代(Nginx/Caddy)终结
- **限流升级**:从固定并发上限 → token bucket 按 IP / API key 计费
- **输入扫描**:大图、可疑 EXIF、PIL 解码异常都要拦,actor 不能被毒图打挂

**易用**
- **gRPC 端点**:重客户端 SDK 走 gRPC + protobuf,延迟和体积都比 JSON 优
- **CORS**:浏览器直调需 `CORSMiddleware`
- **OpenAPI examples**:为每个 schema 加 example,Swagger 体验更好
- **客户端 SDK**(P2-#15):封装 retry / timeout / 序列化

---

## 🟢 P2 优先(锦上添花)

### 13. 加视觉服务
- **OCR**(PaddleOCR / EasyOCR)→ 与 `/detect` 串联做招牌识别
- **Embedding**(复用 CLIP image encoder)→ 加 `/embed`,配合 milvus 做以图搜图
- **Caption / VLM**(BLIP / BLIP-2)→ 输出文本接 LLM 做下游

### 14. 结构化日志(JSON line)
字段:`ts / level / rid / endpoint / status / latency_ms / model`,直接喂 ELK / Loki。

### 15. 客户端 SDK
抽出 `RayInferClient`,封装重试 / 超时 / 序列化。

### 16. CORS
浏览器直调需要 `app.add_middleware(CORSMiddleware, ...)`。

---

## ⚪ 暂不做(明确放弃)

- ❌ batch 端点(actor `max_concurrency` + 客户端 `asyncio.gather` 已能拿到批吞吐)
- ❌ Docker / K8s / CI(本阶段不涉及部署)
- ❌ DNS rebinding 防护(`requests` 直拉,接受风险)

---

## 📋 推荐落地顺序(当前剩余)

1. **架构演进 C - 模型 revision 锁定** —— 5 分钟,可复现性立竿见影
2. **架构演进 C - API URL prefix `/v1`** —— 10 分钟,日后改协议不破坏老客户端
3. **#4 API 鉴权** —— 公网上线前必做,30 分钟
4. **架构演进 D - 性能 - 请求级缓存** —— 1 小时,重复调用大幅提速
5. **#11 image_loader 单测** —— 1 小时,后续重构有底气
6. **架构演进 A - DB 接入** —— 业务驱动,需要向量检索时再做
7. **#12 集成测试 / 故障注入**
8. **P2-#13 OCR / Embedding**(业务驱动)
