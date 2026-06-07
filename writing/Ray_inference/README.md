# Ray 多模型推理平台

单卡 40G GPU 上部署 4 个模型（ResNet50 / YOLO / SAM / CLIP），通过 Ray Actor 实现 **GPU 常驻**，避免重复加载。

## 架构

```
main.py                     # 一键启动入口
└── ray_deploy.py           # Ray 初始化、Actor 创建、健康监控
    ├── models/
    │   ├── base.py         # Actor 基类（GPU 常驻 + 健康上报）
    │   ├── resnet_actor.py # ResNet50 图像分类
    │   ├── yolo_actor.py   # YOLOv8 目标检测
    │   ├── sam_actor.py    # SAM 实体分割
    │   └── clip_actor.py   # CLIP 图文匹配/对话
    └── services/
        ├── online_api.py   # FastAPI 在线推理（单图 + 多图并发）
        ├── batch_api.py    # 离线批处理 API（提交/查询/取消）
        └── batch_worker.py # 批处理调度引擎
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务（在线 API + 离线批处理 API）
python main.py serve

# 3. 访问 API 文档
open http://localhost:8000/docs
```

## API 端点

### 在线推理（单图 / 多图并发）

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/health` | 全模型健康检查 |
| `POST` | `/classify` | ResNet50 单图分类 |
| `POST` | `/classify/batch` | ResNet50 **多图并发**分类 |
| `POST` | `/detect` | YOLO 单图目标检测 |
| `POST` | `/detect/batch` | YOLO **多图并发**检测 |
| `POST` | `/segment` | SAM 单图实体分割 |
| `POST` | `/segment/batch` | SAM **多图并发**分割 |
| `POST` | `/clip/similarity` | CLIP 单图图文相似度 |
| `POST` | `/clip/similarity/batch` | CLIP **多图并发**相似度 |
| `POST` | `/clip/chat` | CLIP 单图简短对话 |
| `POST` | `/clip/chat/batch` | CLIP **多图并发**对话 |

### 离线批处理 API

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/batch/jobs` | 提交 JSON 任务（多模型混合） |
| `POST` | `/batch/jobs/upload` | 上传多图提交统一任务 |
| `GET` | `/batch/jobs` | 列出所有批处理任务 |
| `GET` | `/batch/jobs/{job_id}` | 查询任务状态与结果 |
| `DELETE` | `/batch/jobs/{job_id}` | 取消/清理任务 |

## 使用示例

### 1. 在线多图并发推理

```bash
# 同时传 5 张图片做分类
curl -X POST http://localhost:8000/classify/batch \
  -F "files=@cat1.jpg" \
  -F "files=@cat2.jpg" \
  -F "files=@dog1.jpg" \
  -F "top_k=3"

# 响应：{"total": 3, "success_count": 3, "results": {"cat1.jpg": {...}, ...}}
```

### 2. 离线批处理 - JSON 提交

```bash
# 提交混合任务
curl -X POST http://localhost:8000/batch/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [
      {"model": "resnet", "image_path": "/data/cat.jpg", "top_k": 3},
      {"model": "yolo",   "image_path": "/data/dog.jpg", "conf": 0.5},
      {"model": "sam",    "image_path": "/data/bird.jpg"}
    ],
    "output_dir": "/tmp/batch_results"
  }'

# 查询任务状态
curl http://localhost:8000/batch/jobs/<job_id>
```

### 3. 离线批处理 - 上传多图

```bash
curl -X POST http://localhost:8000/batch/jobs/upload \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "model=resnet" \
  -F "top_k=5"
```

## GPU 显存分配（单卡 40G）

| 模型 | GPU 比例 | 预估显存 |
|------|----------|----------|
| ResNet50 | 15% | ~100MB |
| YOLOv8 | 30% | ~2GB |
| SAM | 40% | ~2.5GB |
| CLIP | 15% | ~1.5GB |

总计 < 7GB，40G 显存充裕。

## 健康保障

- **Ray Actor `max_restarts=3`**：模型崩溃自动重启
- **`lifetime="detached"`**：Actor 独立于主进程存活
- **后台健康监控**：每 30s 检查 Actor 状态并打印
- **任务超时保护**：`max_task_retries=2`
- **GPU 常驻**：Actor 初始化后模型始终加载在 GPU

## 配置

编辑 `config.py` 修改模型路径、GPU 分配比例、服务端口等。
