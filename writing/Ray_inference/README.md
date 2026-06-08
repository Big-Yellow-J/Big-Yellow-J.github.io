# Ray 推理服务(ResNet50 + YOLO)

单卡 GPU 上以 Ray Actor 形式常驻 ResNet50 与 YOLOv8 两个模型,FastAPI 暴露同步 HTTP 接口。

## 架构

```
main.py            一键入口(serve / status)
└── ray_deploy.py  Ray 初始化、Actor 创建、健康监控、FastAPI 装配
    ├── models/
    │   ├── base.py            Actor 基类
    │   ├── image_loader.py    统一图像加载(path / URL / base64 / bytes)
    │   ├── resnet_actor.py    ResNet50
    │   └── yolo_actor.py      YOLOv8
    └── services/
        └── online_api.py      FastAPI 端点
```

## 快速开始

```bash
pip install -r requirements.txt
python main.py serve                       # 默认 0.0.0.0:8000
python main.py serve --host 127.0.0.1 --port 9000
python main.py status                       # 查看已运行 actor
```

API 文档:`http://localhost:8000/docs`,Ray Dashboard:`http://localhost:8265`。

## 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET`  | `/health`   | 全模型健康状态 |
| `GET`  | `/actors`   | Actor 详情 |
| `POST` | `/classify` | ResNet50 分类(`top_k` 可选) |
| `POST` | `/detect`   | YOLOv8 目标检测(`conf` 可选) |

## 输入方式(三种,任选其一)

### 1. 上传文件

```bash
curl -X POST 'http://localhost:8000/classify?top_k=3' \
  -F 'file=@cat.jpg'
```

### 2. JSON + 本地路径 / URL

```bash
curl -X POST 'http://localhost:8000/detect?conf=0.3' \
  -H 'Content-Type: application/json' \
  -d '{"source": "/data/dog.jpg"}'

curl -X POST 'http://localhost:8000/classify' \
  -H 'Content-Type: application/json' \
  -d '{"source": "https://example.com/img.jpg"}'
```

### 3. JSON + base64 / data URI

```bash
curl -X POST 'http://localhost:8000/classify' \
  -H 'Content-Type: application/json' \
  -d '{"source": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB..."}'

curl -X POST 'http://localhost:8000/classify' \
  -H 'Content-Type: application/json' \
  -d '{"source": "data:image/png;base64,iVBORw0KGgo..."}'
```

服务端 `models/image_loader.py` 会按 `URL → 本地路径 → base64` 顺序自动判别。

## 高吞吐设计

- **GPU 常驻**:Actor 由 `lifetime="detached"` 保持,不随主进程退出。
- **Actor 内并发**:`max_concurrency=4`,同一 actor 可同时处理多个请求。
- **异步不阻塞**:FastAPI 端点用 `asyncio.wrap_future(ref.future())` 等待 Ray,事件循环不卡死;客户端用 `asyncio.gather` 并发即可拿到批量吞吐(因此**没有 `/batch` 端点**)。
- **半精度/cuDNN benchmark** 默认开启。

## 配置

`config.py` 集中管理 GPU 比例、端口、超时、最大图像字节数(默认 20 MB)等。

---

## 用 Ray vs 不用 Ray:差异分析

下面把同样的需求(单卡上常驻 ResNet50 + YOLO,通过 HTTP 暴露)用两套方案对比。

### 方案 A:不用 Ray,直接 FastAPI + torch

```python
# 启动时各加载一次,模型作为全局对象
model_resnet = resnet50(...).cuda().eval()
model_yolo   = YOLO(...)

@app.post("/classify")
def classify(file):  # 同步:GIL 串行
    ...
```

### 方案 B:Ray Actor + FastAPI(本仓库)

```python
ResNetActor.options(num_gpus=0.15, lifetime="detached", max_restarts=3).remote()
YOLOActor  .options(num_gpus=0.30, lifetime="detached", max_restarts=3).remote()
```