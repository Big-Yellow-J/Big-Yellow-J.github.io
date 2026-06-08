"""压力测试脚本。

覆盖维度:吞吐(QPS) / 延迟分布(P50-P99) / 错误率 / 限流(429) / 超时(504) /
端点间差异 / 输入方式开销 / 并发爬坡找拐点。

数据源(放在 test/ 下):
    image/             本地图像目录(随机采样,可复用)
    test_image.txt     每行一个 URL,# 开头视为注释

用法:
    python stress.py --endpoint classify --concurrency 8 --duration 30
    python stress.py --endpoint all --concurrency 8 --duration 30
    python stress.py --endpoint classify --ramp 1,2,4,8,16,32 --duration 10
    python stress.py --source url --endpoint detect --concurrency 16
    python stress.py --source base64 --endpoint segment --concurrency 4
"""
import argparse
import asyncio
import base64
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple

import httpx

HERE = Path(__file__).parent
IMAGE_DIR = HERE / "image"
URL_FILE = HERE / "test_image.txt"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CLIP_LABELS = [
    "a person", "a car", "a dog", "a cat", "a bicycle",
    "a tree", "a building", "indoor scene", "outdoor scene", "food",
]
ENDPOINTS = ("classify", "detect", "segment")
SOURCE_MODES = ("path", "url", "base64", "mixed")


# ---------- 数据源 ----------

class SourcePool:
    """从 image/ 与 test_image.txt 加载样本,按 mode 决定输入形态。"""

    def __init__(self, image_dir: Path, url_file: Path, mode: str):
        """构造采样池。

        Args:
            image_dir: 本地图像目录(允许不存在,但要保证 mode 对应的池非空)。
            url_file:  URL 列表文件(每行一个,# 注释)。
            mode:      "path" / "url" / "base64" / "mixed"。
        """
        self.mode = mode
        self.paths: List[str] = []
        self.urls: List[str] = []
        self.base64s: List[str] = []

        if image_dir.is_dir():
            self.paths = [
                str(p.resolve()) for p in sorted(image_dir.iterdir())
                if p.suffix.lower() in IMG_EXTS
            ]
            self.base64s = [
                base64.b64encode(Path(p).read_bytes()).decode() for p in self.paths
            ]
        if url_file.is_file():
            self.urls = [
                line.strip() for line in url_file.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
        self._validate()

    def _validate(self):
        ok = {
            "path": self.paths,
            "url": self.urls,
            "base64": self.base64s,
            "mixed": self.paths + self.urls + self.base64s,
        }[self.mode]
        if not ok:
            raise RuntimeError(
                f"empty pool for mode={self.mode} "
                f"(paths={len(self.paths)} urls={len(self.urls)} base64s={len(self.base64s)})"
            )

    def pick(self) -> Tuple[str, str]:
        """随机抽一个样本。

        Returns:
            (source_kind, source_string),kind ∈ {path, url, base64}。
        """
        if self.mode == "path":
            return "path", random.choice(self.paths)
        if self.mode == "url":
            return "url", random.choice(self.urls)
        if self.mode == "base64":
            return "base64", random.choice(self.base64s)
        # mixed:按非空池等概率分流
        bins = []
        if self.paths:   bins.append(("path", self.paths))
        if self.urls:    bins.append(("url", self.urls))
        if self.base64s: bins.append(("base64", self.base64s))
        kind, pool = random.choice(bins)
        return kind, random.choice(pool)


# ---------- 请求构造 ----------

def build_payload(endpoint: str, source: str) -> dict:
    """按端点名构造对应的 JSON body。"""
    if endpoint == "classify":
        return {"source": source, "labels": CLIP_LABELS, "top_k": 3}
    if endpoint == "detect":
        return {"source": source, "conf": 0.25}
    if endpoint == "segment":
        return {"source": source, "task": "instance", "return_mask": False}
    raise ValueError(f"unknown endpoint: {endpoint}")


# ---------- 采样记录 ----------

@dataclass
class Sample:
    endpoint: str
    source_kind: str
    status: int
    elapsed_ms: float
    error: str = ""


@dataclass
class Bucket:
    samples: List[Sample] = field(default_factory=list)


# ---------- 执行 ----------

async def _one_request(
    client: httpx.AsyncClient, endpoint: str, pool: SourcePool,
) -> Sample:
    """发一次请求,记录 status/耗时/错误摘要。"""
    kind, source = pool.pick()
    t0 = time.time()
    try:
        r = await client.post(f"/{endpoint}", json=build_payload(endpoint, source))
        elapsed = (time.time() - t0) * 1000.0
        err = "" if r.status_code == 200 else r.text[:120]
        return Sample(endpoint, kind, r.status_code, elapsed, err)
    except Exception as e:
        return Sample(endpoint, kind, -1, (time.time() - t0) * 1000.0, str(e)[:120])


async def _worker(
    client: httpx.AsyncClient, ep_picker: Callable[[], str],
    pool: SourcePool, deadline: float, bucket: Bucket,
):
    """worker 循环:到 deadline 为止持续发请求。"""
    while time.time() < deadline:
        bucket.samples.append(await _one_request(client, ep_picker(), pool))


async def run_phase(
    base_url: str, endpoints: List[str], pool: SourcePool,
    concurrency: int, duration: float, timeout: float,
) -> Bucket:
    """一个阶段:固定 concurrency 持续 duration 秒。

    Returns:
        本阶段所有请求的采样记录。
    """
    bucket = Bucket()
    deadline = time.time() + duration
    picker = (
        (lambda: endpoints[0]) if len(endpoints) == 1
        else (lambda: random.choice(endpoints))
    )
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        await asyncio.gather(*[
            _worker(client, picker, pool, deadline, bucket)
            for _ in range(concurrency)
        ])
    return bucket


# ---------- 统计 ----------

def _pct(values: List[float], p: int) -> float:
    """计算分位数(p ∈ [1,99])。"""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100)[p - 1]


def _row(label: str, n: int, errs: int, latencies: List[float]) -> str:
    """渲染一行 per-group 统计。"""
    avg = statistics.mean(latencies) if latencies else 0.0
    p99 = _pct(latencies, 99) if latencies else 0.0
    return f"    {label:10s} n={n:5d} err={errs:4d} avg={avg:7.1f}ms p99={p99:7.1f}ms"


def print_summary(label: str, bucket: Bucket, elapsed_sec: float):
    """打印一个阶段的总览 + 端点分组 + 输入类型分组。"""
    samples = bucket.samples
    total = len(samples)
    ok_latencies = [s.elapsed_ms for s in samples if s.status == 200]
    by_status: dict = {}
    for s in samples:
        by_status[s.status] = by_status.get(s.status, 0) + 1

    qps = total / elapsed_sec if elapsed_sec > 0 else 0.0

    print(f"\n── {label} " + "─" * max(0, 60 - len(label)))
    print(f"  total={total}  ok={by_status.get(200, 0)}  qps={qps:.2f}  duration={elapsed_sec:.1f}s")
    print(f"  status: " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))

    if ok_latencies:
        print(
            f"  latency ms (200):  "
            f"avg={statistics.mean(ok_latencies):.1f}  "
            f"p50={_pct(ok_latencies, 50):.1f}  "
            f"p90={_pct(ok_latencies, 90):.1f}  "
            f"p95={_pct(ok_latencies, 95):.1f}  "
            f"p99={_pct(ok_latencies, 99):.1f}  "
            f"max={max(ok_latencies):.1f}"
        )

    endpoints = sorted({s.endpoint for s in samples})
    if len(endpoints) > 1:
        print("  per endpoint:")
        for ep in endpoints:
            sub = [s for s in samples if s.endpoint == ep]
            ok = [s.elapsed_ms for s in sub if s.status == 200]
            print(_row(ep, len(sub), len(sub) - len(ok), ok))

    kinds = sorted({s.source_kind for s in samples})
    if len(kinds) > 1:
        print("  per source:")
        for k in kinds:
            sub = [s for s in samples if s.source_kind == k]
            ok = [s.elapsed_ms for s in sub if s.status == 200]
            print(_row(k, len(sub), len(sub) - len(ok), ok))


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(description="Ray inference stress test")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--endpoint", choices=list(ENDPOINTS) + ["all"], default="classify")
    p.add_argument("--source", choices=list(SOURCE_MODES), default="mixed")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--duration", type=float, default=30.0, help="单阶段秒数")
    p.add_argument("--ramp", default=None,
                   help="爬坡并发列表,逗号分隔,如 1,2,4,8,16,32(覆盖 --concurrency)")
    p.add_argument("--timeout", type=float, default=60.0, help="客户端 HTTP 超时秒")
    args = p.parse_args()

    pool = SourcePool(IMAGE_DIR, URL_FILE, args.source)
    endpoints = list(ENDPOINTS) if args.endpoint == "all" else [args.endpoint]
    concs = [int(x) for x in args.ramp.split(",")] if args.ramp else [args.concurrency]

    print("# Ray Inference Stress")
    print(f"# base={args.base_url}  endpoints={endpoints}  source={args.source}")
    print(f"# pool: paths={len(pool.paths)} urls={len(pool.urls)} base64s={len(pool.base64s)}")

    for c in concs:
        t0 = time.time()
        bucket = asyncio.run(run_phase(
            args.base_url, endpoints, pool, c, args.duration, args.timeout,
        ))
        print_summary(f"concurrency={c}", bucket, time.time() - t0)


if __name__ == "__main__":
    """
    # 1. 单端点稳态压测(测吞吐 + 延迟分布)
    python stress.py --endpoint classify --concurrency 8 --duration 30

    # 2. 混合三端点(测互相影响)
    python stress.py --endpoint all --concurrency 8 --duration 30

    # 3. 并发爬坡(找拐点)
    python stress.py --endpoint classify --ramp 1,2,4,8,16,32 --duration 10

    # 4. 触发限流(MAX_INFLIGHT_REQUESTS=32,拉到 64 应有大量 429)
    python stress.py --endpoint classify --concurrency 64 --duration 20

    # 5. 触发超时(INFER_TIMEOUT_SEC=30,压 OneFormer 大图)
    python stress.py --endpoint segment --concurrency 8 --duration 60

    # 6. 单独看某种输入方式开销
    python stress.py --source path     --endpoint classify --concurrency 8
    python stress.py --source url      --endpoint classify --concurrency 8
    python stress.py --source base64   --endpoint classify --concurrency 8
    """
    
    main()
