"""客户端示例:演示三种输入方式 + 客户端侧并发。"""
import asyncio
import base64
import json
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
TEST_IMAGE = "test.jpg"


def example_upload():
    print("=" * 50, "\n  1) multipart 上传文件\n", "=" * 50)
    if not Path(TEST_IMAGE).exists():
        print(f"skip: {TEST_IMAGE} not found")
        return
    with open(TEST_IMAGE, "rb") as f:
        r = httpx.post(f"{BASE_URL}/classify", files={"file": f}, params={"top_k": 3})
    print(json.dumps(r.json(), ensure_ascii=False, indent=2)[:400])


def example_local_path():
    print("=" * 50, "\n  2) JSON + 本地路径\n", "=" * 50)
    abs_path = str(Path(TEST_IMAGE).resolve())
    r = httpx.post(f"{BASE_URL}/detect", json={"source": abs_path}, params={"conf": 0.3})
    print(json.dumps(r.json(), ensure_ascii=False, indent=2)[:400])


def example_url():
    print("=" * 50, "\n  3) JSON + URL\n", "=" * 50)
    url = "https://ultralytics.com/images/zidane.jpg"
    r = httpx.post(f"{BASE_URL}/detect", json={"source": url}, params={"conf": 0.3})
    print(json.dumps(r.json(), ensure_ascii=False, indent=2)[:400])


def example_base64():
    print("=" * 50, "\n  4) JSON + base64\n", "=" * 50)
    if not Path(TEST_IMAGE).exists():
        print(f"skip: {TEST_IMAGE} not found")
        return
    b64 = base64.b64encode(Path(TEST_IMAGE).read_bytes()).decode()
    r = httpx.post(f"{BASE_URL}/classify", json={"source": b64}, params={"top_k": 3})
    print(json.dumps(r.json(), ensure_ascii=False, indent=2)[:400])


async def example_client_concurrency():
    """客户端并发取代服务端 batch 端点 —— actor 内部会自动并发处理。"""
    print("=" * 50, "\n  5) 客户端并发(替代批端点)\n", "=" * 50)
    if not Path(TEST_IMAGE).exists():
        print(f"skip: {TEST_IMAGE} not found")
        return
    abs_path = str(Path(TEST_IMAGE).resolve())
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        tasks = [
            client.post("/classify", json={"source": abs_path}, params={"top_k": 3})
            for _ in range(8)
        ]
        responses = await asyncio.gather(*tasks)
    print(f"sent {len(responses)} concurrent requests, "
          f"all success = {all(r.status_code == 200 for r in responses)}")


def example_health():
    print("=" * 50, "\n  health check\n", "=" * 50)
    print(httpx.get(f"{BASE_URL}/health").json())


if __name__ == "__main__":
    example_health()
    example_upload()
    example_local_path()
    example_url()
    example_base64()
    asyncio.run(example_client_concurrency())
