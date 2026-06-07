"""
客户端示例：演示如何调用在线 API（单图/多图）和离线批处理 API。
"""
import json
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
TEST_IMAGE = "test.jpg"  # 替换为实际图片路径


def example_online_single():
    """在线单图推理。"""
    if not Path(TEST_IMAGE).exists():
        print(f"⚠️  测试图片 {TEST_IMAGE} 不存在，跳过单图示例")
        return

    print("=" * 50)
    print("  单图在线推理示例")
    print("=" * 50)

    with open(TEST_IMAGE, "rb") as f:
        image_bytes = f.read()

    # 1. 健康检查
    r = httpx.get(f"{BASE_URL}/health")
    print(f"\n[健康检查] {r.json()}")

    # 2. 分类
    r = httpx.post(f"{BASE_URL}/classify", files={"file": image_bytes}, params={"top_k": 3})
    print(f"\n[分类] {json.dumps(r.json(), ensure_ascii=False, indent=2)[:400]}")

    # 3. 目标检测
    r = httpx.post(f"{BASE_URL}/detect", files={"file": image_bytes}, params={"conf": 0.3})
    print(f"\n[检测] {json.dumps(r.json(), ensure_ascii=False, indent=2)[:400]}")


def example_online_batch():
    """在线多图批量推理 —— 一次传入 N 张图，并发处理。"""
    if not Path(TEST_IMAGE).exists():
        print(f"⚠️  测试图片 {TEST_IMAGE} 不存在，跳过多图示例")
        return

    print("\n" + "=" * 50)
    print("  多图批量推理示例（在线，并发）")
    print("=" * 50)

    # 构造多文件上传：同一张图传 3 次模拟多图
    files = [("files", open(TEST_IMAGE, "rb")) for _ in range(3)]

    # 多图分类
    r = httpx.post(f"{BASE_URL}/classify/batch",
                   files=files, params={"top_k": 3})
    result = r.json()
    print(f"\n[多图分类] total={result['total']}, success={result['success_count']}")

    # 重置文件指针 & 多图检测
    files = [("files", open(TEST_IMAGE, "rb")) for _ in range(3)]
    r = httpx.post(f"{BASE_URL}/detect/batch",
                   files=files, params={"conf": 0.3})
    result = r.json()
    print(f"[多图检测] total={result['total']}, success={result['success_count']}")


def example_batch_api():
    """离线批处理 API —— 提交任务 / 查状态 / 取结果。"""
    print("\n" + "=" * 50)
    print("  离线批处理 API 示例")
    print("=" * 50)

    # ===== 方式 1：JSON 任务提交 =====
    task_payload = {
        "tasks": [
            {"model": "resnet", "image_path": "/path/to/cat.jpg", "top_k": 3},
            {"model": "yolo",   "image_path": "/path/to/dog.jpg", "conf": 0.5},
            {"model": "sam",    "image_path": "/path/to/bird.jpg"},
            {"model": "clip",   "image_path": "/path/to/scene.jpg",
             "mode": "similarity", "texts": ["室内", "室外", "城市"]},
        ],
        "output_dir": "/tmp/batch_results"
    }

    # 提交任务
    r = httpx.post(f"{BASE_URL}/batch/jobs", json=task_payload)
    job = r.json()
    job_id = job["job_id"]
    print(f"\n📤 已提交任务: {job_id}  (status={job['status']})")

    # 轮询查询（生产建议用 webhook）
    import time
    for _ in range(30):
        time.sleep(2)
        r = httpx.get(f"{BASE_URL}/batch/jobs/{job_id}")
        status = r.json()
        print(f"  ⏳ {status['completed']}/{status['total']} 完成")
        if status["status"] in ("completed", "failed"):
            print(f"  ✅ 任务完成: success={status['success_count']}, fail={status['fail_count']}, "
                  f"耗时 {status['elapsed_sec']}s")
            break

    # 列出所有任务
    r = httpx.get(f"{BASE_URL}/batch/jobs")
    jobs_list = r.json()
    print(f"\n📋 当前任务列表: {jobs_list['total_jobs']} 个")

    # ===== 方式 2：直接上传多图提交 =====
    if Path(TEST_IMAGE).exists():
        files = [("files", open(TEST_IMAGE, "rb")) for _ in range(2)]
        r = httpx.post(
            f"{BASE_URL}/batch/jobs/upload",
            files=files,
            params={"model": "resnet", "top_k": 3, "output_dir": "/tmp/batch_results"}
        )
        print(f"\n📤 上传多图任务: job_id={r.json()['job_id']}")

    # 清理任务（可选）
    # httpx.delete(f"{BASE_URL}/batch/jobs/{job_id}")


def example_task_file():
    """JSON 任务文件格式示例。"""
    task_example = {
        "tasks": [
            {"model": "resnet", "image_path": "/data/images/cat.jpg", "top_k": 3},
            {"model": "yolo",   "image_path": "/data/images/cat.jpg", "conf": 0.5},
            {"model": "sam",    "image_path": "/data/images/cat.jpg"},
            {"model": "clip",   "image_path": "/data/images/dog.jpg",
             "mode": "similarity", "texts": ["猫", "狗", "汽车"]},
        ]
    }
    example_path = Path(__file__).parent / "example_tasks.json"
    example_path.write_text(json.dumps(task_example, ensure_ascii=False, indent=2))
    print(f"\n📄 示例任务文件: {example_path}")
    print(f"   提交方式: curl -X POST {BASE_URL}/batch/jobs "
          f"-H 'Content-Type: application/json' -d @{example_path}")


if __name__ == "__main__":
    example_online_single()
    example_online_batch()
    example_batch_api()
    example_task_file()
