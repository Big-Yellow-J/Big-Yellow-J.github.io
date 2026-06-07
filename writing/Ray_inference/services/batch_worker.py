"""
离线批处理服务：使用 Ray 并行处理大批量数据。
支持：
- 图片文件夹批量分类
- 图片文件夹批量检测
- 图片文件夹批量分割
- JSON 批处理任务描述文件
"""
import json
import time
from pathlib import Path

import ray


class BatchProcessor:
    """离线批处理调度器。"""

    def __init__(self, actors: dict):
        """
        Args:
            actors: {"resnet": ActorHandle, "yolo": ActorHandle, ...}
        """
        self._actors = actors

    # ---------- 批量分类 ----------
    def batch_classify(self, image_dir: str, output_file: str = "",
                       top_k: int = 5, extensions: tuple = (".jpg", ".jpeg", ".png", ".webp")):
        """对文件夹内所有图片进行分类。"""
        images = sorted(Path(image_dir).glob("*"))
        images = [p for p in images if p.suffix.lower() in extensions]
        if not images:
            return {"success": False, "error": "No images found"}

        print(f"[Batch] 分类任务: {len(images)} 张图片")
        start = time.time()
        actor = self._actors["resnet"]
        futures = []
        for img in images:
            data = img.read_bytes()
            futures.append(actor.infer.remote(data, top_k))

        results = []
        for img, ref in zip(images, futures):
            r = ray.get(ref)
            r["file"] = img.name
            results.append(r)

        elapsed = time.time() - start
        print(f"[Batch] 分类完成: {len(results)} 张, 耗时 {elapsed:.1f}s")

        if output_file:
            Path(output_file).write_text(json.dumps(results, ensure_ascii=False, indent=2))

        return {"success": True, "total": len(results), "elapsed_sec": round(elapsed, 1), "results": results}

    # ---------- 批量检测 ----------
    def batch_detect(self, image_dir: str, output_file: str = "",
                     conf: float = 0.25, extensions: tuple = (".jpg", ".jpeg", ".png", ".webp")):
        """对文件夹内所有图片进行目标检测。"""
        images = sorted(Path(image_dir).glob("*"))
        images = [p for p in images if p.suffix.lower() in extensions]
        if not images:
            return {"success": False, "error": "No images found"}

        print(f"[Batch] 检测任务: {len(images)} 张图片")
        start = time.time()
        actor = self._actors["yolo"]
        futures = [actor.infer.remote(p.read_bytes(), conf) for p in images]

        results = []
        for img, ref in zip(images, futures):
            r = ray.get(ref)
            r["file"] = img.name
            results.append(r)

        elapsed = time.time() - start
        print(f"[Batch] 检测完成: {len(results)} 张, 耗时 {elapsed:.1f}s")

        if output_file:
            Path(output_file).write_text(json.dumps(results, ensure_ascii=False, indent=2))

        return {"success": True, "total": len(results), "elapsed_sec": round(elapsed, 1), "results": results}

    # ---------- 批量分割 ----------
    def batch_segment(self, image_dir: str, output_file: str = "",
                      extensions: tuple = (".jpg", ".jpeg", ".png", ".webp")):
        """对文件夹内所有图片进行 SAM 分割。"""
        images = sorted(Path(image_dir).glob("*"))
        images = [p for p in images if p.suffix.lower() in extensions]
        if not images:
            return {"success": False, "error": "No images found"}

        print(f"[Batch] 分割任务: {len(images)} 张图片")
        start = time.time()
        actor = self._actors["sam"]
        futures = [actor.infer.remote(p.read_bytes()) for p in images]

        results = []
        for img, ref in zip(images, futures):
            r = ray.get(ref)
            r["file"] = img.name
            results.append(r)

        elapsed = time.time() - start
        print(f"[Batch] 分割完成: {len(results)} 张, 耗时 {elapsed:.1f}s")

        if output_file:
            Path(output_file).write_text(json.dumps(results, ensure_ascii=False, indent=2))

        return {"success": True, "total": len(results), "elapsed_sec": round(elapsed, 1), "results": results}

    # ---------- 通用 JSON 任务文件 ----------
    def run_task_file(self, task_json_path: str) -> dict:
        """
        读 JSON 任务文件，批量执行多种推理。
        JSON 格式:
        {
            "tasks": [
                {"model": "resnet", "image": "/path/to/img.jpg", "top_k": 3},
                {"model": "yolo",   "image": "/path/to/img.jpg", "conf": 0.3},
                {"model": "sam",    "image": "/path/to/img.jpg"},
                {"model": "clip",   "image": "/path/to/img.jpg", "mode": "similarity", "texts": ["cat", "dog"]}
            ]
        }
        """
        task_data = json.loads(Path(task_json_path).read_text())
        tasks = task_data.get("tasks", [])
        if not tasks:
            return {"success": False, "error": "No tasks in file"}

        print(f"[Batch] 任务文件: {len(tasks)} 条")
        start = time.time()
        futures = []
        for t in tasks:
            img_bytes = Path(t["image"]).read_bytes()
            model_name = t["model"]
            actor = self._actors.get(model_name)
            if actor is None:
                futures.append(None)
                continue

            if model_name == "resnet":
                futures.append(actor.infer.remote(img_bytes, t.get("top_k", 5)))
            elif model_name == "yolo":
                futures.append(actor.infer.remote(img_bytes, t.get("conf", 0.25)))
            elif model_name == "sam":
                futures.append(actor.infer.remote(img_bytes))
            elif model_name == "clip":
                mode = t.get("mode", "similarity")
                texts = t.get("texts", [])
                if mode == "chat":
                    futures.append(actor.chat.remote(img_bytes, texts[0] if texts else ""))
                else:
                    futures.append(actor.image_text_similarity.remote(img_bytes, texts))
            else:
                futures.append(None)

        results = []
        for task, ref in zip(tasks, futures):
            r = ray.get(ref) if ref is not None else {"error": "unknown model"}
            r["task"] = task
            results.append(r)

        elapsed = time.time() - start
        print(f"[Batch] 任务文件完成: {len(results)} 条, 耗时 {elapsed:.1f}s")
        return {"success": True, "total": len(results), "elapsed_sec": round(elapsed, 1), "results": results}
