"""把 HF 类模型权重 snapshot 到项目内 weights/,启动时强制本地加载。

调用:`python main.py prepare [--force]` 或 `python -m utils.prepare_models`。
- 默认下载所有 HF 模型(CLIP + OneFormer);已存在的目录直接跳过。
- snapshot_download 内部按 etag 逐文件去重,不会真的重下已有文件;`--force` 才强制重拉。
- YOLO 由 ultralytics 在首次 actor 加载时自动从 GitHub 拉到当前目录,不在此脚本范围。
"""
from pathlib import Path

from huggingface_hub import snapshot_download

from config import (
    CLIP_LOCAL_DIR,
    CLIP_REPO,
    ONEFORMER_LOCAL_DIR,
    ONEFORMER_REPO,
    WEIGHTS_DIR,
)


def _snapshot(repo: str, local_dir: Path, force: bool) -> str:
    """单个 HF repo → 本地目录;非空且非 force 时早返回跳过。"""
    if local_dir.is_dir() and any(local_dir.iterdir()) and not force:
        print(f"[skip] {repo}: already exists at {local_dir}")
        return str(local_dir)
    print(f"[download] {repo} -> {local_dir}")
    snapshot_download(repo_id=repo, local_dir=str(local_dir), local_dir_use_symlinks=False)
    print(f"[done] {local_dir}")
    return str(local_dir)


def download_all(force: bool = False) -> dict:
    """下载所有 HF 模型(CLIP + OneFormer),返回 {repo: local_dir}。"""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return {
        CLIP_REPO: _snapshot(CLIP_REPO, CLIP_LOCAL_DIR, force),
        ONEFORMER_REPO: _snapshot(ONEFORMER_REPO, ONEFORMER_LOCAL_DIR, force),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="即使本地已存在也重新下载")
    download_all(force=p.parse_args().force)
