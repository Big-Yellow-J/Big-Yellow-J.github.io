"""Milvus Lite 备份:把 data/milvus_lite.db 复制到 data/backup/,保留最新 N 个。

老版 milvus-lite 把数据存为单文件(.db),新版改成目录(内含 collections/ + LOCK)。
本脚本同时支持两种布局:
    - 文件 → 复制为 milvus_<ts>.db
    - 目录 → 打包成 milvus_<ts>.tar.gz
非 Lite(远程 standalone)由 milvus 自身机制管理,跳过。
"""
import shutil
import tarfile
import time
from pathlib import Path

from config import DATA_DIR, MILVUS_BACKUP_KEEP, MILVUS_URI
from utils.logging_setup import setup_logger

log = setup_logger("milvus")
_BACKUP_DIR = DATA_DIR / "backup"
# 备份产物两种后缀(老/新版 lite 布局对应)
_BACKUP_GLOBS = ("milvus_*.db", "milvus_*.tar.gz")


def _is_lite() -> bool:
    """MILVUS_URI 以 .db 结尾视为 Lite 模式(无论实际是文件还是目录)。"""
    return MILVUS_URI.endswith(".db")


def _all_backups() -> list:
    """扫描 backup 目录,返回所有备份产物路径,按修改时间升序。"""
    if not _BACKUP_DIR.is_dir():
        return []
    out = []
    for pattern in _BACKUP_GLOBS:
        out.extend(_BACKUP_DIR.glob(pattern))
    return sorted(out, key=lambda p: p.stat().st_mtime)


def backup_lite() -> dict:
    """备份 Lite db;远程 milvus 直接返回 skipped。文件直接 copy,目录 tar.gz 打包。"""
    if not _is_lite():
        return {"skipped": True, "reason": "remote milvus, use milvus-backup tool"}
    src = Path(MILVUS_URI)
    if not src.exists():
        return {"skipped": True, "reason": f"source not found: {src}"}

    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    if src.is_file():
        dst = _BACKUP_DIR / f"milvus_{ts}.db"
        shutil.copy2(src, dst)
    elif src.is_dir():
        # 新版 lite:目录布局,用 tar.gz 打包保留结构与权限
        dst = _BACKUP_DIR / f"milvus_{ts}.tar.gz"
        with tarfile.open(dst, "w:gz") as tar:
            tar.add(src, arcname=src.name)
    else:
        return {"skipped": True, "reason": f"unsupported source type: {src}"}

    removed = _prune_old(MILVUS_BACKUP_KEEP)
    size_mb = dst.stat().st_size / 1024 / 1024
    log.info("milvus backup: %s size=%.1fMB pruned=%d", dst.name, size_mb, removed)
    return {"path": str(dst), "size_mb": round(size_mb, 2), "pruned": removed}


def _prune_old(keep: int) -> int:
    """按修改时间排序,只留最新 keep 个备份,返回删除数(.db 与 .tar.gz 混算)。"""
    if keep <= 0:
        return 0
    files = _all_backups()
    if len(files) <= keep:
        return 0
    to_remove = files[: len(files) - keep]
    for f in to_remove:
        try:
            f.unlink()
        except Exception:
            pass
    return len(to_remove)


def list_backups() -> list:
    """列出所有备份(路径 + 大小 + mtime),最新的在前。"""
    files = _all_backups()
    files.reverse()
    return [
        {
            "path": str(f),
            "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime)),
        }
        for f in files
    ]
