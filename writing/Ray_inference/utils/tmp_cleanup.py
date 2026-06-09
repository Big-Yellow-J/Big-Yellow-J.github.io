"""tmp 清理:删除 tmp/ray_log 与 tmp/image 下比 TMP_CLEANUP_DAYS 老的日期子目录。"""
import shutil
from datetime import datetime, timedelta

from config import TMP_CLEANUP_DAYS, TMP_IMAGE_DIR, TMP_LOG_DIR


def cleanup_tmp(days: int = TMP_CLEANUP_DAYS) -> dict:
    """删除两个 tmp 根下 N 天前的日期目录,返回 {dir: 删除数}。"""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    summary = {}
    for root in (TMP_LOG_DIR, TMP_IMAGE_DIR):
        removed = 0
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if d.is_dir() and d.name.isdigit() and len(d.name) == 8 and d.name < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        summary[root.name] = removed
    return summary
