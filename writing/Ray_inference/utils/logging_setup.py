"""按日期组织的统一日志 setup:tmp/ray_log/<YYYYMMDD>/<name>.log + stderr。"""
import logging
import time
from pathlib import Path

from config import TMP_LOG_DIR


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """配置同名 logger:文件 + stderr 双输出,幂等(重复调用不重复加 handler)。

    Args:
        name: logger 名,同时作为日志文件名(`<name>.log`)。
        level: 日志级别。
    Returns:
        配置好的 logger。
    """
    today = time.strftime("%Y%m%d")
    log_dir = TMP_LOG_DIR / today
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False    # 避免根 logger 重复打印
    fmt = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger
