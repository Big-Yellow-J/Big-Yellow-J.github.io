"""统一日志 setup:按日期写 tmp/ray_log/<YYYYMMDD>/<name>.log + stderr。

格式:
    LOG_FORMAT=text(默认)  人读可读的单行
    LOG_FORMAT=json         结构化 JSON 单行,可直接喂 Loki/ES,extra={...} 字段会被字段化

用法:
    log = setup_logger("api")
    log.info("request done")                                # message-only
    log.info("request done", extra={"rid": rid, "ms": 12})  # JSON 模式下会附 rid/ms 字段
"""
import json
import logging
import os
import time
from pathlib import Path

from config import TMP_LOG_DIR

LOG_FORMAT = os.getenv("LOG_FORMAT", "text").lower()

# logging.LogRecord 默认字段,JsonFormatter 透传 extra 时需要排除这些 key
_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """把 LogRecord 渲染为单行 JSON。extra={...} 中的字段会作为顶级字段附上。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k in _RESERVED or k.startswith("_"):
                continue
            payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _build_formatter() -> logging.Formatter:
    """按 LOG_FORMAT 选 text 或 json formatter。"""
    if LOG_FORMAT == "json":
        return JsonFormatter()
    return logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
    logger.propagate = False
    fmt = _build_formatter()
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger
