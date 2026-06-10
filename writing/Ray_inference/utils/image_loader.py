"""图像加载:本地路径 / base64 / data URI / URL → PIL.Image。

- URL 走 requests 下载,边下边累计字节数,超过 MAX_IMAGE_BYTES 立即中断。
- 所有 URL 下载都落盘到 tmp/image/<YYYYMMDD>/<HHMMSS>_<url_md5_8>.<ext>。
"""
import base64
import hashlib
import re
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

from config import (
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    TMP_IMAGE_DIR,
    URL_FETCH_TIMEOUT_SEC,
)

# 全局解压炸弹防护:超过此像素数 PIL 抛 DecompressionBombError
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_DATA_URI_RE = re.compile(r"^data:image/[^;]+;base64,", re.IGNORECASE)
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def load_image(source) -> Image.Image:
    """统一图像加载入口。

    Args:
        source: bytes / 本地路径 / http(s) URL / base64 / data URI。
    Returns:
        PIL.Image.Image (mode=RGB)。
    Raises:
        ValueError: 超过 MAX_IMAGE_BYTES / 像素超限 / 文件不是合法图像 / base64 解码失败。
    """
    data = _to_bytes(source)
    # 二次校验:先 verify(不解码到内存)防御恶意构造,再重新 open 真正使用。
    try:
        Image.open(BytesIO(data)).verify()
    except Exception as e:
        raise ValueError(f"invalid image: {e}") from e
    return Image.open(BytesIO(data)).convert("RGB")


def _to_bytes(source) -> bytes:
    """归一化为 bytes,统一做大小校验。"""
    if isinstance(source, (bytes, bytearray)):
        _check_size(len(source))
        return bytes(source)
    if not isinstance(source, str):
        raise ValueError(f"unsupported source: {type(source).__name__}")

    if _URL_RE.match(source):
        return _download_to_tmp(source)            # 内部已校验
    if _DATA_URI_RE.match(source):
        source = source.split(",", 1)[1]

    p = Path(source)
    if p.is_file():
        _check_size(p.stat().st_size)
        return p.read_bytes()

    # 兜底当 base64
    _check_size(len(source) * 3 // 4)              # base64 → bytes 估算
    return base64.b64decode(source, validate=True)


def _check_size(size: int):
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"image too large: {size} > {MAX_IMAGE_BYTES}")


def _download_to_tmp(url: str) -> bytes:
    """边下边限大小,落盘到 tmp/image/<YYYYMMDD>/<HHMMSS>_<md5_8>.<ext>。"""
    today = time.strftime("%Y%m%d")
    save_dir = TMP_IMAGE_DIR / today
    save_dir.mkdir(parents=True, exist_ok=True)

    with requests.get(url, timeout=URL_FETCH_TIMEOUT_SEC, stream=True) as resp:
        resp.raise_for_status()
        buf = bytearray()
        for chunk in resp.iter_content(64 * 1024):
            buf.extend(chunk)
            if len(buf) > MAX_IMAGE_BYTES:
                raise ValueError(f"remote image too large: > {MAX_IMAGE_BYTES} bytes")
        data = bytes(buf)

    ts = time.strftime("%H%M%S")
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    ext = Path(urlparse(url).path).suffix.lower()
    if ext not in _IMG_EXTS:
        ext = ".jpg"
    (save_dir / f"{ts}_{h}{ext}").write_bytes(data)
    return data
