"""统一图像输入加载:支持本地路径 / HTTP(S) URL / base64 / 原始 bytes。"""
import base64
import re
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from config import MAX_IMAGE_BYTES, URL_FETCH_TIMEOUT_SEC

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_DATA_URI_RE = re.compile(r"^data:image/[^;]+;base64,", re.IGNORECASE)


def load_image(source) -> Image.Image:
    """把任意输入转成 PIL.Image(RGB)。

    Accepts:
        - bytes:        raw image bytes
        - str (path):   本地文件路径
        - str (URL):    http:// 或 https:// 开头
        - str (base64): 纯 base64 字符串,或 data: URI
    """
    return Image.open(BytesIO(_to_bytes(source))).convert("RGB")


def _to_bytes(source) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif isinstance(source, str):
        data = _str_to_bytes(source)
    else:
        raise ValueError(f"Unsupported image source type: {type(source).__name__}")

    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large: {len(data)} bytes > {MAX_IMAGE_BYTES}")
    if not data:
        raise ValueError("Empty image data")
    return data


def _str_to_bytes(s: str) -> bytes:
    if _URL_RE.match(s):
        resp = requests.get(s, timeout=URL_FETCH_TIMEOUT_SEC, stream=True)
        resp.raise_for_status()
        return resp.content

    if _DATA_URI_RE.match(s):
        s = s.split(",", 1)[1]

    # 启发式:看着像本地路径就先按路径处理
    if len(s) < 4096 and ("/" in s or "\\" in s or s.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))):
        p = Path(s)
        if p.is_file():
            return p.read_bytes()

    # 兜底按 base64 解码
    try:
        return base64.b64decode(s, validate=True)
    except Exception as e:
        raise ValueError(f"Cannot interpret string as URL/path/base64: {e}") from e
