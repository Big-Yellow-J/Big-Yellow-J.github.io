"""图像输入加载:支持 bytes / 本地路径 / HTTP(S) URL / base64 / data URI,带 SSRF 防护与解码校验。"""
import base64
import ipaddress
import re
import socket
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError

from config import (
    MAX_IMAGE_BYTES,
    URL_ALLOW_PRIVATE_NETWORK,
    URL_ALLOWED_SCHEMES,
    URL_FETCH_TIMEOUT_SEC,
)

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_DATA_URI_RE = re.compile(r"^data:image/[^;]+;base64,", re.IGNORECASE)

# SSRF 黑名单:显式 CIDR(不用 ipaddress.is_private,后者把 2001::/23 等公网过渡段也算私网)
_BLOCKED_V4 = [
    ipaddress.IPv4Network("0.0.0.0/8"),
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("100.64.0.0/10"),   # CGN
    ipaddress.IPv4Network("224.0.0.0/4"),     # multicast
]
_BLOCKED_V6 = [
    ipaddress.IPv6Network("::/128"),          # unspecified
    ipaddress.IPv6Network("::1/128"),         # loopback
    ipaddress.IPv6Network("fc00::/7"),        # ULA
    ipaddress.IPv6Network("fe80::/10"),       # link-local
    ipaddress.IPv6Network("ff00::/8"),        # multicast
]


def load_image(source) -> Image.Image:
    """把任意输入解析为 RGB 模式 PIL.Image。

    Args:
        source: bytes / 本地路径 / http(s) URL / base64 字符串 / data URI。
    Returns:
        PIL.Image.Image (mode=RGB)
    Raises:
        ValueError: 解析失败或非合法图像。
    """
    data = _to_bytes(source)
    try:
        Image.open(BytesIO(data)).verify()   # verify 防恶意构造图像;verify 后必须重新 open
    except (UnidentifiedImageError, Exception) as e:
        raise ValueError(f"invalid image: {e}") from e
    return Image.open(BytesIO(data)).convert("RGB")


def _to_bytes(source) -> bytes:
    """把输入归一化为 bytes,并做大小校验。"""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif isinstance(source, str):
        data = _str_to_bytes(source)
    else:
        raise ValueError(f"unsupported source type: {type(source).__name__}")
    if not data:
        raise ValueError("empty image data")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"image too large: {len(data)} > {MAX_IMAGE_BYTES}")
    return data


def _str_to_bytes(s: str) -> bytes:
    """字符串按 URL → 路径 → base64 顺序识别。"""
    if _URL_RE.match(s):
        return _fetch_url(s)
    if _DATA_URI_RE.match(s):
        s = s.split(",", 1)[1]
    if len(s) < 4096 and ("/" in s or "\\" in s):    # 启发式:含分隔符且不太长 → 当本地路径试
        p = Path(s)
        if p.is_file():
            return p.read_bytes()
    try:
        return base64.b64decode(s, validate=True)
    except Exception as e:
        raise ValueError(f"cannot parse string as URL/path/base64: {e}") from e


def _fetch_url(url: str, max_redirects: int = 5) -> bytes:
    """带 SSRF 防护的 HTTP 拉取:scheme 白名单 + 私网拒绝 + 边下边限大小。

    重定向手动跟随而非交给 requests,每跳一次重新跑 scheme/私网校验,
    防止公网 URL 通过 30x 重定向到内网。
    """
    for _ in range(max_redirects + 1):
        parsed = urlparse(url)
        if parsed.scheme not in URL_ALLOWED_SCHEMES:
            raise ValueError(f"disallowed scheme: {parsed.scheme}")
        if not URL_ALLOW_PRIVATE_NETWORK and parsed.hostname:
            _reject_private_host(parsed.hostname)

        resp = requests.get(
            url, timeout=URL_FETCH_TIMEOUT_SEC, stream=True, allow_redirects=False,
        )
        try:
            if resp.is_redirect or resp.is_permanent_redirect:
                loc = resp.headers.get("Location")
                if not loc:
                    resp.raise_for_status()
                url = requests.compat.urljoin(url, loc)
                continue
            resp.raise_for_status()
            buf = bytearray()
            for chunk in resp.iter_content(64 * 1024):
                buf.extend(chunk)
                if len(buf) > MAX_IMAGE_BYTES:
                    raise ValueError("remote image exceeds size limit")
            return bytes(buf)
        finally:
            resp.close()
    raise ValueError(f"too many redirects (> {max_redirects})")


def _reject_private_host(host: str):
    """DNS 解析后逐条 IP 判断,命中 _BLOCKED_V4/V6 即拒绝;公网地址(含 IPv6)放行。"""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"DNS resolve failed: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        nets = _BLOCKED_V4 if ip.version == 4 else _BLOCKED_V6
        if any(ip in n for n in nets):
            raise ValueError(f"blocked private host: {host} -> {ip}")
