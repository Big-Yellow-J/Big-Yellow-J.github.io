"""utils.image_loader 单元测试:四种输入分支 + 大小校验 + 损坏图防御。"""
import base64

import pytest
from PIL import Image

from utils.image_loader import load_image


def test_bytes_path(small_png_bytes):
    img = load_image(small_png_bytes)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (8, 8)


def test_local_file_path(tmp_path, small_png_bytes):
    p = tmp_path / "t.png"
    p.write_bytes(small_png_bytes)
    img = load_image(str(p))
    assert img.size == (8, 8)


def test_base64_plain(small_png_bytes):
    b64 = base64.b64encode(small_png_bytes).decode()
    img = load_image(b64)
    assert img.size == (8, 8)


def test_data_uri(small_png_bytes):
    b64 = base64.b64encode(small_png_bytes).decode()
    img = load_image(f"data:image/png;base64,{b64}")
    assert img.size == (8, 8)


def test_oversize_bytes_raises():
    big = b"\x00" * (40 * 1024 * 1024)        # 40 MB > MAX_IMAGE_BYTES(默认 20MB)
    with pytest.raises(ValueError, match="image too large"):
        load_image(big)


def test_corrupted_image_rejected():
    # 不是任何合法图像编码的字节流
    with pytest.raises(ValueError):
        load_image(b"this is not a valid image file at all")


def test_unsupported_type_rejected():
    with pytest.raises(ValueError, match="unsupported source"):
        load_image(123)
