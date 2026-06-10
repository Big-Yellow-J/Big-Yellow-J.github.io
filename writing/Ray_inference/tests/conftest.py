"""共享 pytest fixture:把项目根加进 sys.path、提供临时 milvus、CLIP 桩。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_milvus_uri(tmp_path, monkeypatch):
    """每个测试一个独立 Milvus Lite db,避免互污染。"""
    db = tmp_path / "test_milvus.db"
    monkeypatch.setattr("config.MILVUS_URI", str(db))
    monkeypatch.setattr("services.db.milvus.MILVUS_URI", str(db))
    # 重置单例
    import services.db.milvus as m
    m._client = None
    yield str(db)
    m.close_milvus()


@pytest.fixture
def small_png_bytes() -> bytes:
    """返回一张最小合法 PNG(8×8 红色)的字节,用于 image_loader 测试。"""
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()
