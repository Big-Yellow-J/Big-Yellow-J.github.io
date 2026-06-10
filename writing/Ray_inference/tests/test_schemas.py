"""Pydantic schema 边界测试。"""
import pytest
from pydantic import ValidationError

from services.schemas import (
    ClassifyBody,
    DetectBody,
    EmbedBody,
    EmbedTextBody,
    SearchBody,
    SearchTextBody,
    SegmentBody,
)


def test_classify_requires_at_least_2_labels():
    with pytest.raises(ValidationError):
        ClassifyBody(source="x.jpg", labels=["cat"])
    ClassifyBody(source="x.jpg", labels=["cat", "dog"])    # 2 个 OK


def test_classify_top_k_bounds():
    with pytest.raises(ValidationError):
        ClassifyBody(source="x.jpg", labels=["a", "b"], top_k=0)
    with pytest.raises(ValidationError):
        ClassifyBody(source="x.jpg", labels=["a", "b"], top_k=100)


def test_detect_iou_bounds():
    with pytest.raises(ValidationError):
        DetectBody(source="x.jpg", iou=1.5)
    DetectBody(source="x.jpg", iou=0.5)


def test_segment_task_enum():
    with pytest.raises(ValidationError):
        SegmentBody(source="x.jpg", task="unknown")
    for t in ("instance", "semantic", "panoptic"):
        SegmentBody(source="x.jpg", task=t)


def test_segment_mask_format_enum():
    with pytest.raises(ValidationError):
        SegmentBody(source="x.jpg", mask_format="jpg")
    SegmentBody(source="x.jpg", mask_format="rle")


def test_embed_default_model_is_clip():
    b = EmbedBody(source="x.jpg")
    assert b.model == "clip"
    assert b.write_db is True


def test_embed_unknown_model_rejected():
    with pytest.raises(ValidationError):
        EmbedBody(source="x.jpg", model="dinov2")


def test_search_top_k_bounds():
    with pytest.raises(ValidationError):
        SearchBody(source="x.jpg", top_k=0)
    with pytest.raises(ValidationError):
        SearchBody(source="x.jpg", top_k=200)


def test_embed_text_empty_rejected():
    with pytest.raises(ValidationError):
        EmbedTextBody(text="")


def test_embed_text_too_long_rejected():
    with pytest.raises(ValidationError):
        EmbedTextBody(text="x" * 3000)


def test_search_text_default_clip():
    b = SearchTextBody(text="a red car")
    assert b.model == "clip"
