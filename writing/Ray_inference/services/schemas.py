"""HTTP 请求体 Pydantic 模型集中定义。所有非 source 字段都有默认值,保证客户端兼容。"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ClassifyBody(BaseModel):
    """CLIP zero-shot 分类请求。"""
    source: str = Field(..., description="path / http(s) URL / base64 / data URI")
    labels: List[str] = Field(
        ..., min_length=2, max_length=50,
        description="候选标签,至少 2 个;单标签 softmax 永远 1.0 无意义",
    )
    top_k: int = Field(5, ge=1, le=50)
    prompt_template: Optional[str] = Field(
        None,
        description="prompt 模板,需含 {label},如 'a photo of {label}';"
                    "留空则直接用 label。模板能显著提升 CLIP 准确率。",
    )
    temperature: float = Field(
        1.0, gt=0.0, le=100.0,
        description="softmax 温度。>1 概率分布更平,<1 更锐(适合接近 argmax)",
    )


class DetectBody(BaseModel):
    """YOLOv8 目标检测请求。"""
    source: str = Field(..., description="path / http(s) URL / base64 / data URI")
    conf: float = Field(0.25, ge=0.0, le=1.0, description="置信度阈值")
    iou: float = Field(0.45, ge=0.0, le=1.0, description="NMS IoU 阈值")
    max_det: int = Field(300, ge=1, le=1000, description="单图最大检测数")
    imgsz: int = Field(640, ge=64, le=2048, description="推理输入尺寸,32 的倍数最佳")
    classes: Optional[List[int]] = Field(
        None, description="只检测指定类索引(COCO 80 类),留空 = 全部。例:[0,2] 只要 person+car",
    )
    agnostic_nms: bool = Field(False, description="True = 类无关 NMS")


class SegmentBody(BaseModel):
    """OneFormer 通用分割请求。"""
    source: str = Field(..., description="path / http(s) URL / base64 / data URI")
    task: Literal["instance", "semantic", "panoptic"] = Field(
        "instance", description="分割任务类型",
    )
    return_mask: bool = Field(
        False,
        description="True 时每个实例返回 mask(响应体显著变大,慎用)",
    )
    mask_format: Literal["png_b64", "rle"] = Field(
        "png_b64",
        description="mask 返回格式。png_b64:PNG 二进制 base64;rle:COCO 列优先 RLE,体积更小",
    )
    score_threshold: float = Field(
        0.5, ge=0.0, le=1.0,
        description="instance/panoptic 实例置信度阈值;semantic 任务下忽略",
    )
    mask_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="mask 二值化阈值",
    )
