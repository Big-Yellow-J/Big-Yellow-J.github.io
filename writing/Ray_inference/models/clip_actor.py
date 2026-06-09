"""CLIP zero-shot 图像分类 Actor。"""
import time
from pathlib import Path
from typing import List

import ray
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import (
    ACTOR_MAX_CONCURRENCY,
    ACTOR_MAX_RESTARTS,
    ACTOR_MAX_TASK_RETRIES,
    CLIP_MODEL,
    GPU_FRACTION_CLIP,
)
from models.base import BaseModelActor
from utils.image_loader import load_image


@ray.remote(
    max_restarts=ACTOR_MAX_RESTARTS,
    max_task_retries=ACTOR_MAX_TASK_RETRIES,
    max_concurrency=ACTOR_MAX_CONCURRENCY,
)
class CLIPActor(BaseModelActor):
    """CLIP zero-shot 分类:给定图像和候选标签,返回每个标签的概率。"""

    def __init__(self):
        super().__init__(model_name="CLIP", gpu_fraction=GPU_FRACTION_CLIP)

    def _load_model(self):
        # 强制本地加载:CLIP_MODEL 指向项目内 weights/ 子目录,跳过 hub HEAD 验证。
        if not Path(CLIP_MODEL).is_dir():
            raise RuntimeError(
                f"CLIP weights not found at {CLIP_MODEL}. "
                f"run `python main.py prepare` to snapshot weights to weights/."
            )
        self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL, local_files_only=True)
        self._model = (
            CLIPModel.from_pretrained(CLIP_MODEL, local_files_only=True)
            .to(self._device).eval()
        )
        # lazy compile:加载时不会真编译,首次推理才触发;静态形状下兼容良好
        try:
            self._model = torch.compile(self._model)
        except Exception:
            pass

    def _warm_up(self):
        try:
            dummy = Image.new("RGB", (224, 224))
            inputs = self._processor(
                text=["x"], images=dummy, return_tensors="pt", padding=True,
            ).to(self._device)
            with torch.inference_mode():
                self._model(**inputs)
        except Exception:
            pass

    def infer(
        self,
        source,
        labels: List[str],
        top_k: int = 5,
        prompt_template: str = None,
        temperature: float = 1.0,
        _rid: str = "",
    ) -> dict:
        """对图像与候选标签做 zero-shot 分类。

        Args:
            source: 任意 image_loader 支持的输入。
            labels: 候选标签列表(显示用的原始 label,即使用模板也以此返回)。
            top_k: 仅返回概率最高的 top_k 个。
            prompt_template: 含 {label} 的模板,会包装每个 label 后送入 text encoder;留空直接用 label。
            temperature: softmax 温度,logits 在 softmax 前除以它。
        Returns:
            {"success": True, "predictions": [{"label": str, "prob": float}, ...]}
        """
        t0 = time.time()
        try:
            if not labels:
                raise ValueError("labels must be non-empty")
            image = load_image(source)
            texts = (
                [prompt_template.format(label=l) for l in labels]
                if prompt_template else labels
            )
            inputs = self._processor(
                text=texts, images=image, return_tensors="pt", padding=True,
            ).to(self._device)
            with torch.inference_mode():
                logits = self._model(**inputs).logits_per_image[0]
                probs = (logits / temperature).softmax(dim=-1).cpu().tolist()
            ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)[:top_k]
            self._track(t0, ok=True)
            return {
                "success": True,
                "predictions": [
                    {"label": l, "prob": round(p, 4)} for l, p in ranked
                ],
            }
        except Exception as e:
            self._track(t0, ok=False)
            return self._error(e, "classify", rid=_rid)
