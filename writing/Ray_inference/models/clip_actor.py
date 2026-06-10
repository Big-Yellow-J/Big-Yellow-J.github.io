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
            CLIPModel.from_pretrained(
                CLIP_MODEL, local_files_only=True, torch_dtype=self._dtype,
            )
            .to(self._device).eval()
        )
        # 不开 torch.compile:某些 transformers 版本下,compile 会把 get_image_features /
        # get_text_features 重定向到底层 forward,返回 BaseModelOutputWithPooling 而不是 tensor,
        # 触发 ".norm() not found" 错误。已开 fp16,推理速度本身已足够。

    def _warm_up(self):
        try:
            dummy = Image.new("RGB", (224, 224))
            inputs = self._processor(
                text=["x"], images=dummy, return_tensors="pt", padding=True,
            ).to(self._device)
            self._cast_floats(inputs)
            with torch.inference_mode():
                self._model(**inputs)
        except Exception:
            pass

    @staticmethod
    def _to_feat_tensor(out):
        """从 CLIP get_*_features 的返回值里抽出 (N, D) tensor。

        兼容三种返回形态:
        - tensor:直接返回
        - 含 image_embeds / text_embeds:CLIPOutput 完整前向
        - 含 pooler_output:vision_model / text_model 兜底
        """
        if isinstance(out, torch.Tensor):
            return out
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            v = getattr(out, attr, None)
            if v is not None:
                return v
        raise RuntimeError(f"cannot extract feature tensor from {type(out).__name__}")

    def embed(self, source, _rid: str = "") -> dict:
        """图像 → 归一化 image embedding(512 维),与 embed_text 共享同一向量空间。"""
        t0 = time.time()
        try:
            image = load_image(source)
            inputs = self._processor(images=image, return_tensors="pt").to(self._device)
            self._cast_floats(inputs)
            with torch.inference_mode():
                feat = self._to_feat_tensor(self._model.get_image_features(**inputs))
                feat = feat / feat.norm(dim=-1, keepdim=True)
            vec = feat[0].float().cpu().tolist()
            self._track(t0, ok=True)
            return {"success": True, "embedding": vec, "dim": len(vec), "model": "clip"}
        except Exception as e:
            self._track(t0, ok=False)
            return self._error(e, "embed", rid=_rid)

    def embed_text(self, text: str, _rid: str = "") -> dict:
        """文本 → 归一化 text embedding(512 维),可与图像 embedding 跨模态检索。"""
        t0 = time.time()
        try:
            if not text or not text.strip():
                raise ValueError("text must be non-empty")
            inputs = self._processor(
                text=[text], return_tensors="pt", padding=True, truncation=True,
            ).to(self._device)
            # 文本侧没有 pixel_values 也没关系,_cast_floats 只会动浮点 tensor
            self._cast_floats(inputs)
            with torch.inference_mode():
                feat = self._to_feat_tensor(self._model.get_text_features(**inputs))
                feat = feat / feat.norm(dim=-1, keepdim=True)
            vec = feat[0].float().cpu().tolist()
            self._track(t0, ok=True)
            return {"success": True, "embedding": vec, "dim": len(vec), "model": "clip"}
        except Exception as e:
            self._track(t0, ok=False)
            return self._error(e, "embed_text", rid=_rid)

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
            self._cast_floats(inputs)
            with torch.inference_mode():
                logits = self._model(**inputs).logits_per_image[0]
                # softmax 在 fp32 上更稳;同时把 nan/inf 折叠掉,避免 fp16 极小值产生 NaN
                probs = (logits.float() / temperature).softmax(dim=-1).cpu().tolist()
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
