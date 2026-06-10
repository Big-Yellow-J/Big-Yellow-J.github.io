"""Qwen3-VL-Embedding 图像 Embedding Actor(与 CLIP 并列的可选 embedder)。

注意:
- Qwen 系列模型必须 `trust_remote_code=True`。
- 实际 pooling 接口因模型版本而异:优先 `model.get_image_features` 或 `pooler_output`,
  兜底取 last_hidden_state.mean(1)。若上游 API 名变了,只需调整 `_encode`。
- 维度由 model.config.hidden_size 动态确定,milvus collection 也按此维度建。
"""
import time
from pathlib import Path

import ray
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from config import (
    ACTOR_MAX_CONCURRENCY,
    ACTOR_MAX_RESTARTS,
    ACTOR_MAX_TASK_RETRIES,
    GPU_FRACTION_QWEN_EMBED,
    QWEN_EMBED_MODEL,
)
from models.base import BaseModelActor
from utils.image_loader import load_image


@ray.remote(
    max_restarts=ACTOR_MAX_RESTARTS,
    max_task_retries=ACTOR_MAX_TASK_RETRIES,
    max_concurrency=max(1, ACTOR_MAX_CONCURRENCY // 2),    # 2B 模型显存占用较大,降一档并发
)
class QwenEmbedActor(BaseModelActor):
    """Qwen3-VL-Embedding 图像编码器。"""

    def __init__(self):
        super().__init__(model_name="QwenEmbed", gpu_fraction=GPU_FRACTION_QWEN_EMBED)

    def _load_model(self):
        """从本地 weights/ 加载,fp16 推理减半显存。"""
        if not Path(QWEN_EMBED_MODEL).is_dir():
            raise RuntimeError(
                f"Qwen embedding weights not found at {QWEN_EMBED_MODEL}. "
                f"run `python main.py prepare` to snapshot weights to weights/."
            )
        self._processor = AutoProcessor.from_pretrained(
            QWEN_EMBED_MODEL, local_files_only=True, trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            QWEN_EMBED_MODEL,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=self._dtype,
        ).to(self._device).eval()

    def _warm_up(self):
        try:
            self._encode(Image.new("RGB", (224, 224)))
        except Exception:
            pass

    def _encode(self, image: Image.Image) -> list:
        """图像 → 归一化 embedding(float list)。三档回退覆盖不同 Qwen 接口。"""
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        self._cast_floats(inputs)
        with torch.inference_mode():
            if hasattr(self._model, "get_image_features"):
                feat = self._model.get_image_features(**inputs)
            else:
                out = self._model(**inputs)
                feat = (
                    getattr(out, "pooler_output", None)
                    if getattr(out, "pooler_output", None) is not None
                    else out.last_hidden_state.mean(dim=1)
                )
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].float().cpu().tolist()

    def infer(self, source, _rid: str = "") -> dict:
        """与 CLIP/YOLO/OneFormer 一致的统一推理入口,这里直接代理到 embed。"""
        return self.embed(source, _rid=_rid)

    def embed(self, source, _rid: str = "") -> dict:
        """图像 → embedding。返回 {success, embedding, dim, model}。"""
        t0 = time.time()
        try:
            image = load_image(source)
            vec = self._encode(image)
            self._track(t0, ok=True)
            return {"success": True, "embedding": vec, "dim": len(vec), "model": "qwen_vl"}
        except Exception as e:
            self._track(t0, ok=False)
            return self._error(e, "embed", rid=_rid)
