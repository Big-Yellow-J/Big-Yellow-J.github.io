"""
CLIP — 图文匹配 / 简短对话 Actor（GPU 常驻）。
"""
from io import BytesIO

import ray
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from models.base import BaseModelActor
from config import GPU_FRACTION_CLIP, CLIP_MODEL_NAME, ACTOR_MAX_RESTARTS, ACTOR_MAX_TASK_RETRIES


@ray.remote(max_restarts=ACTOR_MAX_RESTARTS, max_task_retries=ACTOR_MAX_TASK_RETRIES)
class CLIPActor(BaseModelActor):
    """CLIP 图文匹配 & 简短对话。

    对话通过图文匹配实现：将对话历史编码为文本，计算与候选回复的相似度。
    同时支持 zero-shot 图像分类和图文相似度计算。
    """

    def __init__(self):
        super().__init__(model_name="CLIP", gpu_fraction=GPU_FRACTION_CLIP)

    def _load_model(self):
        self._model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self._device)
        self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self._model.eval()

    # ---------- 图文相似度 ----------
    def image_text_similarity(self, image_bytes: bytes, texts: list[str]) -> dict:
        """计算图片与多个文本的相似度。"""
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            inputs = self._processor(
                text=texts, images=image, return_tensors="pt", padding=True
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits_per_image = outputs.logits_per_image[0]
                scores = torch.softmax(logits_per_image, dim=-1)

            results = [{"text": t, "score": round(s.item(), 4)}
                       for t, s in zip(texts, scores)]
            self._record_request()
            return {"success": True, "similarities": results}
        except Exception as e:
            return self._handle_error(e, "similarity")

    # ---------- 简短对话 ----------
    def chat(self, image_bytes: bytes, user_message: str,
             candidates: list[str] | None = None) -> dict:
        """
        简短对话：给图片和用户消息，从候选中选最佳回复。
        若未提供 candidates，自动生成几个通用候选。

        Args:
            image_bytes: 图片二进制
            user_message: 用户文本
            candidates: 候选回复列表（可选）
        Returns:
            {"success": True, "reply": ..., "candidates_with_scores": [...]}
        """
        try:
            if candidates is None:
                candidates = [
                    f"这张图片展示了 {user_message}",
                    f"图片中可以看到 {user_message}",
                    "这是一张普通的图片。",
                    "图片中有一些有趣的细节。",
                    "让我仔细看看这张图片……",
                ]

            # 将 user_message 与图片一起做匹配，选最佳候选
            combined = [f"问题：{user_message} | 回答：{c}" for c in candidates]

            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            inputs = self._processor(
                text=combined, images=image, return_tensors="pt", padding=True
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                scores = outputs.logits_per_image[0].softmax(dim=-1)

            best_idx = int(scores.argmax())
            scored = [{"reply": c, "score": round(s.item(), 4)}
                      for c, s in zip(candidates, scores)]
            scored.sort(key=lambda x: x["score"], reverse=True)

            self._record_request()
            return {"success": True, "reply": candidates[best_idx], "candidates_with_scores": scored}
        except Exception as e:
            return self._handle_error(e, "chat")

    # ---------- 统一入口 ----------
    def infer(self, image_bytes: bytes, texts: list[str] | None = None,
              mode: str = "similarity") -> dict:
        """
        Args:
            image_bytes: 图片二进制
            texts: 文本列表
            mode: "similarity" | "chat"
        """
        if mode == "chat" and texts:
            return self.chat(image_bytes, texts[0])
        return self.image_text_similarity(image_bytes, texts or ["一张图片"])
