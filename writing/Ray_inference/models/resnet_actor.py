"""
ResNet50 — 图像分类 Actor（GPU 常驻）。
"""
from io import BytesIO

import ray
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

from models.base import BaseModelActor
from config import GPU_FRACTION_RESNET, ACTOR_MAX_RESTARTS, ACTOR_MAX_TASK_RETRIES


@ray.remote(max_restarts=ACTOR_MAX_RESTARTS, max_task_retries=ACTOR_MAX_TASK_RETRIES)
class ResNetActor(BaseModelActor):
    """ResNet50 图像分类，返回 ImageNet-1K top-5 类别。"""

    def __init__(self):
        super().__init__(model_name="ResNet50", gpu_fraction=GPU_FRACTION_RESNET)

    def _load_model(self):
        weights = ResNet50_Weights.IMAGENET1K_V2
        self._categories = weights.meta["categories"]
        self._transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        model = resnet50(weights=weights)
        model.to(self._device)
        model.eval()
        self._model = model

    def infer(self, image_bytes: bytes, top_k: int = 5) -> dict:
        """
        Args:
            image_bytes: 图片二进制数据
            top_k: 返回 top-k 预测
        Returns:
            {"success": True, "predictions": [{"label": ..., "score": ...}]}
        """
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            tensor = self._transform(image).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=-1)[0]
                topk = torch.topk(probs, min(top_k, len(self._categories)))

            predictions = [
                {"rank": i + 1, "label": self._categories[idx.item()],
                 "score": round(score.item(), 4)}
                for i, (score, idx) in enumerate(zip(topk.values, topk.indices))
            ]
            self._record_request()
            return {"success": True, "predictions": predictions}
        except Exception as e:
            return self._handle_error(e, "classify")
