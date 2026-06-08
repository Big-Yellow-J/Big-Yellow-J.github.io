"""ResNet50 图像分类 Actor。"""
import ray
import torch
import torchvision.transforms as T
from torchvision.models import ResNet50_Weights, resnet50

from config import ACTOR_MAX_CONCURRENCY, ACTOR_MAX_RESTARTS, ACTOR_MAX_TASK_RETRIES, GPU_FRACTION_RESNET
from models.base import BaseModelActor
from models.image_loader import load_image


@ray.remote(
    max_restarts=ACTOR_MAX_RESTARTS,
    max_task_retries=ACTOR_MAX_TASK_RETRIES,
    max_concurrency=ACTOR_MAX_CONCURRENCY,
)
class ResNetActor(BaseModelActor):
    """ResNet50,返回 ImageNet-1K 类别。"""

    def __init__(self):
        super().__init__(model_name="ResNet50", gpu_fraction=GPU_FRACTION_RESNET)

    def _load_model(self):
        weights = ResNet50_Weights.IMAGENET1K_V2
        self._categories = weights.meta["categories"]
        self._transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._model = resnet50(weights=weights).to(self._device).eval()

    def infer(self, source, top_k: int = 5) -> dict:
        try:
            image = load_image(source)
            tensor = self._transform(image).unsqueeze(0).to(self._device)

            with torch.inference_mode():
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
