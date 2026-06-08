"""模型 Actor 基类:GPU 常驻、健康上报、统一错误处理。"""
import time
import traceback
from abc import ABC, abstractmethod

import torch


class BaseModelActor(ABC):
    """所有模型 Actor 的基类。"""

    def __init__(self, model_name: str, gpu_fraction: float):
        self.model_name = model_name
        self.gpu_fraction = gpu_fraction
        self._model = None
        self._device = self._pick_device()
        self._last_inference_time = 0.0
        self._total_requests = 0

        torch.backends.cudnn.benchmark = True
        self._load_model()
        self._warm_up()

    @abstractmethod
    def _load_model(self):
        ...

    @abstractmethod
    def infer(self, source, **kwargs) -> dict:
        ...

    def _pick_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _warm_up(self):
        try:
            dummy = torch.randn(1, 3, 224, 224, device=self._device)
            with torch.inference_mode():
                self._model(dummy)
        except Exception:
            pass

    def health_check(self) -> dict:
        return {
            "model": self.model_name,
            "alive": True,
            "device": str(self._device),
            "gpu_fraction": self.gpu_fraction,
            "total_requests": self._total_requests,
            "last_inference_sec_ago": (
                time.time() - self._last_inference_time
                if self._last_inference_time else -1
            ),
        }

    def _record_request(self):
        self._total_requests += 1
        self._last_inference_time = time.time()

    def _handle_error(self, e: Exception, context: str) -> dict:
        traceback.print_exc()
        return {"success": False, "error": f"[{self.model_name}] {context}: {e}"}
