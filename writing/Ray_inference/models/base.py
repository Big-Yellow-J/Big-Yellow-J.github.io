"""
模型 Actor 基类：封装 GPU 常驻、健康上报、优雅降级。
注意：基类不添加 @ray.remote（ABC 无法被 Ray 实例化），
装饰器由各子类分别添加。
"""
import time
import traceback
from abc import ABC, abstractmethod

import torch

from config import ACTOR_MAX_RESTARTS, ACTOR_MAX_TASK_RETRIES, TASK_TIMEOUT_SEC


class BaseModelActor(ABC):
    """所有模型 Actor 的基类，提供 GPU 常驻与健康检查能力。"""

    def __init__(self, model_name: str, gpu_fraction: float):
        self.model_name = model_name
        self.gpu_fraction = gpu_fraction
        self._model = None
        self._device = None
        self._last_inference_time = 0.0
        self._total_requests = 0
        self._alive = True

        self._setup_device()
        self._load_model()
        self._warm_up()

    # ---------- 子类必须实现 ----------
    @abstractmethod
    def _load_model(self):
        """加载模型权重到 self._model。"""
        ...

    @abstractmethod
    def infer(self, **kwargs):
        """同步推理入口，子类实现。"""
        ...

    # ---------- 设备管理 ----------
    def _setup_device(self):
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        # 限制显存增长（可选，防止 OOM）
        if self._device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.95)

    # ---------- GPU 预热 ----------
    def _warm_up(self):
        """用 dummy 输入跑一次前向，避免首次推理卡顿。"""
        try:
            dummy = torch.randn(1, 3, 224, 224, device=self._device)
            with torch.no_grad():
                _ = self._model(dummy)
            self._last_inference_time = time.time()
        except Exception:
            pass  # 预热失败不阻塞启动

    # ---------- 健康检查 ----------
    def health_check(self) -> dict:
        """Ray Actor 健康检查接口。"""
        try:
            gpu_ok = True
            if self._device.type == "cuda":
                gpu_ok = torch.cuda.is_available()

            return {
                "model": self.model_name,
                "alive": self._alive and gpu_ok,
                "device": str(self._device),
                "gpu_fraction": self.gpu_fraction,
                "total_requests": self._total_requests,
                "last_inference_sec_ago": time.time() - self._last_inference_time
                if self._last_inference_time else -1,
            }
        except Exception as e:
            return {"model": self.model_name, "alive": False, "error": str(e)}

    def get_model_info(self) -> dict:
        return {
            "model": self.model_name,
            "device": str(self._device),
            "total_requests": self._total_requests,
        }

    # ---------- 辅助 ----------
    def _record_request(self):
        self._total_requests += 1
        self._last_inference_time = time.time()

    def _handle_error(self, e: Exception, context: str = ""):
        traceback.print_exc()
        return {"success": False, "error": f"[{self.model_name}] {context}: {str(e)}"}
