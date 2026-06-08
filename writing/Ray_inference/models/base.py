"""Actor 基类:GPU 常驻、统一指标、健康上报、错误封装。"""
import time
import traceback
from abc import ABC, abstractmethod

import torch


class BaseModelActor(ABC):
    """所有模型 Actor 的公共能力,子类只需实现 _load_model 与 infer。"""

    def __init__(self, model_name: str, gpu_fraction: float):
        """初始化:加载模型 → 预热 → 准备指标计数器。

        Args:
            model_name: actor 显示名,会出现在日志/health/metrics 标签里。
            gpu_fraction: 仅用于观测,真实配额由 ray.remote(num_gpus=...) 决定。
        """
        self.model_name = model_name
        self.gpu_fraction = gpu_fraction
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._total_requests = 0
        self._total_errors = 0
        self._total_latency_ms = 0.0
        self._last_inference_ts = 0.0
        torch.backends.cudnn.benchmark = True   # 输入尺寸稳定时显著加速卷积
        self._load_model()
        self._warm_up()

    @abstractmethod
    def _load_model(self):
        """加载权重,将模型放到 self._device,设为 eval 模式。"""

    @abstractmethod
    def infer(self, source, **kwargs) -> dict:
        """子类实现的推理入口,统一返回 {"success": bool, ...}。"""

    def _warm_up(self):
        """子类按需覆盖,默认无操作。"""

    def health_check(self) -> dict:
        """返回当前 actor 的健康与累计指标。

        Returns:
            dict: model / alive / device / 累计请求 / 累计错误 / 平均延迟 / 距上次推理秒数。
        """
        avg_ms = (
            self._total_latency_ms / self._total_requests
            if self._total_requests else 0.0
        )
        return {
            "model": self.model_name,
            "alive": True,
            "device": str(self._device),
            "gpu_fraction": self.gpu_fraction,
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "avg_latency_ms": round(avg_ms, 2),
            "last_inference_sec_ago": (
                time.time() - self._last_inference_ts
                if self._last_inference_ts else -1
            ),
        }

    def _track(self, t_start: float, ok: bool):
        """累计一次调用的耗时与成功标志,供 health/metrics 暴露。"""
        self._total_requests += 1
        if not ok:
            self._total_errors += 1
        self._total_latency_ms += (time.time() - t_start) * 1000.0
        self._last_inference_ts = time.time()

    def _error(self, e: Exception, context: str) -> dict:
        """统一错误返回(traceback 打印到日志,响应只带摘要)。"""
        traceback.print_exc()
        return {"success": False, "error": f"[{self.model_name}] {context}: {e}"}
