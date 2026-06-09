"""Actor 基类:GPU 常驻 + 指标 + 健康上报 + 结构化错误日志(写到 tmp/ray_log/<today>/<model>.log)。"""
import time
import traceback
from abc import ABC, abstractmethod

import torch

from utils.logging_setup import setup_logger


class BaseModelActor(ABC):
    """所有模型 Actor 的公共能力,子类只需实现 _load_model 与 infer。"""

    def __init__(self, model_name: str, gpu_fraction: float):
        """初始化:加载模型 → 预热 → 准备指标计数器 + 独立 logger。

        Args:
            model_name: actor 显示名(也是日志文件名 base)。
            gpu_fraction: 仅观测用,真实配额由 ray.remote(num_gpus=...) 决定。
        """
        self.model_name = model_name
        self.gpu_fraction = gpu_fraction
        # actor 在独立进程,setup_logger 会写自己专属的日志文件
        self._log = setup_logger(model_name.lower())
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._total_requests = 0
        self._total_errors = 0
        self._total_latency_ms = 0.0
        self._last_inference_ts = 0.0
        torch.backends.cudnn.benchmark = True
        self._load_model()
        self._warm_up()
        self._log.info("actor ready device=%s gpu_fraction=%s", self._device, gpu_fraction)

    @abstractmethod
    def _load_model(self):
        """加载权重,将模型放到 self._device,设为 eval 模式。"""

    @abstractmethod
    def infer(self, source, **kwargs) -> dict:
        """子类实现的推理入口,统一返回 {"success": bool, ...}。"""

    def _warm_up(self):
        """子类按需覆盖,默认无操作。"""

    def health_check(self) -> dict:
        """返回当前 actor 的健康与累计指标。"""
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
        """累计一次调用的耗时与成功标志。"""
        self._total_requests += 1
        if not ok:
            self._total_errors += 1
        self._total_latency_ms += (time.time() - t_start) * 1000.0
        self._last_inference_ts = time.time()

    def _error(self, e: Exception, context: str, rid: str = "") -> dict:
        """统一错误返回 + 写入 actor 日志(带 rid)。"""
        traceback.print_exc()
        rid_part = f" rid={rid}" if rid else ""
        msg = f"[{self.model_name}] {context}{rid_part}: {e}"
        self._log.error(msg)
        return {"success": False, "error": msg}
