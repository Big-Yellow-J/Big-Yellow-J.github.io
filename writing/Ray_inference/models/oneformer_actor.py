"""OneFormer 通用分割 Actor:支持 instance / semantic / panoptic 三种任务。"""
import base64
import time
from io import BytesIO
from typing import Literal

import numpy as np
import ray
import torch
from PIL import Image
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

from config import (
    ACTOR_MAX_CONCURRENCY,
    ACTOR_MAX_RESTARTS,
    ACTOR_MAX_TASK_RETRIES,
    GPU_FRACTION_ONEFORMER,
    ONEFORMER_MODEL,
)
from models.base import BaseModelActor
from models.image_loader import load_image

_TASKS = ("instance", "semantic", "panoptic")


@ray.remote(
    max_restarts=ACTOR_MAX_RESTARTS,
    max_task_retries=ACTOR_MAX_TASK_RETRIES,
    max_concurrency=max(1, ACTOR_MAX_CONCURRENCY // 2),    # OneFormer 显存大,降一档并发
)
class OneFormerActor(BaseModelActor):
    """OneFormer 通用分割,默认 instance(实体分割)。"""

    def __init__(self):
        super().__init__(model_name="OneFormer", gpu_fraction=GPU_FRACTION_ONEFORMER)

    def _load_model(self):
        self._processor = OneFormerProcessor.from_pretrained(ONEFORMER_MODEL)
        self._model = (
            OneFormerForUniversalSegmentation
            .from_pretrained(ONEFORMER_MODEL)
            .to(self._device)
            .eval()
        )
        # dynamic=True 让多种输入分辨率共用一份编译产物,避免反复 recompile
        self._id2label = self._model.config.id2label

    def _warm_up(self):
        try:
            self._predict(Image.new("RGB", (640, 640)), "instance", return_mask=False)
        except Exception:
            pass

    def infer(
        self,
        source,
        task: Literal["instance", "semantic", "panoptic"] = "instance",
        return_mask: bool = False,
    ) -> dict:
        """对图像执行分割。

        Args:
            source: 任意 image_loader 支持的输入。
            task: "instance"(默认实体分割)/ "semantic" / "panoptic"。
            return_mask: True 时把每个实例的二值 mask 以 PNG base64 一并返回(响应体显著变大)。
        Returns:
            instance/panoptic: {"success": True, "instances": [{"label","score","bbox","area","mask_png_b64?"}]}
            semantic:          {"success": True, "labels":    [{"label","area"}, ...]}
        """
        t0 = time.time()
        try:
            if task not in _TASKS:
                raise ValueError(f"task must be one of {_TASKS}")
            image = load_image(source)
            out = self._predict(image, task, return_mask)
            self._track(t0, ok=True)
            return {"success": True, **out}
        except Exception as e:
            self._track(t0, ok=False)
            return self._error(e, f"segment[{task}]")

    def _predict(self, image: Image.Image, task: str, return_mask: bool) -> dict:
        """执行单次前向 + 任务对应的后处理。"""
        # task_inputs 必须传入与任务匹配的 token,OneFormer 据此切换查询头
        inputs = self._processor(
            images=image, task_inputs=[task], return_tensors="pt",
        ).to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        target_size = [image.size[::-1]]   # PIL.size=(W,H),OneFormer 需要 (H,W)

        if task == "semantic":
            seg = self._processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_size,
            )[0].cpu().numpy()
            return {"labels": self._semantic_summary(seg)}

        post = (
            self._processor.post_process_instance_segmentation if task == "instance"
            else self._processor.post_process_panoptic_segmentation
        )
        result = post(outputs, target_sizes=target_size)[0]
        return {"instances": self._instances(result, return_mask)}

    def _semantic_summary(self, seg: np.ndarray) -> list:
        """semantic 模式:按类别像素数倒序汇总。"""
        ids, counts = np.unique(seg, return_counts=True)
        return [
            {"label": self._id2label.get(int(i), str(int(i))), "area": int(c)}
            for i, c in sorted(zip(ids, counts), key=lambda x: -x[1])
        ]

    def _instances(self, result, return_mask: bool) -> list:
        """instance/panoptic 模式:抽取每个实例的 label/score/bbox/area。"""
        seg_map = result["segmentation"].cpu().numpy()
        out = []
        for info in result["segments_info"]:
            mask = (seg_map == info["id"])
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            item = {
                "label": self._id2label.get(int(info["label_id"]), str(info["label_id"])),
                "score": round(float(info.get("score", 1.0)), 4),
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "area": int(mask.sum()),
            }
            if return_mask:
                item["mask_png_b64"] = _mask_to_b64(mask)
            out.append(item)
        return out


def _mask_to_b64(mask: np.ndarray) -> str:
    """二值 mask → PNG → base64,便于客户端贴回原图。"""
    buf = BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
