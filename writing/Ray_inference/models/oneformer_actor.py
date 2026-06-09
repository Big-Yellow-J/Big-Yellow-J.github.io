"""OneFormer 通用分割 Actor:支持 instance / semantic / panoptic 三种任务。"""
import base64
import time
from io import BytesIO
from pathlib import Path
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
from utils.image_loader import load_image

_TASKS = ("instance", "semantic", "panoptic")
_MASK_FORMATS = ("png_b64", "rle")


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
        # 强制本地加载:ONEFORMER_MODEL 指向项目内 weights/ 子目录,跳过 hub HEAD 验证。
        # 缺失时给出明确指引,不静默回落到在线下载。
        if not Path(ONEFORMER_MODEL).is_dir():
            raise RuntimeError(
                f"OneFormer weights not found at {ONEFORMER_MODEL}. "
                f"run `python main.py prepare` to snapshot weights to weights/."
            )
        self._processor = OneFormerProcessor.from_pretrained(ONEFORMER_MODEL, local_files_only=True)
        self._model = (
            OneFormerForUniversalSegmentation
            .from_pretrained(ONEFORMER_MODEL, local_files_only=True)
            .to(self._device)
            .eval()
        )

        # dynamic=True 让多种输入分辨率共用一份编译产物,避免反复 recompile
        try:
            self._model = torch.compile(self._model, dynamic=True)
        except Exception:
            pass
        self._id2label = self._model.config.id2label

    def _warm_up(self):
        try:
            self._predict(Image.new("RGB", (640, 640)), "instance", False, 0.5, 0.5, "png_b64")
        except Exception:
            pass

    def infer(
        self,
        source,
        task: Literal["instance", "semantic", "panoptic"] = "instance",
        return_mask: bool = False,
        mask_format: Literal["png_b64", "rle"] = "png_b64",
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        _rid: str = "",
    ) -> dict:
        """对图像执行分割,返回 instances 或 semantic 摘要。

        Args:
            source: 任意 image_loader 支持的输入。
            task: 分割任务类型。
            return_mask: True 时每个实例返回 mask;格式由 mask_format 决定。
            mask_format: png_b64 / rle。
            score_threshold: instance/panoptic 实例置信度阈值。
            mask_threshold: mask 二值化阈值。
            _rid: 上游透传的 request id,出错时写入日志。
        Returns:
            instance/panoptic: {"success": True, "instances": [{"label","score","bbox","area","mask_*?"}]}
            semantic:          {"success": True, "labels":    [{"label","area"}, ...]}
        """
        t0 = time.time()
        try:
            if task not in _TASKS:
                raise ValueError(f"task must be one of {_TASKS}")
            if mask_format not in _MASK_FORMATS:
                raise ValueError(f"mask_format must be one of {_MASK_FORMATS}")
            image = load_image(source)
            out = self._predict(image, task, return_mask, score_threshold, mask_threshold, mask_format)
            self._track(t0, ok=True)
            return {"success": True, **out}
        except Exception as e:
            self._track(t0, ok=False)
            return self._error(e, f"segment[{task}]", rid=_rid)

    def _predict(
        self, image: Image.Image, task: str, return_mask: bool,
        score_threshold: float, mask_threshold: float, mask_format: str,
    ) -> dict:
        """前向 + 任务对应的后处理,分发到 _semantic_summary 或 _instances。"""
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
        result = post(
            outputs, target_sizes=target_size,
            threshold=score_threshold, mask_threshold=mask_threshold,
        )[0]
        return {"instances": self._instances(result, return_mask, mask_format)}

    def _semantic_summary(self, seg: np.ndarray) -> list:
        """semantic:按类别像素数倒序汇总。"""
        ids, counts = np.unique(seg, return_counts=True)
        return [
            {"label": self._id2label.get(int(i), str(int(i))), "area": int(c)}
            for i, c in sorted(zip(ids, counts), key=lambda x: -x[1])
        ]

    def _instances(self, result, return_mask: bool, mask_format: str) -> list:
        """instance/panoptic:抽取每个实例的 label/score/bbox/area + 可选 mask。"""
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
                if mask_format == "png_b64":
                    item["mask_png_b64"] = _mask_to_png_b64(mask)
                else:
                    item["mask_rle"] = _mask_to_rle(mask)
            out.append(item)
        return out


def _mask_to_png_b64(mask: np.ndarray) -> str:
    """二值 mask → 1 通道 PNG → base64 字符串。"""
    buf = BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _mask_to_rle(mask: np.ndarray) -> dict:
    """二值 mask → COCO 风格 RLE(列优先)。返回 {"size":[H,W], "counts":[...]}。"""
    flat = mask.astype(np.uint8).flatten(order="F")        # COCO 是列优先
    # 0/1 交替的 run-length:首段固定从 0 开始,若首像素=1 则补 0
    padded = np.concatenate([[0], flat, [0]])
    diffs = np.diff(padded)
    starts = np.where(diffs != 0)[0]
    counts = np.diff(starts).tolist()
    if flat[0] == 1:
        counts = [0] + counts
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}
