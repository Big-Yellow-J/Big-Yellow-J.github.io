"""
SAM — 实体分割 Actor（GPU 常驻）。
"""
from io import BytesIO

import ray
import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from models.base import BaseModelActor
from config import GPU_FRACTION_SAM, SAM_MODEL_TYPE, SAM_CHECKPOINT, MODEL_CACHE_DIR, ACTOR_MAX_RESTARTS, ACTOR_MAX_TASK_RETRIES


@ray.remote(max_restarts=ACTOR_MAX_RESTARTS, max_task_retries=ACTOR_MAX_TASK_RETRIES)
class SAMActor(BaseModelActor):
    """SAM 实体分割，返回 mask + bbox + 面积。"""

    def __init__(self):
        super().__init__(model_name="SAM", gpu_fraction=GPU_FRACTION_SAM)

    def _load_model(self):
        # 自动下载 checkpoint 到缓存目录
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = MODEL_CACHE_DIR / SAM_CHECKPOINT
        if not checkpoint.exists():
            import urllib.request
            url = f"https://dl.fbaipublicfiles.com/segment_anything/{SAM_CHECKPOINT}"
            urllib.request.urlretrieve(url, checkpoint)

        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(checkpoint))
        sam.to(self._device)
        sam.eval()
        self._model = sam
        self._mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,       # 降低到 16 以节省显存/加速
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            min_mask_region_area=100,
        )

    def _warm_up(self):
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        _ = self._mask_generator.generate(dummy)

    def infer(self, image_bytes: bytes) -> dict:
        """
        Args:
            image_bytes: 图片二进制数据
        Returns:
            {"success": True,
             "masks": [{"segmentation": ..., "bbox": [x,y,w,h], "area": ..., "stability": ...}]}
        """
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            arr = np.array(image)
            if arr.shape[0] > 1024 or arr.shape[1] > 1024:
                image = image.resize((1024, int(1024 * arr.shape[0] / arr.shape[1])))
                arr = np.array(image)

            masks = self._mask_generator.generate(arr)
            results = []
            for m in masks:
                # 将 bool mask 转为 RLE 友好的格式（这里存 shape + indices 以节省传输）
                seg = m["segmentation"]
                indices = np.flatnonzero(seg).tolist()
                results.append({
                    "bbox": [round(v) for v in m["bbox"]],
                    "area": int(m["area"]),
                    "stability": round(m.get("stability_score", m.get("predicted_iou", 0)), 4),
                    "segmentation_shape": list(seg.shape),
                    "segmentation_indices": indices[:5000],  # 截断，避免过大数据传输
                })

            self._record_request()
            return {"success": True, "image_size": list(arr.shape), "num_masks": len(results), "masks": results}
        except Exception as e:
            return self._handle_error(e, "segment")
