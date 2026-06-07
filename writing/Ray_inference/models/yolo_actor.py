"""
YOLO — 目标检测 Actor（GPU 常驻）。
"""
from io import BytesIO

import ray
import numpy as np
from PIL import Image
from ultralytics import YOLO

from models.base import BaseModelActor
from config import GPU_FRACTION_YOLO, YOLO_MODEL_NAME, ACTOR_MAX_RESTARTS, ACTOR_MAX_TASK_RETRIES


@ray.remote(max_restarts=ACTOR_MAX_RESTARTS, max_task_retries=ACTOR_MAX_TASK_RETRIES)
class YOLOActor(BaseModelActor):
    """YOLOv8 目标检测，返回 bbox + 类别 + 置信度。"""

    def __init__(self):
        super().__init__(model_name="YOLOv8", gpu_fraction=GPU_FRACTION_YOLO)

    def _load_model(self):
        from ultralytics import YOLO as _YOLO
        self._model = _YOLO(YOLO_MODEL_NAME)

    def _warm_up(self):
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self._model(dummy, verbose=False)

    def infer(self, image_bytes: bytes, conf_threshold: float = 0.25) -> dict:
        """
        Args:
            image_bytes: 图片二进制数据
            conf_threshold: 置信度阈值
        Returns:
            {"success": True, "detections": [{"bbox": [x1,y1,x2,y2], "class": ..., "conf": ...}]}
        """
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            arr = np.array(image)

            results = self._model(arr, conf=conf_threshold, verbose=False)
            detections = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "class": results[0].names[int(box.cls[0])],
                        "conf": round(float(box.conf[0]), 4),
                    })

            self._record_request()
            return {"success": True, "detections": detections}
        except Exception as e:
            return self._handle_error(e, "detect")
