"""YOLOv8 目标检测 Actor。"""
import time

import numpy as np
import ray
from ultralytics import YOLO

from config import (
    ACTOR_MAX_CONCURRENCY,
    ACTOR_MAX_RESTARTS,
    ACTOR_MAX_TASK_RETRIES,
    GPU_FRACTION_YOLO,
    YOLO_MODEL,
)
from models.base import BaseModelActor
from utils.image_loader import load_image


@ray.remote(
    max_restarts=ACTOR_MAX_RESTARTS,
    max_task_retries=ACTOR_MAX_TASK_RETRIES,
    max_concurrency=ACTOR_MAX_CONCURRENCY,
)
class YOLOActor(BaseModelActor):
    """YOLOv8 目标检测,返回 bbox + 类别 + 置信度。"""

    def __init__(self):
        super().__init__(model_name="YOLOv8", gpu_fraction=GPU_FRACTION_YOLO)

    def _load_model(self):
        # 不要套 torch.compile:YOLO 内部含 numpy 预处理(letterbox),
        # dynamo trace 不兼容,首次推理会抛 fake tensor 类型错误。
        # 想加速请用 YOLO 官方的 model.export(format="engine") → TensorRT。
        self._model = YOLO(YOLO_MODEL)

    def _warm_up(self):
        try:
            self._model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        except Exception:
            pass

    def infer(
        self,
        source,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        imgsz: int = 640,
        classes: list = None,
        agnostic_nms: bool = False,
        _rid: str = "",
    ) -> dict:
        """对图像做目标检测。

        Args:
            source: 任意 image_loader 支持的输入。
            conf: 置信度阈值。
            iou: NMS IoU 阈值。
            max_det: 单图最大检测数。
            imgsz: 推理输入尺寸,32 倍数最佳。
            classes: 只检测指定类索引(COCO 80 类),None 表示全部。
            agnostic_nms: True 时类无关 NMS。
        Returns:
            {"success": True, "detections": [{"bbox":[x1,y1,x2,y2], "class": str, "conf": float}, ...]}
        """
        t0 = time.time()
        try:
            arr = np.array(load_image(source))
            results = self._model(
                arr, conf=conf, iou=iou, max_det=max_det, imgsz=imgsz,
                classes=classes, agnostic_nms=agnostic_nms, verbose=False,
            )
            detections = []
            if results and results[0].boxes is not None:
                names = results[0].names
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "class": names[int(box.cls[0])],
                        "conf": round(float(box.conf[0]), 4),
                    })
            self._track(t0, ok=True)
            return {"success": True, "detections": detections}
        except Exception as e:
            self._track(t0, ok=False)
            return self._error(e, "detect", rid=_rid)
