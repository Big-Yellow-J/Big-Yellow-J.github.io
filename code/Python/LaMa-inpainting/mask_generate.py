import numpy as np
import random
import cv2

class FaceMaskGenerator:
    def __init__(self, mode="center", image_size=512, target_ratio=0.5, tol=0.05):
        """
        mode: 'rect' | 'center' | 'freeform'
        image_size: mask 图像大小
        target_ratio: 目标遮挡比例（默认 0.5 = 50%）
        tol: 容忍误差范围（默认 ±5%）
        """
        self.mode = mode
        self.image_size = image_size
        self.target_ratio = target_ratio
        self.tol = tol

    def __call__(self):
        h, w = self.image_size, self.image_size

        if self.mode == "rect":
            mask = self.rect_mask(h, w)
        elif self.mode == "center":
            mask = self.center_rect_mask(h, w)
        elif self.mode == "freeform":
            mask = self.free_form_mask(h, w)
        else:
            raise ValueError(f"Unknown mask mode: {self.mode}")

        return mask.astype(np.uint8)

    # ----------------- 精确矩形 -----------------
    def rect_mask(self, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        # 目标面积
        target_area = int(self.target_ratio * h * w)
        side = int(np.sqrt(target_area))

        x1 = random.randint(0, w - side)
        y1 = random.randint(0, h - side)
        x2, y2 = x1 + side, y1 + side

        mask[y1:y2, x1:x2] = 1
        return mask

    # ----------------- 精确中心矩形 -----------------
    def center_rect_mask(self, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        target_area = int(self.target_ratio * h * w)

        rect_w = int(np.sqrt(target_area * random.uniform(0.8, 1.2)))
        rect_h = int(target_area / rect_w)

        cx, cy = w // 2, h // 2
        x1 = max(0, cx - rect_w // 2 + random.randint(-w//10, w//10))
        y1 = max(0, cy - rect_h // 2 + random.randint(-h//10, h//10))
        x2, y2 = min(w, x1 + rect_w), min(h, y1 + rect_h)

        mask[y1:y2, x1:x2] = 1
        return mask

    # ----------------- 自由形状（面积缩放修正） -----------------
    def free_form_mask(self, h, w, max_lines=20):
        mask = np.zeros((h, w), dtype=np.uint8)
        num_lines = random.randint(10, max_lines)

        for _ in range(num_lines):
            x, y = random.randint(w//4, w*3//4), random.randint(h//4, h*3//4)
            for _ in range(random.randint(10, 30)):
                angle = random.uniform(0, 2*np.pi)
                length = random.randint(20, 80)
                brush_w = random.randint(10, 30)

                x2 = np.clip(x + int(length * np.cos(angle)), 0, w-1)
                y2 = np.clip(y + int(length * np.sin(angle)), 0, h-1)

                cv2.line(mask, (x, y), (x2, y2), 1, brush_w)
                x, y = x2, y2

        # 计算实际比例
        ratio = mask.mean()
        target = self.target_ratio
        if ratio > 0:  
            scale = np.sqrt(target / ratio)
            new_h, new_w = min(int(h * scale), h), min(int(w * scale), w)
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            final = np.zeros((h, w), dtype=np.uint8)
            y_off = (h - new_h) // 2
            x_off = (w - new_w) // 2

            final[y_off:y_off+new_h, x_off:x_off+new_w] = mask_resized[:min(new_h, h - y_off), :min(new_w, w - x_off)]
            mask = final

        return mask
