import os
import cv2
import random
import numpy as np
import torch

def expand_boxes_tensor(boxes_filt, w, h, expand_ratio=0.1):
    '''扩展检查框'''
    boxes = boxes_filt.to(torch.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    bw, bh = x2 - x1, y1- y1
    dw, dh = bw * expand_ratio / 2, bh * expand_ratio / 2
    new_x1, new_y1 = (x1 - dw).clamp(min=0, max=w), (y1 - dh).clamp(min=0, max=h)
    new_x2, new_y2 = (x2 + dw).clamp(min=0, max=w), (y2 + dh).clamp(min=0, max=h)
    boxes_expanded = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)
    return boxes_expanded


def draw_segments_and_masks(image, segments_list, mask_list, output_path="annotated_image.jpg"):
    '''图中标记出切割实体以及bbox'''
    annotated_image = image.copy()
    img_height, img_width = image.shape[:2]
    
    for seg_info in segments_list:
        segment = np.array([float(coord) for coord in seg_info['segment']]).reshape(-1, 2)
        segment_absolute = (segment * np.array([img_width, img_height])).astype(np.int32)
        cv2.polylines(annotated_image, [segment_absolute], isClosed=True, color=(0, 0, 255), thickness=2)
        x1, y1, x2, y2 = map(int, seg_info['bbox'])
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    for mask_info in mask_list:
        x1, y1, x2, y2 = map(int, mask_info['box'])
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{mask_info['label']} ({mask_info['logit']:.2f})"
        cv2.putText(annotated_image, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(f"./{output_path}", annotated_image)


def save_instance_masks(segments_list, image_size, image_name, mask_list=None, output_dir='masks', mask_num='all'):
    '''存储mask  
    mask_num: all所有的mask都单独存储 one:所有的mask存储一起 random:随机存储一张mask
    '''
    os.makedirs(output_dir, exist_ok=True)
    img_height, img_width = image_size
    mask_dict = {}

    for i, seg_info in enumerate(segments_list):
        segment = np.array([float(x) for x in seg_info['segment']], dtype=np.float32).reshape(-1, 2)
        polygon = (segment * np.array([img_width, img_height])).astype(np.int32)

        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], color=255)
        mask_dict[i]= (mask_list[i], mask)

    if mask_num== 'all':
        for i, info in enumerate(mask_dict):
            out_path = os.path.join(output_dir, f"{image_name}_mask_{info[0]['label']}_{i}.png") if isinstance(info, tuple) else os.path.join(output_dir, f"{image_name}_mask_{i}.png")
            cv2.imwrite(out_path, mask)
    elif mask_num== 'random':
        half_size = len(mask_dict) // 2
        random_half = random.sample(list(mask_dict.keys()), half_size)
        for i in random_half:
            info = mask_dict[i]
            out_path = os.path.join(output_dir, 
                                    f"{image_name}_mask_{info[0]['label']}{i}.png") if isinstance(info, tuple) else os.path.join(output_dir, f"{image_name}_mask_{i}.png")
            cv2.imwrite(out_path, info[1])
    elif mask_num== 'one':
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        for _ in mask_dict.items():
            mask += _[1][1]
        out_path = os.path.join(output_dir, f"{image_name}_mask.png")
        cv2.imwrite(out_path, mask)

def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def create_transparent_crop(image, mask, x, y, bw, bh):
    mask_blur = cv2.GaussianBlur(mask[y:y+bh, x:x+bw], (5, 5), 0)
    mask_norm = mask_blur.astype(np.float32) / 255.0
    rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
    rgba[..., :3] = image[y:y+bh, x:x+bw]
    rgba[..., 3] = (mask_norm * 255).astype(np.uint8)
    return rgba
