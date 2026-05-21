import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from ultralytics import SAM, YOLO
from collections import defaultdict
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

os.environ["HF_DATASETS_CACHE"] = "/data/huangjie/"
os.environ["HF_HOME"] = "/data/huangjie/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/huangjie/"

def expand_boxes_tensor(boxes_filt, w, h, expand_ratio=0.1):
    boxes = boxes_filt.to(torch.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    bw, bh = x2 - x1, y1- y1
    dw, dh = bw * expand_ratio / 2, bh * expand_ratio / 2
    new_x1, new_y1 = (x1 - dw).clamp(min=0, max=w), (y1 - dh).clamp(min=0, max=h)
    new_x2, new_y2 = (x2 + dw).clamp(min=0, max=w), (y2 + dh).clamp(min=0, max=h)
    boxes_expanded = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)
    return boxes_expanded


def draw_segments_and_masks(image, segments_list, mask_list, output_path="annotated_image.jpg"):
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
            out_path = os.path.join(output_dir, f"{image_name}_mask_{info[0]['label']}{i}.png") if isinstance(info, tuple) else os.path.join(output_dir, f"{image_name}_mask_{i}.png")
            cv2.imwrite(out_path, info[1])
    elif mask_num== 'one':
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        for _ in mask_dict.items():
            mask += _[1][1]
        out_path = os.path.join(output_dir, f"{image_name}_mask.png")
        cv2.imwrite(out_path, mask)


def create_transparent_crop(image, mask, x, y, bw, bh):
    mask_blur = cv2.GaussianBlur(mask[y:y+bh, x:x+bw], (5, 5), 0)
    mask_norm = mask_blur.astype(np.float32) / 255.0
    rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
    rgba[..., :3] = image[y:y+bh, x:x+bw]
    rgba[..., 3] = (mask_norm * 255).astype(np.uint8)
    return rgba


class YOLOSamPipeline(nn.Module):
    def __init__(self, 
                 detection_model: str,
                 segmention_model: str,
                 detection_conig: dict={'conf': 0.25,'iou': 0.45,'imgsz': 640,'max_det': 300},
                 max_instance: int=100,
                 device= 'cuda' if torch.cuda.is_available else 'cpu'):
        super().__init__()
        self.device = device
        self.max_instance = max_instance
        # detection+ SAM
        self.det_config = detection_conig
        self.det_model = YOLO(detection_model, verbose=False)
        self.seg_model = SAM(segmention_model)

    #TODO: 完成切割实体时候最好是找到实体附近的一些阴影对象等
    def forward_det_seg(self, image):
        # detection+ SAM
        det_result = self.det_model(image, stream=True,
                                    conf= self.det_config['conf'],
                                    iou= self.det_config['iou'],
                                    imgsz= self.det_config['imgsz'],
                                    max_det= self.det_config['max_det'])
        segments_list, mask_list= [], []
        for result in det_result:
            class_ids = result.boxes.cls.int().tolist()  # noqa
            if class_ids:
                boxes = result.boxes.xyxy  # Boxes object for bbox outputs
                boxes_filt_expand = expand_boxes_tensor(boxes, 
                                                        result.orig_img.shape[1], 
                                                        result.orig_img.shape[0], 
                                                        0.1)

                sam_results = self.seg_model(result.orig_img, 
                                             bboxes=boxes_filt_expand, 
                                             verbose=False, 
                                             save=False, 
                                             device= self.device)
                segments = sam_results[0].masks.xyn
                boxes_sam = sam_results[0].boxes.xyxy
                conf_yolo = result.boxes.conf
                img_height, img_width = result.orig_img.shape[:2]
                for i, s in enumerate(segments):
                    if s.any():
                        segment = list(map(str, s.reshape(-1).tolist()))
                        s_absolute = s * np.array([img_width, img_height])
                        x_coords, y_coords = s_absolute[:, 0], s_absolute[:, 1]
                        x1, y1 = float(min(x_coords)), float(min(y_coords))
                        x2, y2 = float(max(x_coords)), float(max(y_coords))
                        bbox = [x1, y1, x2, y2]
                        segments_list.append({
                            'segment': segment,
                            'bbox': bbox
                        })
        
    
                for idx, (box, class_id, conf) in enumerate(zip(boxes_sam, class_ids, conf_yolo)):
                    mask_list.append({
                        'value': idx + 1,
                        'label': self.det_model.names[class_id],
                        'logit': float(conf.cpu()),
                        'box': box.cpu().tolist()
                    })
        
        return segments_list, mask_list


    def forward(self, image_path, instance=False):
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]
        segments_list, mask_list = self.forward_det_seg(image)

        if instance and segments_list and mask_list:
            os.makedirs('instance', exist_ok=True)
            instance_dict = defaultdict(list)

            for idx, (seg_info, mask_info) in enumerate(zip(segments_list, mask_list)):
                segment = np.array([float(x) for x in seg_info['segment']], dtype=np.float32).reshape(-1, 2)
                polygon = (segment * np.array([img_width, img_height])).astype(np.int32)

                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], 255)

                x, y, bw, bh = cv2.boundingRect(polygon)
                if bw <= 0 or bh <= 0:
                    continue

                rgba_crop = create_transparent_crop(image, mask, x, y, bw, bh)
                instance_dict[mask_info['label']].append((rgba_crop, rgba_crop.shape[0] * rgba_crop.shape[1]))

            for label, crops in instance_dict.items():
                sorted_crops = sorted(crops, key=lambda x: x[1], reverse=True)
                count = 0
                for i, (rgba_crop, _) in enumerate(sorted_crops[:2]):
                    while True:
                        save_path = os.path.join('instance', f'{label}-{count}.png')
                        if not os.path.exists(save_path) or count<= self.max_instance:
                            break
                        count += 1
                    cv2.imwrite(save_path, rgba_crop)
        return segments_list, mask_list

def worker(process_images, model):
    for image_path in process_images:
        segments_list, mask_list = model(image_path)
        img = cv2.imread(image_path)
        # draw_segments_and_masks(img, segments_list, mask_list)
        image_name = os.path.basename(image_path).split('.')[0]
        save_instance_masks(segments_list, img.shape[:2], 
                            mask_list= mask_list, 
                            image_name= image_name, 
                            output_dir= 'masks', 
                            mask_num= 'one')
        
def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

'''
TODO: 继续完成找到instance之后去对背景粘贴
'''
if __name__ == '__main__':
    image_path = './image/sa_325462.jpg'
    model = YOLOSamPipeline('./yolo11x-seg.pt',
                            './mobile_sam.pt')
    segments_list, mask_list = model(image_path, True)
    img = cv2.imread(image_path)
    # draw_segments_and_masks(img, segments_list, mask_list)
    save_instance_masks(segments_list, img.shape[:2], mask_list= mask_list, image_name='tmp', output_dir='masks', mask_num='one')

    # mp.set_start_method('spawn') 
    # pipeline = YOLOSamPipeline('./yolo11x-seg.pt','./mobile_sam.pt')
    # pipeline.to('cuda')

    # image_dir = './image/'
    # num_process = 4
    # image_paths = [os.path.join(image_dir, f)
    #                for f in os.listdir(image_dir)
    #                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    # chunks = chunkify(image_paths, n= num_process)

    # processes = []
    # for i in range(num_process):
    #     p = mp.Process(target=worker, args=(chunks[i], pipeline, 'cuda'))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()