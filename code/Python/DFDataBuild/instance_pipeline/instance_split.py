import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from tqdm import tqdm
from ultralytics import SAM, YOLO
from collections import defaultdict

from utils import *

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

os.environ["HF_DATASETS_CACHE"] = "/data/huangjie/"
os.environ["HF_HOME"] = "/data/huangjie/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/huangjie/"

class YOLOSamPipeline(nn.Module):
    def __init__(self, 
                 detection_model: str,
                 segmention_model: str,
                 detection_conig: dict={'conf': 0.8,'iou': 0.45,'imgsz': 640,'max_det': 100},
                 max_instance: int=100,
                 device= 'cuda' if torch.cuda.is_available else 'cpu'):
        '''
        sam支持：['sam_h.pt', 'sam_l.pt', 'sam_b.pt', 'mobile_sam.pt', 'sam2_t.pt', 'sam2_s.pt', 'sam2_b.pt', 'sam2_l.pt', 'sam2.1_t.pt', 'sam2.1_s.pt', 'sam2.1_b.pt', 'sam2.1_l.pt']
        '''
        super().__init__()
        self.device = device
        self.max_instance = max_instance
        # detection+ SAM
        self.det_config = detection_conig
        self.seg_model = SAM(segmention_model)
        self.det_model = YOLO(detection_model, verbose=False)

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
            class_ids = result.boxes.cls.int().tolist()
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
                index = 2 if len(sorted_crops)>=2 else len(sorted_crops)
                for i, (rgba_crop, _) in enumerate(sorted_crops[:index]):
                    count = 0
                    save_path = os.path.join('instance', f'{label}-{random.randint(0, self.max_instance)}.png')
                    while os.path.exists(save_path)== False:
                        if count<= self.max_instance:
                            cv2.imwrite(save_path, rgba_crop)
                            break
                        else:
                            count +=1
        return segments_list, mask_list

def worker(process_images, model, instance=False):
    with tqdm(total= len(process_images)) as pbar:
        for image_path in process_images:
            segments_list, mask_list = model(image_path, instance)
            img = cv2.imread(image_path)
            # draw_segments_and_masks(img, segments_list, mask_list)
            image_name = os.path.basename(image_path).split('.')[0]
            save_instance_masks(segments_list, img.shape[:2], 
                                mask_list= mask_list, 
                                image_name= image_name, 
                                output_dir= 'masks', 
                                mask_num= 'one')
            pbar.update(1)
        
def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

'''
TODO: 继续完成找到instance之后去对背景粘贴
'''
if __name__ == '__main__':
    # import time
    # image_path = '../image/sa_324930.jpg'
    # s_time = time.time()
    # model = YOLOSamPipeline('./yolo11x-seg.pt',
    #                         './mobile_sam.pt')
    # segments_list, mask_list = model(image_path, True)
    # e_time = time.time()
    # print(f"Split Segment Used Time: {e_time- s_time}")
    # img = cv2.imread(image_path)
    # draw_segments_and_masks(img, segments_list, mask_list)
    # save_instance_masks(segments_list, img.shape[:2], mask_list= mask_list, image_name='tmp', output_dir='masks', mask_num='one')

    mp.set_start_method('spawn') 
    pipeline = YOLOSamPipeline('./yolo11x-seg.pt','./mobile_sam.pt')
    pipeline.to('cuda')

    image_dir = '../image/'
    num_process = 4
    image_paths = [os.path.join(image_dir, f)
                   for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    chunks = chunkify(image_paths, n= num_process)

    processes = []
    for i in range(num_process):
        p = mp.Process(target=worker, args=(chunks[i], pipeline, 'cuda'))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()