import os
import cv2
import json
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from ultralytics import YOLO
from scipy.ndimage import label
from transformers import pipeline
from collections import defaultdict
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

os.environ["HF_DATASETS_CACHE"] = "/data/huangjie/"
os.environ["HF_HOME"] = "/data/huangjie/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/huangjie/"

class BackGroundPastePipeline():
    def __init__(self, 
                 detection_model: str,
                 instance_dir: str,
                 paste_dir: str,
                 depth_model: str="LiheYoung/depth-anything-base-hf",
                 detection_conig: dict={'conf': 0.8,'iou': 0.45,'imgsz': 640,'max_det': 100},):
        ''''''
        self.det_config = detection_conig
        self.det_model = YOLO(detection_model, verbose=False)
        self.det_class_name = self.det_model.names
        self.depth_model = pipeline(task="depth-estimation", 
                                    model= depth_model,)
        if os.path.isdir(instance_dir):
            #TODO: 可以将instance进一步细化如：风格、图像完整度等
            self.instance_dict = defaultdict(list)
            for _ in os.listdir(instance_dir):
                self.instance_dict[_.split('-')[0]].append(os.path.join(instance_dir, _))
        os.makedirs(paste_dir, exist_ok= True)
        self.paste_dir = paste_dir
    
    def detection_forward(self, image):
        '''检测图片中所有的类别数量'''
        det_result = self.det_model(image, stream=True,
                                    conf= self.det_config['conf'],
                                    iou= self.det_config['iou'],
                                    imgsz= self.det_config['imgsz'],
                                    max_det= self.det_config['max_det'])
        class_instance_dict = {}

        for result in det_result:
            if result.boxes is None or result.boxes.cls.numel() == 0:
                continue

            class_ids = result.boxes.cls.int().tolist()
            confs = result.boxes.conf.tolist()
            boxes = result.boxes.xyxy.tolist()

            for idx, (class_id, box, conf) in enumerate(zip(class_ids, boxes, confs)):
                class_name = self.det_class_name[class_id]
                if class_name not in class_instance_dict:
                    class_instance_dict[class_name] = {}

                instance_idx = len(class_instance_dict[class_name]) + 1
                class_instance_dict[class_name][instance_idx] = [box, conf]
        return class_instance_dict
        
    def depth_forward(self, image_path):
        depth_np = np.array(self.depth_model(image_path)['depth'])
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6)
        return depth_norm

    def choose_instance(self, depth_norm, class_instance_dict, threshold=0.5):
        '''选择需要粘贴的instanc对象'''
        region_max = 0
        choose_instance = []
        labeled, num_features = label(depth_norm > threshold)
        for i in range(1, num_features + 1):
            region_mask = (labeled == i)
            ys, xs = np.where(region_mask)
            
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            if region_mask.sum()> region_max:
                x0, y0, x1, y1= x_min, y_min, x_max, y_max
        for instance_name, instance_info in class_instance_dict.items():
            for info in instance_info.values():
                tmp_x0, tmp_y0, tmp_x1, tmp_y1 = info[0]
                index_1 = x0<= tmp_x0<= x1 or y0<= tmp_y0<= y1 
                index_2 = x0<= tmp_x1<= x1 or y0<= tmp_y1<= y1
                if index_1 or index_2:
                    choose_instance.append((instance_name, info[0]))
        return choose_instance

    def image_paste(self, image, refer_instance, num=1):
        '''将实列粘贴到图片中'''
        def exam_image():
            '''检测图片中还有没有可以粘贴的区域'''
            pass
        def exam_overlap():
            '''检测图像中同一类别之间的重叠度'''
            pass
        
        if refer_instance is None:
            return image
        refer_info = [_ for _ in refer_instance if _[0] in self.instance_dict]
        if len(refer_info)==0:
            return image

        image = image.convert("RGBA")
        refer_name, refer_bbox = random.choices(refer_info)[0]
        instance_image_list = random.choices(self.instance_dict[refer_name], k= num)
        x0, y0, x1, y1 = refer_bbox
        for image_path in instance_image_list:
            print(image_path)
            instance_image = Image.open(image_path)
            instance_image = instance_image.convert("RGBA")
            x0, y0 = int(x0), int(y0)
            image.paste(instance_image, (x0, y0), mask=instance_image)
        return image

    def main(self, image_path):
        '''
        1、图片paste需要满足：不去过分影响到其他物体、paste看看能不能找到一个合适位置然后去粘贴
        '''
        # 获取instance以及 depth分布
        image = Image.open(image_path).convert("RGB")
        class_instance_dict = self.detection_forward(image)
        depth_norm = self.depth_forward(image)

        # 图片粘贴
        choose_instance = self.choose_instance(depth_norm, class_instance_dict)
        print(choose_instance)
        image = self.image_paste(image, choose_instance)
        image.save(os.path.join(self.paste_dir, 
                                f"{os.path.basename(image_path).split('.')[0]}.png"))

if __name__ == '__main__':
    from instance_split import *

    image_path = '../image/sa_325551.jpg'
    img = cv2.imread(image_path)
    pipeline_split = YOLOSamPipeline('./yolo11x-seg.pt','./mobile_sam.pt')
    pipeline_paste = BackGroundPastePipeline('./yolo11x-seg.pt', './instance/', './paste/')
    pipeline_paste.main(image_path)
    segments_list, mask_list = pipeline_split(image_path, False)
    draw_segments_and_masks(img, segments_list, mask_list)