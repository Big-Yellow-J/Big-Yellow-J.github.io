import PIL.Image
import torch
import torchvision
import PIL
import tqdm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from Model.pix2pixhd import MultidilatedConv, MultiDilatedGlobalGenerator, GlobalGenerator, FFCResnetBlock
from Model.ffc import FFCResNetGenerator
from data_loader import InpaintingDataset
from collate import default_collate

# def load_image_and_mask(image_path, mask_path, image_size=(512, 512)):
#     transform_image = transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.ToTensor()  # 输出 shape: [C, H, W], C=3
#     ])
    
#     transform_mask = transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.ToTensor()  # 输出 shape: [1, H, W]
#     ])
    
#     # 读取图像和mask
#     image = Image.open(image_path).convert("RGB")
#     mask = Image.open(mask_path).convert("L")  # 灰度图
    
#     image_tensor = transform_image(image)  # [3, H, W]
#     mask_tensor = transform_mask(mask)     # [1, H, W]
    
#     combined = torch.cat([image_tensor, mask_tensor], dim=0)  # [4, H, W]
#     return combined.unsqueeze(0)

# def move_to_device(obj, device):
#     if isinstance(obj, nn.Module):
#         return obj.to(device)
#     if torch.is_tensor(obj):
#         return obj.to(device)
#     if isinstance(obj, (tuple, list)):
#         return [move_to_device(el, device) for el in obj]
#     if isinstance(obj, dict):
#         return {name: move_to_device(val, device) for name, val in obj.items()}
#     raise ValueError(f'Unexpected type {type(obj)}')

# # model = MultiDilatedGlobalGenerator(input_nc= 4, output_nc= 3)
# model = FFCResNetGenerator(input_nc= 4, output_nc= 3, n_blocks= 18, add_out_act= 'sigmoid',
#                            init_conv_kwargs= {'ratio_gin': 0, 'ratio_gout': 0, 'enable_lfu': False})
# state = torch.load('/data/huangjie/big-lama/models/best.ckpt', map_location='cuda:0', weights_only=False)
# model.load_state_dict(state['state_dict'], strict=False)
# model.to("cuda:0")
# model.eval()

# dataset = InpaintingDataset('./image/')
# for img_i in tqdm.trange(len(dataset)):
#     batch = default_collate([dataset[img_i]])
#     with torch.no_grad():
#         batch = move_to_device(batch, "cuda:0")
#         batch['mask'] = (batch['mask'] > 0) * 1
#         batch['mask'] = batch['mask'].float()
#         batch['image'] = F.interpolate(batch['image'], size=512, mode='bilinear', align_corners=False)
#         batch['mask'] = F.interpolate(batch['mask'], size=512, mode='nearest')

#         img = batch['image']
#         mask = batch['mask']
#         masked_img = img * (1 - mask)
#         masked_img = torch.cat([masked_img, mask], dim=1)

#         batch['predicted_image'] = model(masked_img)
#         batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']

#         cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
#         unpad_to_size = batch.get('unpad_to_size', None)
#         if unpad_to_size is not None:
#             orig_height, orig_width = unpad_to_size
#             cur_res = cur_res[:orig_height, :orig_width]

# cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
# cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
# cv2.imwrite('./tmp.jpg', cur_res)

import os
print(len(os.listdir('/data/huangjie/data/Mulan/GT')))