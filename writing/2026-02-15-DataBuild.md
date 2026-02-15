---
layout: mypost
title: 深度学习数据构建-1：图像生成模型数据构建
categories: 数据构建
extMath: true
images: true
address: 长沙🌷
show_footer_image: true
tags:
- 生成模型
- 数据构建
show: true
stickie: true
---
本文主要去收集看到的所有论文中关于**生成模型数据集构建过程**
## Z-Image数据构建过程
在Z-Image[^1]中对于比较常规的数据处理方式比如使用多模态模型对图像描述，使用llm去做prompt augmentation等不同的是在Z-Image中的数据构建方式如下：
**1、Data Profiling Engine**
这个过程是一个多维度（元数据 → 技术质量 → 信息密度 → 美学语义 → 图文对齐）的自动化数据体检 + 精选系统，通过 pHash 去重、自研质量/AIGC/VLM 模型打分、CN-CLIP 对齐过滤 + VLM 一体化生成丰富 caption，最终从海量脏数据中提炼出高质量、强对齐、偏中文文化的训练语料
**数据收集过程**
![image.png](https://test.fukit.cn/autoupload/f/vkB-Pqb1HNqojGbyEgL65tiO_OyvX7mIgxFBfDMDErs/Blog-Image/image.png)
上面为Z-Image中整个数据收集过程在对图像/文本进行编码之后去对图像进行粗过滤主要是去重+基于规则方式进行过来，而后去进行图像-文本匹配方式最后得到匹配的文本-图像数据。而对于去重方式主要是使用pHash方式进行图像数据去重、同时，Z-Image利用了borderpixel variance与瞬时JPEG重编码后的BPP(bytes-per-pixel)作为图像复杂度的一种表示方式，有效过滤低信息熵样本、使用CN-CLIP方式去计算图-文匹配度
## 数据去重方法
### pHash
**缩小尺寸**：将图片缩小为32\*32大小。**灰度化处理**：计算DCT，并选取左上角8*8的矩阵。DCT是一种特殊的傅立叶变换，将图片从像素域变换为频率域，并且DCT矩阵从左上角到右下角代表越来越高频率的系数，但是除左上角外，其他地方的系数为0或接近0，因此只保留左上角的低频区域。**计算DCT均值**。**哈希值计算**：将每个DCT值，与平均值进行比较。大于或等于平均值，记为1，小于平均值，记为0，由此生成二进制数组。**图片配对**：计算汉明距离
```python
# pip install imagehash
from PIL import Image
import imagehash
import os
from tqdm import tqdm
import pandas as pd

def compute_phashes(folder_path, extensions=('.jpg', '.jpeg', '.png')):
    results = []
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    
    print(f"找到 {len(image_paths)} 张图片，开始计算 pHash...")
    
    for path in tqdm(image_paths, desc="计算 pHash"):
        try:
            img = Image.open(path).convert('RGB')  # 统一转 RGB 避免模式问题
            ph = imagehash.phash(img)              # 默认 8×8 = 64
            results.append({
                'path': path,
                'phash_hex': str(ph),
                'phash_int': int(ph),
            })
        except Exception as e:
            print(f"处理失败 {path}: {e}")
            continue
    df = pd.DataFrame(results)
    df.to_csv('phashes.csv', index=False)
    return df
folder = "xxx"
df = compute_phashes(folder)
# 去重/找相似
from collections import defaultdict
phash_to_paths = defaultdict(list)
for _, row in df.iterrows():
    phash_to_paths[row['phash_hex']].append(row['path'])
duplicates = {k: v for k, v in phash_to_paths.items() if len(v) > 1}
```
## 参考
[^1]: [Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer](https://arxiv.org/pdf/2511.22699)