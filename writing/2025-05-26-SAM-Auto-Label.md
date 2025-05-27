---
layout: mypost
title: 基于 SAM 的半自动标注数据
categories: 数据标注
extMath: true
images: true
address: 武汉🏯
show_footer_image: true
tags: [SAM,数据标注,label studio]
show: true
description: 之前有介绍使用SAM基本原理，本文主要介绍如何将SAM和Label Studio进行结合半自动的对数据进行标注
---

前面已经介绍了SAM的基本原理以及基本使用操作，本文主要介绍如何将SAM和自动化工具Label Studio进行结合对数据进行半自动化的进行标注，主要是从下面两个方面进行出发：1、**Point2Labl**：用户只需要在物体的区域内点一个点就能得到物体的掩码和边界框标注。2、**Bbox2Label**：用户只需要标注物体的边界框就能生成物体的掩码，社区的用户可以借鉴此方法，提高数据标注的效率。
[CV中常用Backbone-3：Clip/SAM原理以及代码操作](2025-05-18-Clip-sam.md)
[Label Studio](https://github.com/HumanSignal/label-studio)
## 使用教程
> 本文主要是在Linux系统上进行操作，对于Win可以直接使用WSL2然后操作

### 基本环境搭建[^1]
首先创建一个文件夹然后在这个文件夹里面进行操作
```cmd
mdkir SAM-Label-Studio
cd SAM-Label-Studio
```
通过Conda创建一个环境（可选操作，如果电脑上环境是一致的可以选择不这样）并且安装torch
```bash
conda create -n rtmdet-sam python=3.9 -y conda activate rtmdet-sam
pip install torch torchvision torchaudio
```
而后克隆label-studio项目：
```bash
git clone https://github.com/HumanSignal/label-studio.git
```
**下载需要的SAM模型权重**（Fron：[https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)）
```bash
cd SAM-Label-Studio/playground/label_anything
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
# 下载模型权重
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
对于模型权重，在[前文](./2025-05-18-Clip-sam.md) 中介绍了SAM有三类权重，不同的SAM模型权重（**模型权重大小依次递减**）：  
vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  
vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth  
vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
下载权重之后**安装 Label-Studio 和 label-studio-ml-backend**：
```bash
# Requires Python >=3.8
pip install label-studio
pip install label-studio-ml
```

### 服务启动
**首先**，启用 SAM 后端推理后再启动网页服务才可配置模型（个人习惯喜欢使用 `CUDA_VISIBLE_DEVICES=3`来指定显卡，也可以使用参数`device=cuda:1`）
```bash
CUDA_VISIBLE_DEVICES=3 label-studio-ml start sam --port 8003 --with \
sam_config=vit_b \
sam_checkpoint_file=./sam_vit_h_4b8939.pth \
out_mask=True \
out_bbox=True
# device=cuda:1
```
![image.png](https://s2.loli.net/2025/05/26/wKCxTc3sGEhn8bg.png)

**而后**，**新建一个终端窗口**启动 Label-Studio 网页服务
```bash
cd SAM-Label-Studio/playground/label_anything
label-studio start
```

### 前端配置


## 参考

[^1]: https://zhuanlan.zhihu.com/p/633699458
[^2]: https://labelstud.io/tutorials/segment_anything_model