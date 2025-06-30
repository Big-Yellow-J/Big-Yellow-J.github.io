---
layout: mypost
title: 图像擦除论文-3：FreeCompose
categories: 图像消除
address: 武汉🏯
extMath: true
show_footer_image: true
tags:
- diffusion model
- 图像消除
description: 图像擦除是图像生成模型重要应用，本文介绍CVPR-2025相关的SmartEraser、Erase Diffusion、OmniEraser模型，涵盖数据集构建（实体过滤、混合高斯算法MOG）、关键技术（语义分割SAM、CLIP、IoU、alpha
  blending、GroundDINO+SAM2）及模型优化（输入改进、mask处理、微调FLUX.1-dev）等内容。
---

图像生成模型应用系列——图像擦除：
[图像擦除论文-1：PixelHacker、PowerPanint、Attentive Eraser](https://www.big-yellow-j.top/posts/2025/06/11/ImageEraser1.html)
[图像擦除论文-2：SmartEraser、Erase Diffusion、OmniEraser](https://www.big-yellow-j.top/posts/2025/06/26/ImageEraser2.html)

## FreeCompose
> [FreeCompose:Generic Zero-Shot Image Composition with Diffusion Prior](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02529.pdf)
> https://www.yongshengyu.com/OmniPaint-Page/
> ECCV-24

### 1、模型结构

![image.png](https://s2.loli.net/2025/06/30/2SFfl34tXRDTh9V.png)

论文主要实现3个子任务：1、Object removal；2、Image harmonization（将两幅图片进行组合）；3、Semantic image composition（通过指定形状去修改目标图像中物品形状）
**Session-1：object removal**。这部分操纵和[之前介绍](https://www.big-yellow-j.top/posts/2025/06/11/ImageEraser1.html)的的Attentive Eraser、Erase Diffusion模型相似，都是在计算KV时候将mask结果加到进去让模型只关注特定内容。
**Session-2：Image harmonization**

## OmniPaint
> [OmniPaint: Mastering Object-Oriented Editing via Disentangled Insertion-Removal Inpainting](https://arxiv.org/pdf/2503.08677)
> ICCV-25


