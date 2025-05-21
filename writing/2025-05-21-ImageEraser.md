---
layout: mypost
title: 图像擦除论文研究综述
categories: Backbone
extMath: true
images: true
address: 武汉🏯
show_footer_image: true
tags: [多模态,multimodal,diffusion model]
show: False
description: 
---

# Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways
> CVPR2025, Taobao

模型整体结构，使用的还是latent diffusion model：
![image.png](https://s2.loli.net/2025/05/21/CjMS9fPuYLgDAwe.png)

## Chain-Rectifying Optimization paradigm（CRO）
![image.png](https://s2.loli.net/2025/05/21/DczHhU9XE2PBSg6.png)

**数据集构建**、将传统的Eraser使用df模型直接预测mask中noise内容，这这会导致模型生成“意料之外的内容”，本文构建一个 **链条数据**：对于原始的图像$x_o^{ori}$，首先从图像中“扣出”图像，而后将扣出来的图像去和原始图像进行组合（找到空白地方而后随机进行数据增强（选择，缩放等））得到$x_p^{obj}$其中动态混合操作：$\bar{x_t}^{mix}=(1-\lambda_t)x_o^{ori}+ \lambda_tx_o^{obj}$其中$\lambda_t$是一个动态变化的（可以理解为添加的内容逐渐 *消失*）。因此就可以将df模型加噪过程变为：
$$
x_t^{mix}=\sqrt{\bar{\alpha_t}}\bar{x_t}^{mix}+\sqrt{1-\bar{\alpha}}\epsilon
$$

> 区别传统的只是将$x_t$变成一个动态变化的

**优化目标**、给的状态$x_t^{mix}$模型的预测目标是$x_{t-\gamma}^{mix}$（$\gamma$代表先前步骤）得到：
![image.png](https://s2.loli.net/2025/05/21/vNgR4WBHOZjiqnx.png)

最后模型的优化目标为:
![image.png](https://s2.loli.net/2025/05/21/FRiVTsEz2YWoUva.png)

## Self-Rectifying Attention（SRA）
在使用mask时候会导致模型在最开始的时候更加多的是去关注mask区域内容而不是背景信息。因此改进了注意力计算方式，直接将mask内容拿出来而后去添加到注意力计算里面也就是$QK^T$过程中使用mask
![image.png](https://s2.loli.net/2025/05/21/FtAYSDR3bsi2rmO.png)