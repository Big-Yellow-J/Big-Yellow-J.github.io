---
layout: mypost
title: 图像擦除论文-2：SmartEraser、Erase Diffusion、OmniEraser
categories: 图像消除
address: 武汉🏯
extMath: true
show_footer_image: true
tags:
- diffusion model
- 图像消除
description: 本文围绕图像擦除展开，涉及SmartEraser、Erase Diffusion、OmniEraser等模型。SmartEraser有合成数据集构建步骤；Erase
  Diffusion改进模型输入等；OmniEraser通过视频获取数据集并微调模型，各模型在数据集构建与模型结构上有不同改进及测试情况。
---

图像生成模型应用系列——图像擦除：
[图像擦除论文-1：PixelHacker、PowerPanint等](https://www.big-yellow-j.top/posts/2025/06/11/ImageEraser1.html)
[图像擦除论文-2：擦除类型数据集构建(1)](https://www.big-yellow-j.top/posts/2025/06/26/ImageEraser2.html)

## SmartEraser
> [SmartEraser: Remove Anything from Images using Masked-Region Guidance](https://openaccess.thecvf.com/content/CVPR2025/papers/Jiang_SmartEraser_Remove_Anything_from_Images_using_Masked-Region_Guidance_CVPR_2025_paper.pdf)
> CVPR-2025

### 1、数据集构建

### 2、模型结构测试效果

## Erase Diffusion
> [Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Erase_Diffusion_Empowering_Object_Removal_Through_Calibrating_Diffusion_Pathways_CVPR_2025_paper.pdf)
> https://github.com/longtaojiang/SmartEraser
> CVPR-2025

### 1、数据集构建

![](https://s2.loli.net/2025/06/28/7ojzDsGYEHKc3XC.webp)

合成数据集构建思路上使用思路是：实体过滤背景检测而后将两部分进行组合。**Step-1：实体过滤**：直接通过语义分割模型（如SAM等）分割出实体之后，通过CLIP计算实体的score并且过滤掉过大/小的分割实体（保留5%-95%）进而获得需要粘贴的实体；**Step-2：过滤背景图片**：直接通过计算分辨率等从COCONut 和SAM-1B数据集中挑选出合适图片背景；**Step-3：图片组合**：首先将实体和背景图像中相同实体大小保持一致，而后通过计算我分割实体$c_1$ 以及背景中的实体 $c_i$之间的IoU：$R_1$，以及保证需要粘贴实体在整个背景中的位置（保证不超出背景图片）：$R_2$而后取两部分交集得到图像应该插入的合理位置。最后通过 `alpha blending`将两部分图像（实体+背景）进行组合。

### 2、模型结构测试效果
![](https://s2.loli.net/2025/06/28/1Wv6XI9bD87UTBs.webp)

论文主要就是将模型的输入进行改进：将模型图像输入由$[mask, image\bigodot (1-mask)]$ 改为 $[mask, image]$，除此之外将DF模型的condition改进（将图像编码嵌入到文本编码中）：$[\text{CLIP-TextEncoder(text)}, \text{MLP}(\text{Image}\bigodot \text{Mask})]$。除此之外就是将mask由“规则”（实体分割是规则的）变为“不规则”（将实体分割mask进行额外处理如膨胀处理等）最后测试效果是：

![](https://s2.loli.net/2025/06/28/G8HOtWoB1bhYEqP.webp)
> ME：将mask变不规则；RG：改变模型输入；VG：将图像编码嵌入到clip文本编码中

## OmniEraser
> https://pris-cv.github.io/Omnieraser/

### 1、数据集构建
通过视频来获取（mask-image）数据集，具体操作流程如下：

![image.png](https://s2.loli.net/2025/06/26/LYclhNt4WmgRJpz.webp)

首先获取一段视频 $\mathbf{V}$ 通过 **混合高斯算法**（MOG）去检查视频中移动的物体以及静止的物体这样一来就可以得到两部分内容：Background和Foreground而后通过计算两部分之间的MSE（$MSE(V_i^{fg}, V_j^{bg})$）就可以得到source-image和 target-image对。对于mask内容直接通过 *GroundDINO+SAM2* 算法来构建mask这样一来就可以得到：foreground-image，mask，background-image。模型算法这是直接去微调 `FLUX.1-dev`

### 2、模型结构测试效果
![image.png](https://s2.loli.net/2025/06/26/tcIhCEDeuGf3UXv.webp)

实际测试效果（使用prompt为：`'There is nothing here.'`）

| 原图 | Mask | 结果 | 测试细节 |
|-----|------|------|--------|
|![sa_324952.jpg](https://s2.loli.net/2025/06/26/znSUtwamOk9r47I.webp)|![sa_324952-0.jpg](https://s2.loli.net/2025/06/26/QXdWSb46FREakVN.webp) |![sa_324952.jpg](https://s2.loli.net/2025/06/26/7pdgqO45CbDhluw.webp) | |
|![sa_325886.jpg](https://s2.loli.net/2025/06/26/Bw4D9pEi7McULbv.webp)|![sa_325886-1.jpg](https://s2.loli.net/2025/06/26/P8mKbFdTqxZ19Yn.webp) |![sa_325886.jpg](https://s2.loli.net/2025/06/26/89qmPaIY3tW1uUv.webp) | |
|![sa_324501.jpg](https://s2.loli.net/2025/06/26/kxZjsRLSvpX96ne.webp)|![sa_324501-2.jpg](https://s2.loli.net/2025/06/26/bHMSowgfXm4sqO5.webp) |![sa_324501.jpg](https://s2.loli.net/2025/06/26/GV9n6u1As3ZoqkJ.webp) | |
|![sa_324930.jpg](https://s2.loli.net/2025/06/26/SA8rRFMc4Zjlp21.webp)|![sa_324930-1.jpg](https://s2.loli.net/2025/06/26/fQdXwRUCg5JVjs6.webp) |![sa_324930.jpg](https://s2.loli.net/2025/06/26/Npr6tT9A75gwcY4.webp) |![image.png](https://s2.loli.net/2025/06/26/xfBuX4RniAj7Z2D.webp)|