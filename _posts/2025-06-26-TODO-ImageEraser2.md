---
layout: mypost
title: 图像擦除论文综述-2：擦除类型数据集构建(1)
categories: 图像消除
address: 武汉🏯
extMath: true
show_footer_image: true
tags: [diffusion model,图像消除]
description: 本文主要介绍几篇图像擦除论文中如何构建一个image-mask数据集
---

## SmartEraser: Remove Anything from Images using Masked-Region Guidance
> [SmartEraser: Remove Anything from Images using Masked-Region Guidance](https://openaccess.thecvf.com/content/CVPR2025/papers/Jiang_SmartEraser_Remove_Anything_from_Images_using_Masked-Region_Guidance_CVPR_2025_paper.pdf)
> CVPR-2025

## Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways
> [Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Erase_Diffusion_Empowering_Object_Removal_Through_Calibrating_Diffusion_Pathways_CVPR_2025_paper.pdf)
> CVPR-2025


## OmniEraser: Remove Objects and Their Effects in Images with Paired Video-Frame Data
> https://pris-cv.github.io/Omnieraser/

通过视频来获取（mask-image）数据集，具体操作流程如下：

![image.png](https://s2.loli.net/2025/06/26/LYclhNt4WmgRJpz.webp)

首先获取一段视频 $\mathbf{V}$ 通过 **混合高斯算法**（MOG）去检查视频中移动的物体以及静止的物体这样一来就可以得到两部分内容：Background和Foreground而后通过计算两部分之间的MSE（$MSE(V_i^{fg}, V_j^{bg})$）就可以得到source-image和 target-image对。对于mask内容直接通过 *GroundDINO+SAM2* 算法来构建mask这样一来就可以得到：foreground-image，mask，background-image。模型算法这是直接去微调 `FLUX.1-dev`

![image.png](https://s2.loli.net/2025/06/26/tcIhCEDeuGf3UXv.webp)

实际测试效果（使用prompt为：`'There is nothing here.'`）

| 原图 | Mask | 结果 | 测试细节 |
|-----|------|------|--------|
|![sa_324952.jpg](https://s2.loli.net/2025/06/26/znSUtwamOk9r47I.webp)|![sa_324952-0.jpg](https://s2.loli.net/2025/06/26/QXdWSb46FREakVN.webp) |![sa_324952.jpg](https://s2.loli.net/2025/06/26/7pdgqO45CbDhluw.webp) | |
|![sa_325886.jpg](https://s2.loli.net/2025/06/26/Bw4D9pEi7McULbv.webp)|![sa_325886-1.jpg](https://s2.loli.net/2025/06/26/P8mKbFdTqxZ19Yn.webp) |![sa_325886.jpg](https://s2.loli.net/2025/06/26/89qmPaIY3tW1uUv.webp) | |
|![sa_324501.jpg](https://s2.loli.net/2025/06/26/kxZjsRLSvpX96ne.webp)|![sa_324501-2.jpg](https://s2.loli.net/2025/06/26/bHMSowgfXm4sqO5.webp) |![sa_324501.jpg](https://s2.loli.net/2025/06/26/GV9n6u1As3ZoqkJ.webp) | |
|![sa_324930.jpg](https://s2.loli.net/2025/06/26/SA8rRFMc4Zjlp21.webp)|![sa_324930-1.jpg](https://s2.loli.net/2025/06/26/fQdXwRUCg5JVjs6.webp) |![sa_324930.jpg](https://s2.loli.net/2025/06/26/Npr6tT9A75gwcY4.webp) |![image.png](https://s2.loli.net/2025/06/26/xfBuX4RniAj7Z2D.webp)|

