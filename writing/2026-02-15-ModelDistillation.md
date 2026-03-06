---
layout: mypost
title: 深入浅出了解生成模型-10：模型蒸馏
categories: 生成模型
extMath: true
images: true
address: 长沙🌷
show_footer_image: true
tags:
- 生成模型
- diffusion model
- 模型蒸馏
description: 本文在介绍模型量化、不同attention方式及cache策略等模型生成加速方法基础上，重点阐述模型蒸馏技术，具体介绍分配匹配蒸馏（DMD）作为其中一种方式，以实现模型生成加速。
---

在最开始的文章中介绍了模型量化、使用不同attention方式、cache策略去对模型生成进行加速，这里主要介绍几种模型蒸馏的方式去加速模型生成。
## 模型蒸馏简单概述
## 分配匹配蒸馏（DMD）
对于DMD[^1][^2]方法原理如下（DMD1的算法流程）：
![](https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260219145134921.png)
对于上诉算法流程图简单描述
## 参考
[^1]: [Improved Distribution Matching Distillation for Fast Image Synthesis](https://tianweiy.github.io/dmd2/)
[^2]: [One-step Diffusion with Distribution Matching Distillation](https://tianweiy.github.io/dmd/)