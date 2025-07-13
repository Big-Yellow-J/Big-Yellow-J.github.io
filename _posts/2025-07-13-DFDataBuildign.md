---
layout: mypost
title: 深入浅出了解生成模型-7：构建扩散模型数据集
categories: 生成模型
extMath: true
images: true
address: 武汉🗂️
show_footer_image: true
tags:
- 生成模型
- diffusion model
- ControlNet
- 数据集构建
show: true
description: 对比Stable Diffusion SD 1.5与SDXL模型差异，SDXL采用双CLIP编码器（OpenCLIP-ViT/G+CLIP-ViT/L）提升文本理解，默认1024x1024分辨率并优化处理；介绍ControlNet（空间结构控制）、T2I-Adapter、DreamBooth（解决语言偏离）等Adapters，实现风格迁移与高效生成。
---

一般来说扩散模型中训练需要大量数据（其实无论是扩散模型还是大语言模型拼的可能都是数据以及算力），如何构建一个高质量的数据就十分重要，基本数据获取：一般而言（就图像类型数据）可以直接通过爬虫等方式直接获取到数据，但是！获取到数据后处理才是最重要，比如说如何筛选出高质量的、如何去对数据打标签（人力标签是不现实的，但是人力筛选是现实的，毕竟很多优秀模型对不同任务都有SOTA，但是泛化能力如何有待商榷）是至关重要，当然知道如何打标签还不行还需要知道如何去构建满足下游任务的数据。就扩散模型而言在数据集构建上运用如下方法。
## 文生图任务
文生图任务，此类任务数据上就会比较简单，基本就是 `text2img`配对即可，所以假设在获取得到合适的image之后只需要去对图像去生成描述，那么就有如下的合适的方法：
### 基础标记方法
直接通过Clip或者BLIP-2去生成图像的描述
