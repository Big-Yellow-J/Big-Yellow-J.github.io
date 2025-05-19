---
layout: mypost
title: CV中常用Backbone-：Clip/SAM原理以及代码操作
categories: Backbone
extMath: true
images: true
address: 武汉🏯
show_footer_image: true
tags: [cv-backbone,多模态,multimodal]
show: False
description: 
---

前面已经介绍了简单的视觉编码器，这里主要介绍多模态中使用比较多的两种视觉编码器：1、Clip；2、SAM
## SAM
SAM已经出了两个版本分别是：SAM v1和SAM v2这里对这两种分别进行解释，并且着重了解一下他的数据集是怎么构建的（毕竟很多论文里面都会提到直接用SAM作为一种数据集生成工具）
### SAM v1
> https://arxiv.org/pdf/2304.02643
> 官方Blog：[Introducing Segment Anything: Working toward the first foundation model for image segmentation](https://ai.meta.com/blog/segment-anything-foundation-model-image-segmentation/)

![](https://s2.loli.net/2025/05/18/GLP4R1db8eYvoOM.png)

结构上还是比较简单，首先在 **Image Encoder**：选择的是[MAE](https://www.big-yellow-j.top/posts/2025/01/18/CV-Backbone.html#:~:text=768-,MAE%20%E4%B8%BB%E8%A6%81%E6%93%8D%E4%BD%9C%E6%B5%81%E7%A8%8B,-1%E3%80%81patch)；**Prompt Encoder**：

### SAM v2
> https://arxiv.org/pdf/2408.00714


## Clip

## 参考
1、Segment Anything Model (SAM): a new AI model from Meta AI that can "cut out" any object, in any image, with a single click