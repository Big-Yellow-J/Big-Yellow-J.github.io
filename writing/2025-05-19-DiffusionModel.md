---
layout: mypost
title: 深入浅出了解生成模型-4：一致性模型（consistency model）
categories: 生成模型
extMath: true
images: true
address: 武汉🏯
show_footer_image: true
tags: [生成模型,diffusion model,一致性模型]
show: true
description: 前面已经介绍了扩散模型，扩散模型往往需要多步才能生成较为满意的图像，但是可一致性模型可以通过几步生成图像，因此这里主要是介绍一致性模型（consistency model）基本原理以及代码实践。
---

前面已经介绍了扩散模型，在最后的结论里面提到一点：扩散模型往往需要多步才能生成较为满意的图像。不过现在有一种新的方式来加速（旨在通过少数迭代步骤）生成图像：**一致性模型（consistency model）**，因此这里主要是介绍一致性模型（consistency model）基本原理以及代码实践。


## 参考
