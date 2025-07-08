---
layout: mypost
title: 深入浅出了解生成模型-6：常用基础模型与 Adapters等解析
categories: 生成模型
extMath: true
images: true
address: 长沙🌷
show_footer_image: true
tags:
- 生成模型
- diffusion model
- ControlNet
- T2I-Adapter
- SD
- SDVL
show: true
description: 本文主要介绍常用基础模型与 ControlNet 等控制权重解析
---

## Stable Diffusion系列
主要介绍SD以及SDXL两类模型，但是SD迭代版本挺多的（从1.2到3.5）因此本文主要重点介绍SD 1.5以及SDXL两个基座模型，以及两者之间的对比差异。
### SDv1.5 vs SDXL[^1]
> **SDv1.5**: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
> **SDXL**:https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

两者模型详细的模型结构：[SDv1.5--SDXL模型结构图](../Dio.drawio)，其中具体模型参数的对比如下：
1、CLIP编码器区别：
在SD1.5中选择的是**CLIP-ViT/L**（得到的维度为：768）而在SDXL中选择的是两个CLIP文本编码器：**OpenCLIP-ViT/G**（得到的维度为：1280）以及**CLIP-ViT/L**（得到维度为：768）

继续阅读：https://docs.google.com/document/d/144xqK_QSIbP-yJxQsztgRfm5S80rpK-UUrz0mqLy5gg/edit?tab=t.0

## Adapters
> https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference

此类方法是在完备的 DF 权重基础上，额外添加一个“插件”，保持原有权重不变。我只需修改这个插件，就可以让模型生成不同风格的图像。下面介绍的 ControlNet 和 T2I-Adapter，可以理解为在原始模型之外新增一个“生成条件”，通过修改这一条件即可灵活控制模型生成各种风格或满足不同需求的图像。

### ControlNet[^2]
> https://github.com/lllyasviel/ControlNet
> 建议直接阅读：[https://github.com/lllyasviel/ControlNet/discussions/categories/announcements](https://github.com/lllyasviel/ControlNet/discussions/categories/announcements) 来了解更加多细节

![](https://s2.loli.net/2025/07/07/6tDO7kAPgoupvr1.png)

ControlNet的处理思路就很简单，再左图中模型的处理过程就是直接通过：$y=f(x;\theta)$来生成图像，但是在ControlNet里面会将我们最开始的网络结构复制然后通过在其前后引入一个**zero-convolution**层来“指导”（$Z$）模型的输出也就是说将上面的生成过程变为：$y=f(x;\theta)+Z(f(x+Z(c;\theta_{z_1});\theta);\theta_{Z_2})$。通过冻结最初的模型的权重保持不变，保留了Stable Diffusion模型原本的能力；与此同时，使用额外数据对“可训练”副本进行微调，学习我们想要添加的条件。因此在最后我们的SD模型中就是如下一个结构：

![](https://s2.loli.net/2025/07/07/ELsxZXMYfjKeiNu.png)

在论文里面作者给出一个实际的测试效果可以很容易理解里面条件c（条件 𝑐就是提供给模型的显式结构引导信息，**用于在生成过程中精确控制图像的空间结构或布局**，一般来说可以是草图、分割图等）到底是一个什么东西，比如说就是直接给出一个“线稿”然后模型来输出图像。

![](https://s2.loli.net/2025/07/07/nRvXN9alxIo8yDE.png)

> **补充-1**：为什么使用上面这种结构
> 在[github](https://github.com/lllyasviel/ControlNet/discussions/188)上作者讨论了为什么要使用上面这种结构而非直接使用mlp等（作者给出了很多测试图像），最后总结就是：**这种结构好**
> **补充-2**：使用0卷积层会不会导致模型无法优化问题？
> 不会，因为对于神经网络结构大多都是：$y=wx+b$计算梯度过程中即使 $w=0$但是里面的 $x≠0$模型的参数还是可以被优化的

### T2I-Adapter[^3]

### 实际代码操作

## 参考
[^1]:https://arxiv.org/pdf/2307.01952
[^2]:https://arxiv.org/pdf/2302.05543
[^3]:https://arxiv.org/pdf/2302.08453