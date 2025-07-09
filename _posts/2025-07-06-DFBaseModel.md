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
description: 本文重点对比Stable Diffusion SD 1.5与SDXL基座模型，分析CLIP编码器差异（SDXL采用OpenCLIP-ViT/G与CLIP-ViT/L拼接，文本理解能力更强）、图像输出维度（SDXL默认1024x1024并使用refiner模型）及SDXL分辨率与裁剪优化策略；同时介绍Adapters中的ControlNet（通过zero-convolution指导输出）和T2I-Adapter（特征相加控制生成）。
---

## Stable Diffusion系列
主要介绍SD以及SDXL两类模型，但是SD迭代版本挺多的（从1.2到3.5）因此本文主要重点介绍SD 1.5以及SDXL两个基座模型，以及两者之间的对比差异。
### SDv1.5 vs SDXL[^1]
> **SDv1.5**: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
> **SDXL**:https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

两者模型详细的模型结构：[SDv1.5--SDXL模型结构图](../Dio.drawio)，其中具体模型参数的对比如下：
**1、CLIP编码器区别**：
在SD1.5中选择的是**CLIP-ViT/L**（得到的维度为：768）而在SDXL中选择的是两个CLIP文本编码器：**OpenCLIP-ViT/G**（得到的维度为：1280）以及**CLIP-ViT/L**（得到维度为：768）在代码中对于两个文本通过编码器处理之后SDXL直接通过cat方式拼接：`prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)` 也就是说最后得到的维度为：[..,..,1280+768]。最后效果很明显：**SDXL对于文本的理解能力大于SD1.5**
**2、图像输出维度区别**：
再SD1.5中的默认输出是：512x512而再SDXL中的默认输出是：1024x1024，如果希望将SD1.5生成的图像处理为1024x1024可以直接通过超分算法来进行处理，除此之外在SDXL中还会使用一个refiner模型（和Unet的结构相似）来强化base模型（Unet）生成的效果。
**3、SDXL论文中的技术细节**：
* 1、**图像分辨率优化策略**。

数据集中图像的尺寸图像利用率问题（选择512x512舍弃256x256就会导致图像大量被舍弃）如果通过超分辨率算法将图像就行扩展会放大伪影，这些伪影可能会泄漏到最终的模型输出中，例如，导致样本模糊。（The second method, on the other hand, usually introduces upscaling artifacts which may leak into the final model outputs, causing, for example, blurry samples.）作者做法是：**训练阶段**直接将原始图像的分辨率 $c=(h_{org},w_{org})$作为一个条件，通过傅里叶特征编码而后加入到time embedding中，**推理阶段**直接将分辨率作为一个条件就行嵌入，进而实现：**当输入低分辨率条件时，生成的图像较模糊；在不断增大分辨率条件时，生成的图像质量不断提升。**
![image.png](https://s2.loli.net/2025/07/09/pMcLmdHThu2CnNx.webp)

* 2、**图像裁剪优化策略**

直接统一采样裁剪坐标top和cleft（分别指定从左上角沿高度和宽度轴裁剪的像素数量的整数），并通过傅里叶特征嵌入将它们作为调节参数输入模型，类似于上面描述的尺寸调节
第1，2点代码中的处理方式为：
```python
def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    passed_add_embed_dim = (
        self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
    )
    expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
    ...
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids
```

> **推荐阅读**：
> 1、[SDv1.5-SDXL-SD3生成效果对比](https://www.magicflow.ai/showcase/sd3-sdxl-sd1.5)

## Adapters
> https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference

此类方法是在完备的 DF 权重基础上，额外添加一个“插件”，保持原有权重不变。我只需修改这个插件，就可以让模型生成不同风格的图像。下面介绍的 ControlNet 和 T2I-Adapter，可以理解为在原始模型之外新增一个“生成条件”，通过修改这一条件即可灵活控制模型生成各种风格或满足不同需求的图像。

### ControlNet[^2]
> https://github.com/lllyasviel/ControlNet
> 建议直接阅读：[https://github.com/lllyasviel/ControlNet/discussions/categories/announcements](https://github.com/lllyasviel/ControlNet/discussions/categories/announcements) 来了解更加多细节

![](https://s2.loli.net/2025/07/09/Tfji2LMv15tgr6d.webp)

ControlNet的处理思路就很简单，再左图中模型的处理过程就是直接通过：$y=f(x;\theta)$来生成图像，但是在ControlNet里面会将我们最开始的网络结构复制然后通过在其前后引入一个**zero-convolution**层来“指导”（$Z$）模型的输出也就是说将上面的生成过程变为：$y=f(x;\theta)+Z(f(x+Z(c;\theta_{z_1});\theta);\theta_{Z_2})$。通过冻结最初的模型的权重保持不变，保留了Stable Diffusion模型原本的能力；与此同时，使用额外数据对“可训练”副本进行微调，学习我们想要添加的条件。因此在最后我们的SD模型中就是如下一个结构：

![](https://s2.loli.net/2025/07/09/uVNAEnleRMJ6p4v.webp)

在论文里面作者给出一个实际的测试效果可以很容易理解里面条件c（条件 𝑐就是提供给模型的显式结构引导信息，**用于在生成过程中精确控制图像的空间结构或布局**，一般来说可以是草图、分割图等）到底是一个什么东西，比如说就是直接给出一个“线稿”然后模型来输出图像。

![](https://s2.loli.net/2025/07/09/rkWH3o1MOaNs6pg.webp)

> **补充-1**：为什么使用上面这种结构
> 在[github](https://github.com/lllyasviel/ControlNet/discussions/188)上作者讨论了为什么要使用上面这种结构而非直接使用mlp等（作者给出了很多测试图像），最后总结就是：**这种结构好**
> **补充-2**：使用0卷积层会不会导致模型无法优化问题？
> 不会，因为对于神经网络结构大多都是：$y=wx+b$计算梯度过程中即使 $w=0$但是里面的 $x≠0$模型的参数还是可以被优化的

### T2I-Adapter[^3]
> https://github.com/TencentARC/T2I-Adapter

![image.png](https://s2.loli.net/2025/07/09/gZLDtFSGr25kCwa.webp)

T2I的处理思路也比较简单（T2I-Adap 4 ter Details里面其实就写的很明白了），对于输入的条件图片（比如说边缘图像）:512x512，首先通过 pixel unshuffle进行下采样将图像分辨率改为：64x64而后通过一层卷积+两层残差连接，输出得到特征 $F_c$之后将其与对应的encoder结构进行相加：$F_{enc}+ F_c$，当然T2I也支持多个条件（直接通过加权组合就行）
### 实际代码操作

## 参考
[^1]:https://arxiv.org/pdf/2307.01952
[^2]:https://arxiv.org/pdf/2302.05543
[^3]:https://arxiv.org/pdf/2302.08453