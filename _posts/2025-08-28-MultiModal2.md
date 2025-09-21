---
layout: mypost
title: 多模态算法QwenVL、KimiVL等算法原理
categories: 深度学习基础理论
extMath: true
images: true
address: 武汉🏯
tags:
- cv-backbone
- 多模态
- multimodal
show_footer_image: true
description: 多模态大语言模型通用框架通过视觉编码器（如ViT）、文本编码器及映射层对齐维度输入LLM。QwenVL系列（QwenVL、QwenVL2、QwenVL2.5）为典型实现，QwenVL采用ViT-bigG视觉编码器，经可学习query的Cross-Attention压缩视觉token至256长度，融合二维绝对位置编码；QwenVL2改进为动态分辨率（无需固定尺寸，2x2
  token拼接+MLP）及多模态旋转位置编码（M-RoPE，含时序、高度、宽度信息），提升处理性能。
---

对于多模态系列模型大致的多模态大语言模型的通用模型框架和每个模块的一些实现方法[^1]：
![](https://s2.loli.net/2025/09/21/JF9YdeEAhuMyzkZ.webp)
基本上就是对于图片/视频等通过不同的视觉编码器（Vit/Clip等）进行编码，对于text通过编码器进行编码，而后将视觉模态信息通过映射层（q-former/mlp等）将两部分维度对齐而后丢到LLM中输出结果。简单总结常用的多模态模型。
## QwenVL系列
目前QwenVL迭代更新道理2.5（**截至2025.09.21**）主要了解QwenVL、QwenVL2、QwenVL2.5
### QwenVL
在QwenVL[^4]中在论文里面作者提到的其模型的整个训练过程如下：
![](https://s2.loli.net/2025/09/21/HEhlRPFJBMKpjoZ.webp)
> 仅从提供的不同阶段还是很容易发现QwenVL还是是采用和BLIP相似的使用 learned-query来对齐模态信息
> 语言模型使用（7.7B）：Qwen-7B
> 视觉编码器（1.9B）：Vit-bigG
> 融合器（0.08B）：Learnable Query

不过论文里面对于模型细节介绍不是很多，从代码角度出发窥其模型结构：
**模型视觉编码器**：视觉编码器使用的是ViT架构（Vision Transformer），ViT的网络设置和初始化参数使用了OpenCLIP预训练好的**ViT-bigG模型**。具体的代码处理过程（[代码](https://huggingface.co/Qwen/Qwen-VL/blob/main/visual.py)），其中模型输出维度变化过程：1x3x448x448-->1x1664x32x32（首先卷积处理）-->1x1024x1664（拉平交换维度）
**特征融合器**：上述ViT处理后，对于$448\times 448$分辨率的图像，生成一个 **[1024, 1664]**的序列，也就是向量维度为1664的长度为1024的序列。为了压缩视觉token的输入长度，Qwen-VL引入了一个Adapter来压缩图像特征。这个Adaper就是一个随机初始化的单层Cross-Attention模块。该模块使用一组可学习的query向量，将来自ViT的图像特征作为Key向量。通过Cross-Attention操作后将视觉特征序列压缩到固定的256长度（也就是将视觉特征压缩到 **256 1644**）
此外，考虑到位置信息对于精细图像理解的重要性，Qwen-VL将二维绝对位置编码（三角位置编码）整合到Cross-Attention的 $q,k$中，以减少压缩过程中可能丢失的位置细节。随后将长度为256的压缩图像特征序列输入到大型语言模型中。
### QwenVL-2
对于QwenVL-2[^3]其模型的基本结构如下：
![](https://s2.loli.net/2025/09/21/5c1jovnLVOaS62H.webp)
QwenVL2区别上一代QwenVL模型所作出改进如下：**1、使用动态分辨率**（也就是说输入图像不需要再去改变图像尺寸到一个固定值），于此同时为了减少 **visual-token**数量，将**2x2的的相邻的token进行拼接**到一个token而后通过MLP层进行处理。
![](https://s2.loli.net/2025/09/21/w3agENHmLVcoSdt.webp)
动态分辨率处理如上，通过指定`[mix_pixels, max_pixels]`范围然后将图像保持原始的纵横比去缩减图像到上面的范围中（[处理过程](https://github.com/QwenLM/Qwen2.5-VL/blob/c15045f8829fee29d4b3996e068775fe6a5855db/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L59)，首先计算原始图像的像素数量，而后判断和上面指标的范围，如果超出范围就去计算需要修改的比例，在将整个比例去处理到分辨率上）
在通过使用动态分辨率处理图像之后会在单一**图片增加时间维度**也就是将：CHW-->TCHW，在源码中T选择数值为2也就是将图片“复制一次”
2、**多模态的旋转位置编码（M-RoPE）**,也就是将原来位置编码所携带的信息处理为：时序（temporal）、高度（height）、宽度（width）。比如下图中对于文本处理直接初始化为：$(i,i,i)$。但是对于图片而言就是：$(i,x,y)$ 其中 $i$ 是恒定的，而对于视频就会将 $i$ 换成视频中图像的顺序
### QwenVL-2.5
在QwenVL2.5中[^6]模型具体的代码处理过程参考Blog[^5]具体模型结构：
![](https://s2.loli.net/2025/09/21/R8yLfVqpznvkgZw.webp)
## KimiVL系列
## 参考
[^1]: [https://arxiv.org/abs/2504.07491](https://arxiv.org/abs/2504.07491)
[^2]: [https://zhuanlan.zhihu.com/p/25267823390](https://zhuanlan.zhihu.com/p/25267823390)
[^3]: [http://arxiv.org/abs/2409.12191](http://arxiv.org/abs/2409.12191)
[^4]: [https://arxiv.org/pdf/2308.12966](https://arxiv.org/pdf/2308.12966)
[^5]: [https://www.big-yellow-j.top/posts/2025/08/29/QwenVLCode.html](https://www.big-yellow-j.top/posts/2025/08/29/QwenVLCode.html)
[^6]: [https://arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923)