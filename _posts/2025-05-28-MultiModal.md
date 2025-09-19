---
layout: mypost
title: 多模态算法Clip、Albef、Blip等算法原理
categories: 深度学习基础理论
extMath: true
images: true
address: 武汉🏯
tags:
- cv-backbone
- 多模态
- multimodal
show_footer_image: true
description: 视觉多模态模型通常通过视觉编码器（如ViT/ResNet）处理图像信息，结合文本输入LLM，核心挑战在于不同模态信息的对齐。本文重点介绍模态对齐方法，包括对比学习范式（如CLIP）通过对比学习和嵌入空间对齐，将图像与文本映射到共享语义空间，学习跨模态相似度表示，展现强大通用性和零样本学习能力；以及BLIP、BLIP-2等模型，通过Learned-Queries弥补模态差异，结合ITC（Image-Text
  Contrastive Learning）优化图文特征相似度、ITG（Image-grounded Text Generation）训练生成文本所需视觉特征提取、ITM（Image-Text
  Matching）实现细粒度对齐与匹配分类，有效解决多模态信息结合问题，为图文检索、零样本分类等任务提供基础。
---
视觉多模态模型在结构上比较统一，一个视觉编码器（较多使用的是Vit/Resnet等）对图像信息进行处理，然后将其和文本信息一起结合然后输入到LLM模型中得到最后的结果，因此在此过程中一个最大的挑战就是：**如果将不同模态信息进行结合**（当然有些可能还需要考虑如何将图像进行压缩，这里主要是考虑有些图像的分辨率比较高）。

## Clip

![](https://s2.loli.net/2025/06/22/H6kEoxgzYAWNhXp.webp)

代表模型 [**CLIP**](https://arxiv.org/pdf/2103.00020)[^2]，更加像一种 **图像-文本**对齐模型，按照论文里面他自己提到的计算范式：

```python
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter
# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]
# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)
# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```

在将 Image 和 Text 编码完成之后，直接计算它们之间的**相似度**，实现模态之间的对齐。优化过程的目标是让匹配的图文对的相似度尽可能大，同时让不匹配对的相似度尽可能小。换言之，CLIP 的对比学习机制本质上是在学习一种跨模态的相似度表示。其核心机制是通过对比学习和嵌入空间对齐，将图像和文本映射到一个共享的语义空间中。
尽管 CLIP 本身并不直接包含复杂的推理能力或任务特定的知识，但它通过大规模预训练，展现出了强大的通用性和零样本学习能力。在论文中，CLIP 表现出了不俗的零样本性能，但需要注意的是，CLIP 的主要目标是学习跨模态的对齐表示，这使得它能够胜任多种任务（如图文检索、零样本分类等）。相比于传统的目标识别模型，CLIP 更像是一个多模态的基础模型，具备更广泛的适用性和灵活性。
## ALBEF
Albef[^1]模型基本结构如下：
![](https://s2.loli.net/2025/09/19/wCK5MxvBQITkuhE.png)
**模型结构**：1、图像编码器（12层的Vit-B/16）；2、文本编码器（6层 $\text{BERT}_{\text{base}}$）；3、多模态编码器（6层 $\text{BERT}_{\text{base}}$）。对于文本和图像都会编码为带前缀的向量，图像：${v_{cls},v_1,...,v_N}$，文本：${w_{cls},w_1,...,w_N}$。
**训练过程**：**1、模态对齐（ITC）**：这个过程主要是计算image-to-text以及text-to-image相似性计算过程如下：
![](https://s2.loli.net/2025/09/19/SqxarzjPtbegiQZ.png)
其中 $s(I, T_m)=g_v(v_{cls})^Tg^′(w^′_{cls})$，相似性计算公式中 $g$主要是将 `[CLS]`通过线性处理处理到256维，而 $g^′$则是通过动量编码器的规范化特征表示。$y$代表GT。
> 对于这个loss计算过程再Albef中会改写为：
> ![](https://s2.loli.net/2025/09/19/X9I8ZEzxOyeg3GS.png)
> 其中$s$代表score function（比如说直接计算点乘）

**2、遮蔽语言模型( MLM )**：直接预测被MASK掉的词；**3、图文匹配（ITM）**：主要是判断图文之间匹配，对于这两个过程数据处理为：
![](https://s2.loli.net/2025/09/19/eVdW7hRcSwn3Ial.png)
## BLIP
### BLIPv1
BLIP-1[^3]模型结构如下
![](https://s2.loli.net/2025/09/19/vOkf7aWluqItKEh.png)
对于模型使用对于**视觉编码器直接使用Vit**，对于**文本编码器直接使用BERT**，不过值得注意的是和Albef中处理相同的是在特征前面都会选择添加一个`[CLS]`标记然后其他结构集合上面的一致。在模型结构上主要分为3块：1、Text Encder；2、Image grounded Text encoder；3、Image-grouned Text decoder；对于这3块都分别对应的去计算ITC、ITM以及LM3个损失，其中前两个和Albef中计算方式相同。除此之外虽然设计了3个模块但是模块之间参数是共享的（**颜色相同那么参数就是相同的**）
![](https://s2.loli.net/2025/09/19/kvuBxLI18JtEjAC.png)
论文中数据合成方法，其实还是基于BLIP自身的encoder-decoder结构，首先是通过标注的数据（$I_h,T_h$）进行训练模型在得到很好的效果之后，将未标注的图片 $I_w$直接输入到模型中生成图-文对（$I_w,T_s$）以及从网络上搜索得到的图-文对（$I_w,T_w$）此时这两部分图文对不是很“恰当的”通过filter去过滤掉不合适的配对这样一来最后就可以得到相对干净的图-文对。
> 其中`filter`设计就是直接使用 image-ground text encoder通过直接微调来让模型知道 图-文匹配效果
### BLIPv2
**BLIP-2**[^4]模型结构如下：
![](https://s2.loli.net/2025/06/22/LejUt6HI5XRYZcr.webp)
在 BLIP-2中**同时冻结了Image Encoder以及LLM**因此为了弥补不同模态之间的差异，就需要设计一个“模块”来进行表示（在论文中做法是：**将Image/Text上的信息都”反映“到一个Learned-Queries上**）。具体操作为：
1、**图文对比损失ITC**：优化目标是对齐图像特征和文本特征，也就是对齐image transformer输出的query representation与来自text transformer输出的text representation。为了避免信息泄漏，ITC采用了单模态自注意掩码，不允许query和text看到对方。计算时先计算每个query与文本embedding之间的相似度，然后选择最高的作为图文相似度
2、**ITG**(Image-grounded Text Generation)：优化目标是给定输入图像作为条件，训练 Q-Former 生成文本，迫使query提取包含所有文本信息的视觉特征。由于 Q-Former 的架构不允许冻结的图像编码器和文本标记之间的直接交互，因此生成文本所需的信息必须首先由query提取，然后通过自注意力层传给text token。ITG采用多模态causal attention mask来控制query和text的交互，query可以相互感知，但不能看见text token，每个text token都可以感知所有query及其前面的text标记【半矩阵，生成式任务的常见做法】。这里将 [CLS] 标记替换为新的 [DEC] 标记，作为第一个文本标记来指示解码任务。
3、**ITM**( Image-Text Matching)：优化目标是进行图像和文本表示之间的细粒度对齐，学一个二分类任务，即图像-文本对是正匹配还是负匹配。这里将image transformer输出的每个query嵌入输入到一个二类线性分类器中以获得对应的logit，然后将所有的logit平均，再计算匹配分数。ITM使用双向自注意掩码，所有query和text都可以相互感知。
## 总结

## 参考
[^1]: [https://arxiv.org/pdf/2107.07651](https://arxiv.org/pdf/2107.07651)
[^2]: [https://arxiv.org/pdf/2103.00020](https://arxiv.org/pdf/2103.00020)
[^3]: [https://arxiv.org/pdf/2201.12086](https://arxiv.org/pdf/2201.12086)
[^4]: [https://arxiv.org/pdf/2301.12597](https://arxiv.org/pdf/2301.12597)