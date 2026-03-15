---
layout: mypost
title: 深入浅出了解生成模型-10：模型蒸馏与剪枝
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
在[最开始的文章](https://www.big-yellow-j.top/posts/2025/12/29/SDAcceralate.html)中介绍了模型量化、使用不同attention方式、cache策略去对模型生成进行加速，这里主要介绍几种模型蒸馏以及模型剪枝的方式去优化模型参数
## 模型蒸馏
模型蒸馏是一种模型压缩和优化技术。简单来说，就是让一个小模型（学生模型）去模仿一个已经训练好的大模型（教师模型）的行为，从而用更小的体积、更低的计算量，尽量接近大模型的性能。比如说Qwen0.5B模型就是直接从一个较大的模型进行蒸馏得到。而对于蒸馏过程也有很多，最常见的就是直接通过数据进行蒸馏，比如说通过CahtGPT生成高质量数据而后将这部分高质量数据进行模型训练也可以达到蒸馏的目的。
**最简单的蒸馏例子**（计算students的预测loss以及students和teacher模型之间的KD）：**1、KD知识蒸馏过程**：直接定义一个参数较小的学生模型而后对于相同的数据分别通过小模型以及大模型处理，再去计算“叠加loss”：$L=L_{student}+ L_{KD}$ 其中第一项是小模型的loss第二项是计算小-大模型之间的KD；**2、DKD解耦知识蒸馏**：$L=\alpha L_{TCKD}+ \beta L_{NCFD}$ 其中 $\mathcal{L}_{TCKD} = \mathrm{KL} \left( [p^S_t,\ 1-p^S_t]\ \Vert\ [p^T_t,\ 1-p^T_t] \right)$ 以及 $\mathcal{L}_{NCKD} = \mathrm{KL} \left( \frac{p^S}{\sum_{j\neq t} p^S_j},\ \frac{p^T}{\sum_{j\neq t} p^T_j} \right)$
在实际应用过程中有一个比较重要参数：temperature使用方式和llm中的相同都是输出概率去除温度系数，比如说softmax中：$\frac{e^{x/t}}{\sum e^{x/t}}$
> 对于两种简单的知识蒸馏代码：[代码](https://github.com/shangxiaaabb/ProjectCode/tree/main/code/Python/DFModelCode/DF_acceralate)

### 分配匹配蒸馏（DMD）
对于DMD[^1][^2]方法原理如下（DMD1的算法流程）：
![](https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260219145134921.png)
对于上诉算法流程图简单描述
## 模型剪枝
在一个训练好的大模型中，数以亿计的参数（权重）里，也存在大量冗余或贡献微弱的连接。模型剪枝的核心思想，就是识别并移除这些“不重要”的权重，从而得到一个更小、更高效的模型。在剪枝方法上主要两大类，1、**非结构化剪枝**，将单个权重值置为零。这会产生一个“稀疏”的权重矩阵，即矩阵中包含大量零值。2、**结构化剪枝**，移除整个结构单元，例如整个神经元（矩阵的行/列）、注意力头，甚至是整个网络层。模型剪枝过程必须**剪枝->微调->剪枝->微调**不断地重复这个过程。**最简单的剪枝过程**可以直接使用`torch.nn.utils.prune`来进行操作，在代码中torch提供多种剪枝条方法（一般后缀中有 `_unstructured`表示的是非结构化的剪枝，而`_structured`则是表示结构化的剪枝），**非结构化剪枝代码**
```python
import torch.nn.utils.prune as prune
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=10,
)
```
不过需要注意的是按照[官方介绍](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.prune.global_unstructured.html)：对于非结构化剪枝的区域回去添加一个名为name+'_mask'的命名缓冲区，对应于修剪方法对参数名称应用的二进制掩码（mask是一个0/1张量量化过程就是：`weight = weight_orig * mask`）。将参数名称替换为修剪后的版本，而原始（未修剪）参数存储在名为name+'_orig'的新参数中。也就意味着在 `prune.global_unstructured` **不会改变模型大小只是添加mask让模型推理可以加速**。**结构化剪枝代码**：
```python
import torch.nn.utils.prune as prune
prune.ln_structured(module,...)
```
具体测试脚本：[ModelPrune.py](https://github.com/shangxiaaabb/ProjectCode/tree/main/code/Python/RL-TRL/ModelPrune.py)
## 参考
[^1]: [Improved Distribution Matching Distillation for Fast Image Synthesis](https://tianweiy.github.io/dmd2/)
[^2]: [One-step Diffusion with Distribution Matching Distillation](https://tianweiy.github.io/dmd/)