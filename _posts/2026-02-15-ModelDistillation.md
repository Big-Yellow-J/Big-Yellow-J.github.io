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
description: 大模型推理优化除量化、多attention方案、cache策略外，可通过模型蒸馏、模型剪枝两类方案实现参数优化。模型蒸馏核心是让小体量学生模型学习预训练大模型的行为逻辑，主流方案包括叠加双损失的KD知识蒸馏、带温度参数的DKD解耦知识蒸馏、基于双损失约束的DMD分配匹配蒸馏，可在降低计算量的前提下尽可能保留大模型性能。模型剪枝分为置零单权重的非结构化剪枝、移除整体结构单元的结构化剪枝两类，需通过多轮剪枝+微调迭代实现，可借助PyTorch内置prune工具快速落地。
---

在[最开始的文章](https://www.big-yellow-j.top/posts/2025/12/29/SDAcceralate.html)中介绍了模型量化、使用不同attention方式、cache策略去对模型生成进行加速，这里主要介绍几种模型蒸馏以及模型剪枝的方式去优化模型参数
## 模型蒸馏
模型蒸馏是一种模型压缩和优化技术。简单来说，就是让一个**小模型（学生模型）去模仿一个已经训练好的大模型（教师模型）的行为，从而用更小的体积、更低的计算量，尽量接近大模型的性能**。比如说Qwen0.5B模型就是直接从一个较大的模型进行蒸馏得到。而对于蒸馏过程也有很多，最常见的就是**直接通过数据进行蒸馏**，比如说通过CahtGPT生成高质量数据而后将这部分高质量数据进行模型训练也可以达到蒸馏的目的。亦或者直接通过模型进行蒸馏，**最简单的蒸馏例子**（计算students的预测loss以及students和teacher模型之间的KD）：**1、KD知识蒸馏过程**：直接定义一个参数较小的学生模型而后对于相同的数据分别通过小模型以及大模型处理，再去计算“叠加loss”：$L=L_{student}+ L_{KD}$ 其中第一项是小模型的loss第二项是计算小-大模型之间的KD；**2、DKD解耦知识蒸馏**：$L=\alpha L_{TCKD}+ \beta L_{NCFD}$ 其中 
$$\mathcal{L}_{TCKD} = \mathrm{KL} ( [p^S_t,\ 1-p^S_t]\ \Vert\ [p^T_t,\ 1-p^T_t] )$$ 
以及 
$$\mathcal{L}_{NCKD} = \mathrm{KL} ( \frac{p^S}{\sum_{j\neq t} p^S_j},\ \frac{p^T}{\sum_{j\neq t} p^T_j} )$
在实际应用过程中有一个比较重要参数：temperature使用方式和llm中的相同都是输出概率去除温度系数，比如说softmax中：$\frac{e^{x/t}}{\sum e^{x/t}}$$
> 对于两种简单的知识蒸馏代码：[代码](https://github.com/shangxiaaabb/ProjectCode/tree/main/code/Python/DFModelCode/DF_acceralate)

### 分配匹配蒸馏（DMD）
对于DMD[^2]方法原理如下（DMD1的算法流程）：

<!-- ![20260331164300749](https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260331164300749.png) -->

![DMD1](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260219145134921.png)

对于上诉算法流程图简单描述在DMD蒸馏中主要是通过两个Loss实现，**1、regression loss（回归损失）**：对于教师模型生成过程中得到一批noise-image对（对应 $z_{ref}, y_{ref}$），对于蒸馏的学生模型生成器 $G_\theta$ 直接用初始化噪声以及noise-image中的噪声进行单步生成得到image分别得到：$x$ 以及 $x_{ref}$，而后对于noise-image中的噪声直接去计算**LPIPS损失函数**（主要是计算两组图像之间的相似度，[代码实现](https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html)）；**2、diffusion loss**：蒸馏过程中核心损失，在一般的蒸馏模型过程如LCM是去强迫学生模型模仿 teacher 的每一步去噪轨迹，而DMD则是去计算最终生成的图像分布是否和真实分布之间是否一致，具体的处理过程直接去计算distributionMatchingLoss：
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260331210034629.png)
从上面过程很容易知道处理过程，这里简单介绍之所以这么做的原因，处理思路和GAN相似，通过计算“真假两个扩散模型”在噪声预测上的偏差，产生一个指引梯度，推着学生模型去生成更符合真实统计规律的图像。

<!-- **在DMDv2**[^1]中直接取消了回归损失（主要是在DMDv1中需要预先通过教师模型生成一个庞大的“文本/噪声-图像”对数据集） -->

> [LCM蒸馏过程](https://www.big-yellow-j.top/posts/2025/06/17/CM.html)：训练过程可以简单理解为：对于输入图像 $x$，直接添加 $n$ 步的噪声得到 $x_n$，而后我的学生模型直接去预测 $t_0$ 时候的结果 $y_1$；同时，我的教师模型（预训练好的扩散模型）从 $x_n$ 出发，通过 DDIM 采样器向前走一步（跨越 $k$ 个时间步），得到 $t_{n-k}$ 时刻在轨迹上的观察点 $x_{n-k}$；而后再去用学生模型通过 $x_{n-k}$ 预测 $t_0$ 的结果得到 $y_2$。最后计算 $y_1$ 与 $y_2$ 之间的距离损失（Consistency Loss），迫使模型无论从哪个时间步出发，预测的终点都指向同一点。

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