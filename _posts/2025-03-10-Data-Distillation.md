---
layout: mypost
title: 数据蒸馏（Data Distillation）操作原理
categories: 深度学习基础理论
extMath: true
images: true
address: changsha
show_footer_image: true
description: 主要介绍数据蒸馏（Data Distillation）操作原理
---

主要介绍数据蒸馏操作，并且介绍CVPR-2025上海交大满分论文：Dataset Distillation with Neural Characteristic Function: A Minmax Perspective。本文主要是借鉴论文1中的整体结构，大致了解什么是DD而后再去介绍（CVPR-2025）论文。

## Data Distillation

**数据蒸馏**（Data Distillatiob）是一种从大量数据中提取关键信息，生成高质量、小规模合成数据集的技术。它的目标是通过这些合成数据来替代原始数据集，用于模型训练、验证或其他任务，从而提高效率、降低成本或保护隐私。数据蒸馏的核心思想是“从数据中提取数据”，让合成数据集中保留原始数据集的关键特征和分布信息，同时去除冗余和噪声。参考论文1中的描述：

![](https://s2.loli.net/2025/03/10/w3xVtlISa9mnvAW.png)

数据蒸馏（DD）目标为：对于一个真实的数据集：$\mathrm{T}=(X_t,Y_t)$ 其中 $X_t\in R^{N\times d}$ 其中 $N$ 代表样本数量 $d$ 代表特征数量，$Y_t\in R^{N\times C}$ 其中$C$为输出实体。对于蒸馏得到的数据集：$\mathrm{S}={X_s,Y_s}$其中$X_s\in R^{M\times D}$其中$M$代表数据蒸馏后的样本数量。最终的优化目标为： $\text{arg min} \mathrm{L}(\mathrm{S}, \mathrm{T})$

比如说对于图像分类任务而言$D$代表的是：HWC而y代表的是独热编码，C代表类别数量

![](https://s2.loli.net/2025/03/10/RuIAWElQZ3FqSDL.png)

论文1中对于损失函数优化主要分析3种处理思路

## 1、Performance Matching

$$
\begin{aligned}
\mathcal{L}(\mathcal{S},\mathcal{T}) & =\mathbb{E}_{\theta^{(0)}\sim\Theta}[l(\mathcal{T};\theta^{(T)})], \\
\theta^{(t)} & =\theta^{(t-1)}-\eta\nabla l(\mathcal{S};\theta^{(t-1)})
\end{aligned}
$$

其中$\theta, l, T, \eta$分别代表：神经网络参数、损失函数、迭代次数、学习率

![](https://s2.loli.net/2025/03/10/cXI6hvqWruSd3E9.png)

对于上面公式以及优化过程理解：似乎整体优化过程没有体现源数据：$\mathrm{T}$ 和蒸馏数据： $\mathrm{S}$ 两者之间是如何进行优化的，第二个过程直接优化参数和源数据之间差异，可以理解为借助源数据优化得到一个较好的参数$\theta$。第一个过程则是借助第$T$步得到的参数去计算蒸馏数据集之间差异（这个过程可以理解为模型参数是固定的，但是数据是变化的，需要的是一个数据集在通过源数据集上也有较好的表现）

## 2、Parameter Matching

**分别使用合成数据集和原始数据集对同一个网络进行若干步训练，并促使它们训练得到的神经网络参数保持一致**。根据使用合成数据集（S）和原始数据集（T）进行训练的步数，参数匹配方法可以进一步分为两类：单步参数匹配和多步参数匹配。

![Parameter Matching](https://s2.loli.net/2025/03/10/gjtrKVp1Ccy86Ue.png)

左图为单参数匹配，右图为多参数匹配

* 1、单参数匹配

$$
\begin{aligned}
\mathcal{L}(S, T) &= \mathbb{E}_{\theta^{(0)} \sim \Theta} \left[ \sum_{t=0}^{T} \mathcal{D}(S, T; \theta^{(t)}) \right] \\
\theta^{(t)} &= \theta^{(t-1)} - \eta \nabla l(S; \theta^{(t-1)})
\end{aligned}
$$

其中$\mathrm{D}$代表两部分梯度之间的距离

$$
\begin{aligned}
\mathcal{D}(S, T; \theta) &= \sum_{c=0}^{C-1} d(\nabla l(S_c; \theta), \nabla l(T_c; \theta)), \\
d(A, B) &= \sum_{i=1}^{L} \sum_{j=1}^{J_i} \left(1 - \frac{\mathbf{A}_j^{(i)} \cdot \mathbf{B}_j^{(i)}}{\|\mathbf{A}_j^{(i)}\| \|\mathbf{B}_j^{(i)}\|}\right),
\end{aligned}
$$

* 2、多参数匹配

![https://georgecazenavette.github.io/mtt-distillation/](https://georgecazenavette.github.io/mtt-distillation/resources/method.gif)

对于单步参数匹配，由于只匹配单步梯度，因此在评估中可能会积累误差，而模型是通过多步合成数据更新的

$$
\begin{aligned}
\mathcal{L}(S, T) &= \mathbb{E}_{\theta^{(0)} \sim \Theta} \left[ \mathcal{D}(\theta_S^{(T_s)}, \theta_T^{(T_t)}) \right] \\
\theta_S^{(t)} &= \theta_S^{(t-1)} - \eta \nabla l(S; \theta_S^{(t-1)}) \\
\theta_T^{(t)} &= \theta_T^{(t-1)} - \eta \nabla l(T; \theta_T^{(t-1)}) \\
\mathcal{D}(\theta_S^{(T_s)}, \theta_T^{(T_t)}) &= \frac{\|\theta_S^{(T_s)} - \theta_T^{(T_t)}\|^2}{\|\theta_T^{(T_t)} - \theta^{(0)}\|^2}
\end{aligned}
$$

多步参数则是直接对数据S和T参数进行多步更新，优化目标为两部分数据所得到的参数$\theta_S$ 以及 $\theta_ T$

## 参考
1、[Dataset Distillation: A Comprehensive Review](https://arxiv.org/pdf/2301.07014)
2、A Comprehensive Survey of Dataset Distillation
3、（CVPR-2025）Dataset Distillation with Neural Characteristic Function: A Minmax Perspective
4、（CVPR-2024）On the Diversity and Realism of Distilled Dataset: An Efficient Perspective
5、（CVPR-2023）Accelerating Dataset Distillation via Model Augmentation