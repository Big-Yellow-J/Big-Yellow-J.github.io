---
layout: mypost
title: 深入浅出了解生成模型-4：一致性模型（consistency model）
categories: 生成模型
extMath: true
images: true
address: 武汉🏯
show_footer_image: true
tags: [生成模型,diffusion model,一致性模型]
description: 前面已经介绍了扩散模型，扩散模型往往需要多步才能生成较为满意的图像，但是可一致性模型可以通过几步生成图像，因此这里主要是介绍一致性模型（consistency model）基本原理以及代码实践。
---

前面已经介绍了[扩散模型](https://www.big-yellow-j.top/posts/2025/05/19/DiffusionModel.html)，在最后的结论里面提到一点：扩散模型往往需要多步才能生成较为满意的图像。不过现在有一种新的方式来加速（旨在通过少数迭代步骤）生成图像：**一致性模型（consistency model）**，因此这里主要是介绍一致性模型（consistency model）基本原理以及代码实践。
## 一致性模型（Consistency Model）
![](https://s2.loli.net/2025/06/16/aC17TAmX4JRkLMS.png)
> 其中`ODE`（常微分方程），在传统的扩散模型（Diffusion Models, DM）中，前向过程是从原始图像 $x_0$开始，不断添加噪声，经过 $T$步得到高斯噪声图像 $x_T$。反向过程（如 DDPM）通常通过训练一个逐步去噪的模型，将 $x_T$逐步还原为 $x_0$ ，每一步估计一个中间状态，因此推理成本高（需迭代 T 步）。而在 **Consistency Models（CM）** 中，模型训练时引入了 **Consistency Regularization**，使得模型在不同的时间步 $t$都能一致地预测干净图像。这样在推理时，无需迭代多步，而是可以通过一个单一函数$f(x ,t)$ 直接将任意噪声图像$x_t$ 还原为目标图像$x_0$ 。这大大减少了推理时间，实现了一步（或少数几步）生成。

一致性模型（consistency model）在论文[^1]里面主要是通过使用常微分方程角度出发进行解释的，尽可能的避免数学公式理解（主要是自己也搞不明白🤪🤪🤪🤪）Consistency Model 在 Diffusion Model 的基础上，新增了一个约束：**从某个样本到某个噪声的加噪轨迹上的每一个点，都可以经过一个函数 $f$ 映射为这条轨迹的起点**，用数学描述就是：$f:(x_t, t)\rightarrow x_\epsilon$，换言之就是需要满足： $f(x_t,t)=f(x_{t^\prime},t^\prime)$ 其中 $t,t^\prime \in [\epsilon,T]$，比如论文里面的图片描述：
![](https://s2.loli.net/2025/06/16/ID9yRkQvKj2CO5m.png)

![](https://s2.loli.net/2025/06/16/Qr2AUsq48bD1mYx.png)

## Latent Consistency Model
潜在一致性模型（Latent Consistency Model）[^2]

## TCD[^3]

## 参考
[^1]:https://arxiv.org/abs/2303.01469
[^2]:https://arxiv.org/abs/2310.04378
[^3]:https://arxiv.org/abs/2402.19159