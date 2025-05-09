---
layout: mypost
title: 深入浅出了解生成模型-1：GAN/VAE/Diffusion Model 介绍
categories: 生成模型
extMath: true
images: true
address: changsha
show_footer_image: true
tags: [生成模型, GAN, VAE, Diffusion model]
description: 日常使用比较多的生成模型比如GPT/Qwen等这些大多都是“文生文”模型（当然GPT有自己的大一统模型可以“文生图”）但是网上流行很多AI生成图像，而这些生成图像模型大多都离不开下面三种模型：1、GAN；2、VAE；3、Diffusion Model。因此本文通过介绍这三个模型作为生成模型的入门。
---

日常使用比较多的生成模型比如GPT/Qwen等这些大多都是“文生文”模型（当然GPT有自己的大一统模型可以“文生图”）但是网上流行很多AI生成图像，而这些生成图像模型大多都离不开下面三种模型：1、GAN；2、VAE；3、Diffusion Model。因此本文主要介绍这三个基础模型作为生成模型的入门。
> **此处安利一下**何凯明老师在MiT的课程：
> https://mit-6s978.github.io/schedule.html

## Generative Adversarial Nets（GAN）
> From: https://arxiv.org/pdf/1406.2661

- [ ] 1、GAN基本原理
- [ ] 2、GAN数学原理
- [ ] 3、GAN代码，去替换我的 采样分布测试效果

在GAN里面一个比较核心的概念就是：通过生成模型G去捕获数据分布，而后通过一个判别模型D，判断样品来自训练数据而不是G。
> A generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G
通过下面图像来了解：
![](https://s2.loli.net/2025/05/08/2hjKs9GRE5uYqwZ.png)
其中：**判别模型会尝试在数据空间中划定边界，而生成式模型会尝试对数据在整个空间中的放置方式进行建模**

换言之就是：有两组模型1、生成模型G；2、判别模型D。其中生成模型用来生成我们需要的图像而我们的判别模型则是用来判断所生产的图像是不是“合理”的（就像老师和学生关系，老师只去关注学生的作品怎么样，而学生只去关注如何生成老师满足的作品）。了解基本原理之后，接下来深入了解其理论知识：假设数据$x$ 存在一个分布 $p_g$ 那么可以通过随机生成一个噪音变量 $p_z(z)$ 而后通过一个模型（生成模型） $G(z;\theta _g)$ 来将我们的噪音变量映射到我们正式的数据分布上，而后通过另外一个模型（判别模型） $D(x;\theta _d)$ 来判断数据是来自生成模型还是原始数据分布，因此就可以定义一个下面损失函数：

![](https://s2.loli.net/2025/05/08/uiIUjJcXg23QVrh.png)

其中两种不同颜色在最值处理上是因为：1、$D(x)$：判别器给真实样本的概率输出（判断真实的样本标记1，对于生成的样本标记0）；那么对于这部分计算值：$log(D(x))$ 自然而然的希望他是越大越好（*希望判别器经可能的判别真实样本*）；2、$D(G(z))$：判别器对于生成样本的概率输出，对于这部分值（$D(G(z))$的计算值）我们希望越接近0越好（*越接近0也就意味着判别模型能够区分生成样本*），但是对于生成器模型而言希望的是：通过随机生成的样本：z越贴近我们真实分布越好。
> **两个模型就像是零和博弈，一个尽可能的生成假的东西，一个尽可能判别出假东西**

整个训练过程如下所示：
![](https://s2.loli.net/2025/05/08/reFmGb756tzkhdg.png)

从左到右边：最开始生成模型所生成的效果不佳，判别模型可以很容易就判断出哪些是正式数据哪些是生成数据，但是随着模型迭代，生成模型所生成的内容越来越贴近正式的数据分布进而导致判别模型越来越难以判断。

算法流程：
![](https://s2.loli.net/2025/05/08/TpqWhU3SEVlAdeB.png)

GAN训练过程分为两部分：第一部分学习优化判别器；第二部分学习优化生成器。
模型架构：
![](https://s2.loli.net/2025/05/08/1fLQYVGavAyTBRx.png)

### 进一步了解GAN数学原理
这部分内容主要参考**李宏毅老师Youtube教程**：
<div class="video-center">
  <iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/DMA4MrNieWo?si=kk0HuutqIOT-CLp4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>


https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN/blob/master/pytorch_MNIST_GAN.py

> 值得注意的是：
> 1、为什么不去用生成模型自己判断呢？
> 2、为什么不去用判别模型自己生成呢？
## 参考
1、https://arxiv.org/pdf/1406.2661
2、https://developers.google.cn/machine-learning/gan/gan_structure