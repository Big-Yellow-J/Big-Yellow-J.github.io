---
layout: mypost
title: 深度学习基础理论————各类Attention(Flash Attention/MLA/Page Attention)
categories: 深度学习基础理论
extMath: true
images: true
address: changsha
show_footer_image: true
description: 主要介绍各类Attention(Flash Attention/MLA/Page Attention)
---

## 深度学习基础理论————各类Attention操作

## 1、`Attention`

https://spaces.ac.cn/archives/8620

## 2、`Flash Attention`

[论文](https://arxiv.org/pdf/2205.14135)提出，是一种高效的注意力计算方法，旨在解决 Transformer 模型在处理长序列时的计算效率和内存消耗问题。**其核心思想是通过在 GPU 显存中分块执行注意力计算，减少显存读写操作，提升计算效率并降低显存占用**。

![1](https://s2.loli.net/2025/01/31/Gqe94YpAXKftVJg.png)

`Flash Attention`计算机制：
**分块计算**：传统注意力计算会将整个注意力矩阵 (N×N) 存入 GPU 内存（HBM），这对长序列来说非常消耗内存，FlashAttention 将输入分块，每次只加载一小块数据到更快的 SRAM 中进行计算，传统`Attention`计算和`flash attention`计算：
![1](https://s2.loli.net/2025/01/31/IbjDs6EKdO9VUJ2.png)

对比上：传统的计算和存储都是发生再`HBM`上，而对于`flash attention`则是**首先**会将`Q,K,V`进行划分（算法1-4：整体流程上首先根据`SRAM`的大小`M`去计算划分比例（$\lceil \frac{N}{B_r} \rceil$）然后根据划分比例去对`QKV`进行划分这样一来Q（$N\times d$就会被划分为不同的小块，然后只需要去遍历这些小块然后计算注意力即可）），**然后计算**`Attention`（算法5-15），计算中也容易发现：先将分块存储再`HBM`上的值读取到`SRAM`上再它上面进行计算，不过值得注意的是：在传统的$QK^T$计算之后通过`softmax`进行处理，但是如果将上述值拆分了，再去用普通的`softmax`就不合适，因此使用`safe softmax`

---
1、**HBM**（High Bandwidth Memory，高带宽内存）:是一种专为高性能计算和图形处理设计的内存类型，旨在提供高带宽和较低的功耗。HBM 常用于需要大量数据访问的任务，如图形处理、大规模矩阵运算和 AI 模型训练。
2、 **SRAM**（Static Random Access Memory，静态随机存取存储器）:是一种速度极快的存储器，用于存储小块数据。在 GPU 中，SRAM 主要作为缓存（如寄存器文件、共享内存和缓存），用于快速访问频繁使用的数据。例如在图中 FlashAttention 的计算中，将关键的计算块（如小规模矩阵）存放在 SRAM 中，减少频繁的数据传输，提升计算速度。
3、不同`softmax`计算：
`softmax`:$x_i=\frac{e^{x_i}}{\sum e^{x_j}}$
`safe softmax`（主要防止输出过大溢出，就减最大值）:$x_i=\frac{e^{x_i-max(x_{:N})}}{\sum e^{x_j-max(x_{:N})}}$

---

https://zhuanlan.zhihu.com/p/676655352
https://mloasisblog.com/blog/ML/AttentionOptimization

## 3、`Multi-head Latent Attention`（`MLA`）

https://zhuanlan.zhihu.com/p/696380978

## 4、`Page Attention`（`vLLM`）

https://mloasisblog.com/blog/ML/AttentionOptimization
https://github.com/vllm-project/vllm

## 5、`Multi-Head Latent Attention`

https://planetbanatt.net/articles/mla.html
https://arxiv.org/pdf/2412.19437v1
https://www.cnblogs.com/theseventhson/p/18683602

## 参考
1、https://mloasisblog.com/blog/ML/AttentionOptimization
2、https://github.com/vllm-project/vllm
3、https://arxiv.org/pdf/2205.14135