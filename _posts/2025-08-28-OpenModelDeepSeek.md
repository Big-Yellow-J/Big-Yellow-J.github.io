---
layout: mypost
title: 开源模型技术总结-2————DeepSeek系列模型
categories: 多模态
extMath: true
images: true
address: 武汉🏯
tags:
- cv-backbone
- 多模态
- llm
- multimodal
- GRPO
- PPO
show_footer_image: true
special_tag: 长期更新
description: DeepSeek v3采用混合专家模型，新增辅助损失平衡专家不均衡，结构创新包括低秩优化KV-cache（降维升维减少显存）和混合专家机制（Routed
  Expert直接传入隐藏层，Shared Expert经门控筛选），集成Multi-Token Prediction技术。DeepSeek-R1基于思维链（CoT），通过Group
  Relative Policy Optimization（GRPO）训练，含生成回答、评分、裁剪更新（clip函数限定值）及KL散度惩罚偏差步骤。DeepSeek
  OCR以视觉压缩长文本上下文，将多token文本转为图片减少tokens，模型采用SAM-base（patch-size 16）+2层Conv（16倍下采样）+CLIP-large（去patch-embedding），1024x1024图片经处理后token从4096压缩至256，类似技术如Glyph通过文本转image实现压缩。
---

## DeepSeek系列
### DeepSeek v3
**DeepSeek v3**[^4]（简称**DS**）各类技术细节，对于**DS**在模型结构上和之前迭代版本的 **DS-2**无太大区别，还是使用混合专家模型，只是补充一个辅助损失去平衡不同专家之间的不均衡问题。
![](https://s2.loli.net/2025/06/21/oqabhYwBMjPSzZU.webp)
> 左侧结构和 **GPT-2**结构类似

在结构上**DS**主要的创新点在于：1、[Multi-Head Latent Attention](https://www.big-yellow-j.top/posts/2025/01/29/Attention.html)；2、[DeepSeekMoE](https://www.big-yellow-j.top/posts/2025/MoE-KV-cache.html)。前者为优化 **KV-cache** 操作，通过一个低秩的$c_r^{KV}$代替原本占用较高的QV的值（首先通过降维方式降低原本维度，这样一来在显存占用上就会降低，而后通过升维方式，恢复到原本的维度），后者为混合专家模型，不过区别于常用的`MoE`方法，在**DS**中将专家模型分为两类：1、**Routed Expert**；2、**Shared Expert**，前者**直接**将隐藏层的输入进行传入，后者则是通过门控网络**筛选**而后隐藏层的输入进行传入。
除此之外，在**DS**中使用**Multi-Token Prediction**（MTP）技术
![](https://s2.loli.net/2025/06/21/4OlDfbA6pgo5NrF.webp)
在**DS**中一个很耀眼的功能就是：**DeepSeek-R1**（一种思维链技术：**CoT**:**Chain of Thought**，在GPT-o1中也使用到这种技术）结合论文[^5]中对 **CoT**技术的描述，可以简单的理解为：让LLM可以自主去思考问题，相较之直接让GPT输出答案，区别在于还要他给出推理过程，比如说在DeepSeek中对于思维链的prompt：
![](https://s2.loli.net/2025/06/21/l7eHa3xMZjwGOP8.webp)
直接让模型去输出think内容，一般而言如果要训练一个具有思维链功能的模型可以简单按照如下过程进行处理：
```python
# 数据集简单类型
{
  "instruction": "请一步一步思考并解答以下问题。",
  "input": "小明有5个苹果，给了小红3个，还剩多少？",
  "output": "<think>第一步：...\n第二步：...\n第三步：...</think>答案：2个"
}
# 微调模型过程中 prompt
你是一个严谨的推理助手，回答任何问题前必须先在 <think> 标签内进行完整逐步思考，再给出最终答案。
```
通过上面简单过程去“强迫”模型去输出思考过程而不是直接就给出答案。除此之外在**DS-R1**中模型的整体训练过程使用**Group Relative Policy Optimization**（GRPO）策略进行优化
![](https://s2.loli.net/2025/06/21/CBnfpTwjQXNkybG.webp)
对于上述优化过程[理解](https://medium.com/@sahin.samia/the-math-behind-deepseek-a-deep-dive-into-group-relative-policy-optimization-grpo-8a75007491ba)：比如说对于一个数学问题：$8+5=?$，这就是上面公式中所提到的question $q$，按照上面的描述，将会生成一系列的输出：${o_1,...,o_G}$
**Step-1**：生成若干的回答。${o_1,...,o_G}$
**Step-2**：对于生成的回答进行评分。${r_1,...,r_G}$，而后计算$A_i=\frac{r_i- \text{mean}({r_1,...,r_G})}{\text{std}({r_,...,r_G})}$
**Step-3**：使用裁剪更新策略：$\text{clip}(\frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}(o_i|q)}},1-\epsilon,\epsilon)$比如说：如果新策略开始给o1分配过高的概率，裁剪机制确保不会过度强调这个响应。这种方式保证了即使在像推理这样复杂的任务中，策略优化也能保持稳定和可靠。通过clip函数将内部值限定在$(1-\epsilon, 1+\epsilon)$之间
**Step-4**：通过KL散度（用来度量两个概率分布相似度的指标）惩罚偏差
## DeepSeek OCR
### DeepSeek OCRv1
DeepSeek OCR[^1]主要内容就是尝试**使用视觉的方式去压缩长文本上下文**，按照论文里面的描述就是：$f_{dec}:R^{n\times d_{latent}}\rightarrow R^{N\times d_{text}}, \hat{X}=f_{dec}(X)$
前面部分代表压缩的视觉tokens后面代表重构的文本表述。其实从上面公式就可以了解在DeepSeek OCR中做的就是：对于原始文本输入需要较长的tokens数量（比如说1w个字），但是如果这1w个文本都在图片上可能就是512个tokens。
> 但是作者只是在OCR邻域做测试，正如论文里面说的：
> It is reasonable to conjecture that LLMs, through specialized pretraining optimization, would demonstrate more natural integration of such capabilities.

![](https://s2.loli.net/2025/11/11/IxuHpXCj2hJ3sTU.webp)
对于传统多模态中的视觉结构：第一种使用多个视觉编码器进行编码处理，第二种：将图片切割为不同的patch而后进行处理，第三种：使用动态分辨率而后将图片去切割为不同patch进行编码。论文中使用的模型结构（为了实现：1、处理高分辨率；2、高分辨率小低激活；3、较少的视觉tokens；4、支持多分辨率输入；5、计算参数少）为：**SAM-base**（patch-size：16）+**Conv**（2层，kernel_size=3,strid=2, paddingg=1去对视觉token进行16倍下采样）+**CLIP-large**（去掉patch-embedding因为我的输入就是patch了），那么对于1024x1024首先划分为1024/16 × 1024/16 = 4096个patch token，在对4096个token进行压缩，数量变为4096/16 = 256。
![](https://s2.loli.net/2025/11/11/GhRspCQc9LHOJPA.webp)
在许多论文里面也用到了压缩技术（*截至到：2025.10.23*部分论文），比如说Glyph（Zhipu-清华）[^2]和另外一篇论文[^3]
![](https://s2.loli.net/2025/11/11/hosSXQyPOlxLYvc.webp)
对于这些内容核心的思路都是将文本转化为image来进行压缩tokens比如在论文[^11]中直接将text转化为latex格式的图片而后通过模型进行处理。

<!-- ### DeepSeek OCRv2
在论文中[^6] -->

## 参考
[^1]: [https://www.arxiv.org/pdf/2510.18234](https://www.arxiv.org/pdf/2510.18234)
[^2]: [https://arxiv.org/pdf/2510.17800](https://arxiv.org/pdf/2510.17800)
[^3]: [https://arxiv.org/pdf/2510.18279](https://arxiv.org/pdf/2510.18279)
[^4]: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948)
[^5]: [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903)
[^6]: [https://arxiv.org/pdf/2601.20552](https://arxiv.org/pdf/2601.20552)