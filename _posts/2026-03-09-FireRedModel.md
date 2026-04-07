---
layout: mypost
title: 开源模型技术总结-3————FireRed（小红书）开源模型
categories: 多模态
extMath: true
images: true
address: 长沙🌷
show_footer_image: true
tags:
- 生成模型
- diffusion model
- 开源模型
description: FireRED OCR基于Qwen3-vl微调，数据构建环节采用聚类去重保留长尾数据、多维度分类保证分布均衡、多工具联动清洗修复三类处理，训练分三阶段推进：先预训练强化目标检测、特定区域识别、页面转Markdown三类文档识别能力，再用高质量数据做监督微调，最后通过GRPO强化学习优化输出格式，奖励函数覆盖数学公式、表格完整性、文本准确性、标签闭合四类维度。FireRED
  Edit采用分桶采样策略提升显存利用率，优化DPO策略区分择优避坏参数，搭配DiffusionNFT自生成训练数据，双维度奖励函数结合一致性损失规避生成图像细节崩溃。
---

首先对于FireRED OCR模型以及FireRED Edit虽然都是模型微调，但是对于其训练过程还是很有参考意义，比如Edit模型中通过训练强化模型对于细节的感知能力（这里可以对一些对细节要求很高的生成模型训练很有启发意义，而不是直接拿着数据直接SFT看效果）
## FireRED OCR
OCR[^1]模型主要是对Qwen3-vl进行微调的一个模型，因此主要了解一下其数据构建过程以及其训练思路即可，因此对于其模型 **数据构建**过程中：
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260309220245004.png)
**对数据进行聚类处理**（筛选去重），直接通过轻量的ResNet/Vit对图像进行编码而后通过聚类算法（KNN等）去筛选出相似度高的、保留长尾数据（如表格等）；**数据分类**、这部分主要是对数据的语言类型、布局、数据来源（PDF、扫描件等）进行分类保证数据类别的分布；**数据清洗**：这部分主要是对数据进行识别（比如说有些数据可能只有图像）就需要使用PaddleOCR-VL进行识别、合成数据（通过提前设计好html/css模板去合成表格等数据）、还有一些内容就直接通过大模型去判断最后标签是不是有效的，比如说对于识别失败的，直接用llm判断markdown中格式不是都正确是不是有缺失，并且进行修复，判断图像是不是“好的”没有质量太低的（直接丢弃）

**模型训练**过程不是直接SFT+RL而是在此之前先去给模型赋予“文档”识别能力（可能基于Qwen3-VL非专门的OCR模型因此**先去强化OCR中主要的3类任务能力，减轻后续SFT、RL难度**）：**第一阶段**赋予模型文档识别能力：主要是对文档中进行多任务的微调包含如下3组任务：目标检测以及OCR、特定区域OCR（主要是通过提供prompt去识别对应区域内容）、页面转换（将布局转换为Markdown内容）；**第二阶段**监督微调：这部分数据相对去第一部分中的数学质量更加高，比如说转化公式严格满足和markdown保持一致，文档的种类更加丰富。**第三阶段**GRPO强化学习（主要是争对模型最后输出**内容的格式**），对于GRPO中奖励函数主要对4类内容进行打分：数学公式评分（直接通过Latex解析器处理数学公式如果失败处理记分-1，成功则是按照公式复杂度进行评分）、表格完整性评分（检查行/列是否一致及1或0）、文本准确性评分（计算文本之间Levenshtein distance，这个距离主要是计算从A-->B字符串需要进行多少次计算）、层级闭合（主要是记录标签是否闭合，比如html中 tag 之间）。最后对上面4类进行加权即为最后奖励值。
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260309220052574.png)
> 对于训练过程可以总结为：粗-->细-->强化

## FireRED Edit
模型[^2]结构如下（[还是对qwen-image-edit做的微调](https://github.com/FireRedTeam/FireRed-Image-Edit/issues/10#issuecomment-3959708960)，那么qwen-image-edit的lora也可以直接拿来用）
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260309221805928.png)
对于训练过程中，**输入数据处理过程**主要是对数据进行分桶保证显卡最大利用：计算batch的总视觉token数量尽量接近某个固定值C，而后在一个桶里面找一个最合适的分辨率（$\text{argmin}\sum \vert (H_iW_i-hw \vert)$ 小的hw为桶的尺寸）保证尺寸一致。image-text打乱处理，比如说交换Fig位置对应文本中位置也发生改变、随机丢弃图像文本也对应丢弃描述。
> 分桶后，训练过程一般优先从相同bucket中挑选样本组成batch；当bucket内样本不足batch_size时，会通过设置repeat参数（重复次数）来虚拟扩充样本量，从而保证训练能稳定、连续地进行，而不会频繁丢弃不完整batch。

**模型训练策略**：将VLM部分视觉提取特征提前处理好存储在本地避免训练过程中再计算、FSDP（梯度优化器状态切分放到节约显存）
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260310213427159.png)
模型训练分为5个阶段，每个阶段数据质量是越来越高的，主要看里面的DPO以及NFT处理过程。**首先DPO过程**，作者训练过程中发现对于Win以及Lose都会发生上升（也就是选择“好”以及选择“坏”的概率都在上升）：
$$\mathcal{L}_{\text {Ours }}=-\mathbb{E}_{\left(c, x_{w}, x_{l}\right) \sim \mathcal{D}}[\log \sigma(\beta[\underbrace{\left(\mathcal{L}_{l}^{\theta}-\mathcal{L}_{l}^{\text {ref }}\right)}_{\text {Lose Diff }}-\omega \cdot \underbrace{\left(\mathcal{L}_{w}^{\theta}-\mathcal{L}_{w}^{\text {ref }}\right)}_{\text {Win Diff }}])-\lambda \mathcal{L}_{w}^{\theta}]$$
对比普通的DPO就是将择优以及避坏都用不同参数而非相同的参数进行控制（$\beta$ 以及 $w$）。使用 *DiffusionNFT*优化策略，在DPO中需要成对的数据进行训练而DiffusionNFT[^3]模型自己生成图片，然后通过一个奖励模型给这张图打分，这个分数被转化为最优概率 $r \in [0, 1]$
$$\mathcal{L}_{\mathrm{NFT}}=\mathbb{E}_{t, x_{0} \sim \pi^{\mathrm{old}}}[r \underbrace{\left\|v_{\theta}^{+}\left(x_{t}, t\right)-v\right\|^{2}}_{\text {Positive Match }}+(1-r) \underbrace{\left\|v_{\theta}^{-}\left(x_{t}, t\right)-v\right\|^{2}}_{\text {Negative Match }}]$$
对于里面奖励函数评分主要是两个：1、Fine-grained Logit-Weighted Ensembling Reward：一般RL过程中奖励值一般是离散的也就是1-5，论文里面做法让 reward model 先写 CoT 推理，然后只看它对[1,2,3,4,5]这几个数字 token 的 logits（**具体过程**：输入一张图像，先通过VLM输出COT回答最后给出评分，因为LLM生成token会在1-5每个数字都生成一个概率然后选择最大概率最为模型输出，因此我在llm输出数字时候变相的得到了1-5这几个数字的概率），用 softmax 做软概率加权平均，得到一个 [1.0～5.0] 区间内的连续软分数 → 再 ensemble 多条推理路径取平均 → 得到最终奖励 R；2、Layout-Aware OCR-based Reward：这个奖励函数主要是针对文字生成进行优化处理同时考虑文字的生成以及文字的布局是否准确；
**一致性损失**主要是为了保证训练过程中图像不“崩溃”（解噪过程中前期生成“轮廓”，后期生成“细节”，这个过程主要是保证细节不崩溃，如人脸改变等）
$$\mathcal{L}_{id} = \frac{1}{N} \sum_{i=1}^{N} \left( 1 - \frac{\phi(\mathcal{T}_i(x_0)) \cdot \phi(\mathcal{T}_i(x_{gt}))}{\|\phi(\mathcal{T}_i(x_0))\|_2 \cdot \|\phi(\mathcal{T}_i(x_{gt}))\|_2} \right)$$
里面的 $x_0$表示的是直接一步生成结果（之所以不用最后结果是因为你最后生成图像整个图像信息已经被确定了，如果对中间过程进行改进最后图像也会发生改进）
## 参考
[^1]: [FireRed-OCR Technical Report](https://arxiv.org/pdf/2603.01840)
[^2]: [FireRed-Image-Edit-1.0 Technical Report](https://arxiv.org/pdf/2602.13344)
[^3]: [https://arxiv.org/pdf/2509.16117](https://arxiv.org/pdf/2509.16117)