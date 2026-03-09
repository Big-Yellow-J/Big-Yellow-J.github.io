---
layout: mypost
title: 开源模型技术总结-3————FireRed（小红书）开源模型
categories: 生成模型
extMath: true
images: true
address: 长沙🌷
show_footer_image: true
tags:
- 生成模型
- diffusion model
- 开源模型
---
## FireRED OCR
OCR[^1]模型主要是对Qwen3-vl进行微调的一个模型，因此主要了解一下其数据构建过程以及其训练思路即可，因此对于其模型 **数据构建**过程中：
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260309220245004.png)
**对数据进行聚类处理**（筛选去重），直接通过轻量的ResNet/Vit对图像进行编码而后通过聚类算法（KNN等）去筛选出相似度高的、保留长尾数据（如表格等）；**数据分类**、这部分主要是对数据的语言类型、布局、数据来源（PDF、扫描件等）进行分类保证数据类别的分布；**数据清洗**：这部分主要是对数据进行识别（比如说有些数据可能只有图像）就需要使用PaddleOCR-VL进行识别、合成数据（通过提前设计好html/css模板去合成表格等数据）、还有一些内容就直接通过大模型去判断最后标签是不是有效的，比如说对于识别失败的，直接用llm判断markdown中格式不是都正确是不是有缺失，并且进行修复，判断图像是不是“好的”没有质量太低的（直接丢弃）

**模型训练**过程不是直接SFT+RL而是在此之前先去给模型赋予“文档”识别能力（可能基于Qwen3-VL非专门的OCR模型因此**先去强化OCR中主要的3类任务能力，减轻后续SFT、RL难度**）：**第一阶段**赋予模型文档识别能力：主要是对文档中进行多任务的微调包含如下3组任务：目标检测以及OCR、特定区域OCR（主要是通过提供prompt去识别对应区域内容）、页面转换（将布局转换为Markdown内容）；**第二阶段**监督微调：这部分数据相对去第一部分中的数学质量更加高，比如说转化公式严格满足和markdown保持一致，文档的种类更加丰富。**第三阶段**GRPO强化学习（主要是争对模型最后输出**内容的格式**），对于GRPO中奖励函数主要对4类内容进行打分：数学公式评分（直接通过Latex解析器处理数学公式如果失败处理记分-1，成功则是按照公式复杂度进行评分）、表格完整性评分（检查行/列是否一致及1或0）、文本准确性评分（计算文本之间Levenshtein distance，这个距离主要是计算从A-->B字符串需要进行多少次计算）、层级闭合（主要是记录标签是否闭合，比如html中 tag 之间）。最后对上面4类进行加权即为最后奖励值。
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260309220052574.png)
> 对于训练过程可以总结为：粗-->细-->强化

## FireRED Edit模型
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260309221805928.png)

## 参考
[^1]: [FireRed-OCR Technical Report](https://arxiv.org/pdf/2603.01840)
[^2]: [FireRed-Image-Edit-1.0 Technical Report](https://arxiv.org/pdf/2602.13344)
