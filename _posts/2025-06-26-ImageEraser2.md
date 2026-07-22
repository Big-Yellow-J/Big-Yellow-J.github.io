---
layout: mypost
title: 图像消除论文-2：SmartEraser、Erase Diffusion、OmniEraser
categories: 图像消除
address: 武汉🏯
extMath: true
show_footer_image: true
tags:
- diffusion model
- 图像消除
description: 图像消除是图像生成模型的重要应用领域，本文围绕Erase Diffusion、SmartEraser及OmniEraser等模型，阐述其核心技术与数据集构建方法。Erase
  Diffusion通过动态图像组合（输入与目标图像随解噪过程动态调整）、改进预测过程（计算“图像链”间损失）及注意力机制（融入mask）优化图像消除效果；SmartEraser构建合成数据集，包括实体过滤（基于语义分割如SAM提取实体，CLIP评分筛选合适大小实体）、背景筛选（从COCONut和SAM-1B数据集选取）、图像组合（保持实体大小一致，计算IoU与位置约束，经alpha
  blending合成），并改进模型输入与条件（图像编码嵌入文本编码），将规则mask处理为不规则；OmniEraser则利用视频数据构建数据集，通过混合高斯算法(MOG)分离背景与前景，计算MSE获取source-target图像对，结合GroundDINO+SAM2生成mask，微调FLUX.1-dev模型，测试采用prompt“'There
  is nothing here.'”。各模型从动态调整、数据集构建及模型优化等方面推动图像消除技术发展。
---

图像生成模型应用系列——图像消除：
[图像消除论文-1：PixelHacker、PowerPanint等](https://www.big-yellow-j.top/posts/2025/06/11/ImageEraser1.html)
[图像消除论文-2：消除类型数据集构建(1)](https://www.big-yellow-j.top/posts/2025/06/26/ImageEraser2.html)

## Erase Diffusion
> [Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Erase_Diffusion_Empowering_Object_Removal_Through_Calibrating_Diffusion_Pathways_CVPR_2025_paper.pdf)
> https://github.com/longtaojiang/SmartEraser
> CVPR-2025

### 1、模型结构

<img src="https://s2.loli.net/2025/06/28/dcKx2kr71oGFwV9.webp" alt="image" width="1280" height="648" loading="lazy" decoding="async" />

论文出发点主要为：1、**动态图像组合**：区别常规的图像去除实验**target image**就是我们的去除内容之后的图片，在该文中将其替换为：$x_t^{mix} = (1-\lambda_t)x_0^{ori}+ \lambda_t x_0^{obj}$ 也就是随着解噪过程（t逐渐减小）图片中所添加的实体（$x^{obj}_0$）所占的权重越来越小，同时将 **input image**也替换为动态的过程：$x_t^{min}=\sqrt{\alpha_t}x_t^{min}+ \sqrt{1- \alpha_t}\epsilon$；2、**改变模型的预测过程**：上面两部分公式处理之后那么得到的输入图像是一个“图像链”输出图像也是一个“图像链”，那么模型需要做的就是将对应“图像链”之间的loss进行计算。
<img src="https://s2.loli.net/2025/06/28/XHodtjyncSCDLV6.webp" alt="image" width="799" height="300" loading="lazy" decoding="async" />
3、**改进注意力计算方式**：这部分比较容易理解在计算注意力过程中将mask加入到计算也就是：$QK^T\bigodot Mask$
<img src="https://s2.loli.net/2025/06/28/EXbq2QGRWlImUjK.webp" alt="image" width="779" height="410" loading="lazy" decoding="async" />

## SmartEraser
> [SmartEraser: Remove Anything from Images using Masked-Region Guidance](https://openaccess.thecvf.com/content/CVPR2025/papers/Jiang_SmartEraser_Remove_Anything_from_Images_using_Masked-Region_Guidance_CVPR_2025_paper.pdf)
> CVPR-2025

### 1、数据集构建

<img src="https://s2.loli.net/2025/06/28/7ojzDsGYEHKc3XC.webp" alt="image" width="1127" height="370" loading="lazy" decoding="async" />

合成数据集构建思路上使用思路是：实体过滤背景检测而后将两部分进行组合。**Step-1：实体过滤**：直接通过语义分割模型（如SAM等）分割出实体之后，通过CLIP计算实体的score并且过滤掉过大/小的分割实体（保留5%-95%）进而获得需要粘贴的实体；**Step-2：过滤背景图片**：直接通过计算分辨率等从COCONut 和SAM-1B数据集中挑选出合适图片背景；**Step-3：图片组合**：首先将实体和背景图像中相同实体大小保持一致，而后通过计算我分割实体$c_1$ 以及背景中的实体 $c_i$之间的IoU：$R_1$，以及保证需要粘贴实体在整个背景中的位置（保证不超出背景图片）：$R_2$而后取两部分交集得到图像应该插入的合理位置。最后通过 `alpha blending`将两部分图像（实体+背景）进行组合。

### 2、模型结构测试效果
<img src="https://s2.loli.net/2025/06/28/1Wv6XI9bD87UTBs.webp" alt="image" width="1221" height="735" loading="lazy" decoding="async" />

论文主要就是将模型的输入进行改进：将模型图像输入由$[mask, image\bigodot (1-mask)]$ 改为 $[mask, image]$，除此之外将DF模型的condition改进（将图像编码嵌入到文本编码中）：$[\text{CLIP-TextEncoder(text)}, \text{MLP}(\text{Image}\bigodot \text{Mask})]$。除此之外就是将mask由“规则”（实体分割是规则的）变为“不规则”（将实体分割mask进行额外处理如膨胀处理等）最后测试效果是：

<img src="https://s2.loli.net/2025/06/28/G8HOtWoB1bhYEqP.webp" alt="image" width="1621" height="641" loading="lazy" decoding="async" />
> ME：将mask变不规则；RG：改变模型输入；VG：将图像编码嵌入到clip文本编码中

| 微调测试效果 |
|----------|
|<img src="https://s2.loli.net/2025/07/01/zkB2nCjVIdSwm6W.webp" alt="55_000000138891.jpg" width="1536" height="512" loading="lazy" decoding="async" />|
|<img src="https://s2.loli.net/2025/07/01/KrQehLwg1yuaEYB.webp" alt="sa_324589.jpg" width="1536" height="512" loading="lazy" decoding="async" />|
|<img src="https://s2.loli.net/2025/07/01/fhtiqNJug9Lz4WG.webp" alt="sa_326708.jpg" width="1536" height="512" loading="lazy" decoding="async" />|
|<img src="https://s2.loli.net/2025/07/01/V7eBwIMGoK9RAzZ.webp" alt="sa_324873.jpg" width="1536" height="512" loading="lazy" decoding="async" />|
|<img src="https://s2.loli.net/2025/07/01/PWQJ5gi39YthMBf.webp" alt="sa_5278781.jpg" width="1536" height="512" loading="lazy" decoding="async" />|

**值得注意的是**，在其合成的数据里面，合成得到结果**很粗糙**（感觉就像是随机贴图），因此感觉数据可用性不高
<img src="https://s2.loli.net/2025/07/01/QV4FMjNP2BgfhwS.webp" alt="image.png" width="516" height="516" loading="lazy" decoding="async" />
<img src="https://s2.loli.net/2025/07/01/816hmFUBvpQKuJX.webp" alt="image.png" width="619" height="612" loading="lazy" decoding="async" />

## OmniEraser
> https://pris-cv.github.io/Omnieraser/

### 1、数据集构建
通过视频来获取（mask-image）数据集，具体操作流程如下：

<img src="https://s2.loli.net/2025/06/26/LYclhNt4WmgRJpz.webp" alt="image.png" width="1618" height="627" loading="lazy" decoding="async" />

首先获取一段视频 $\mathbf{V}$ 通过 **混合高斯算法**（MOG）去检查视频中移动的物体以及静止的物体这样一来就可以得到两部分内容：Background和Foreground而后通过计算两部分之间的MSE（$MSE(V_i^{fg}, V_j^{bg})$）就可以得到source-image和 target-image对。对于mask内容直接通过 *GroundDINO+SAM2* 算法来构建mask这样一来就可以得到：foreground-image，mask，background-image。模型算法这是直接去微调 `FLUX.1-dev`

### 2、模型结构测试效果
<img src="https://s2.loli.net/2025/06/26/tcIhCEDeuGf3UXv.webp" alt="image.png" width="785" height="450" loading="lazy" decoding="async" />

实际测试效果（使用prompt为：`'There is nothing here.'`）

| 原图 | Mask | 结果 | 测试细节 |
|-----|------|------|--------|
|<img src="https://s2.loli.net/2025/06/26/znSUtwamOk9r47I.webp" alt="sa_324952.jpg" width="2258" height="1500" loading="lazy" decoding="async" />|<img src="https://s2.loli.net/2025/06/26/QXdWSb46FREakVN.webp" alt="sa_324952-0.jpg" width="2258" height="1500" loading="lazy" decoding="async" /> |<img src="https://s2.loli.net/2025/06/26/7pdgqO45CbDhluw.webp" alt="sa_324952.jpg" width="1024" height="1024" loading="lazy" decoding="async" /> | |
|<img src="https://s2.loli.net/2025/06/26/Bw4D9pEi7McULbv.webp" alt="sa_325886.jpg" width="2250" height="1500" loading="lazy" decoding="async" />|<img src="https://s2.loli.net/2025/06/26/P8mKbFdTqxZ19Yn.webp" alt="sa_325886-1.jpg" width="2250" height="1500" loading="lazy" decoding="async" /> |<img src="https://s2.loli.net/2025/06/26/89qmPaIY3tW1uUv.webp" alt="sa_325886.jpg" width="1024" height="1024" loading="lazy" decoding="async" /> | |
|<img src="https://s2.loli.net/2025/06/26/kxZjsRLSvpX96ne.webp" alt="sa_324501.jpg" width="1500" height="2250" loading="lazy" decoding="async" />|<img src="https://s2.loli.net/2025/06/26/bHMSowgfXm4sqO5.webp" alt="sa_324501-2.jpg" width="1500" height="2250" loading="lazy" decoding="async" /> |<img src="https://s2.loli.net/2025/06/26/GV9n6u1As3ZoqkJ.webp" alt="sa_324501.jpg" width="1024" height="1024" loading="lazy" decoding="async" /> | |
|<img src="https://s2.loli.net/2025/06/26/SA8rRFMc4Zjlp21.webp" alt="sa_324930.jpg" width="2250" height="1500" loading="lazy" decoding="async" />|<img src="https://s2.loli.net/2025/06/26/fQdXwRUCg5JVjs6.webp" alt="sa_324930-1.jpg" width="2250" height="1500" loading="lazy" decoding="async" /> |<img src="https://s2.loli.net/2025/06/26/Npr6tT9A75gwcY4.webp" alt="sa_324930.jpg" width="1024" height="1024" loading="lazy" decoding="async" /> |<img src="https://s2.loli.net/2025/06/26/xfBuX4RniAj7Z2D.webp" alt="image.png" width="2094" height="643" loading="lazy" decoding="async" />|