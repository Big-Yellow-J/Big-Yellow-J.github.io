---
layout: mypost
title: 深入浅出了解生成模型-7：生成加速策略概述
categories: 生成模型
extMath: true
images: true
address: 长沙🌷
show_footer_image: true
tags:
- 生成模型
- diffusion model
- 量化技术
show: true
stickie: true
description: 
---
## 扩散模型生成加速策略
Diffusion推理加速的方案，主要包括Cache、量化、分布式推理、采样器优化和蒸馏等。下面内容主要是去对Cache、计算加速框架以及量化技术进行介绍
> SD模型加速方式：[https://github.com/xlite-dev/Awesome-DiT-Inference?tab=readme-ov-file#Quantization](https://github.com/xlite-dev/Awesome-DiT-Inference?tab=readme-ov-file#Quantization)

### 一般加速框架
这部分内容的话比较杂，主要是介绍attn计算加速方式，比如说一般就是直接使用比如说`flash_attn`进行attention计算加速，比如说：
```python
pipeline.transformer.set_attention_backend("_flash_3_hub") # 启用flash attn计算加速
pipeline.transformer.reset_attention_backend()             # 关闭flash attn计算加速
```
使用`torch.compile`进行加速，不过值得注意的是**在开始使用过程中会比较慢**，因为在执行时，它会将模型编译为优化的内核，所以相对会比较慢，但是如果对编译后模型进行批量测试在时间上就会有所提升比如说在代码[df_acceralate.ipynb](code/Python/DFModelCode/DF_acceralate/df_acceralate.ipynb)中测试结果使用compile在z-image上生成5张图片耗时：86.49s（**平均生图时间**4s）不使用compile：29.92（**平均生图时间**5s）
#### 1、xFormers加速
> 项目地址：https://github.com/facebookresearch/xformers

在SD模型中对于xformers基本使用方式如下所示：
```python
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")
# 使用xformer加速
pipeline.enable_xformers_memory_efficient_attention()
# 关闭xformer加速
pipeline.disable_xformers_memory_efficient_attention()
```
xformers作用在于**加速attention计算并降低显存**，除此之外还提供了多种注意力实现方式，如casual attention等。根据[官方文档](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.fmha.cutlass.FwOp)中的描述，对于对于`xformers.ops.memory_efficient_attention`在使用上参数主要是：1、输入数据也就是QKV的格式上必须满足为：`[B, M, H, K]`分别表示的是其中B 为batch size, N为序列长度, num_heads为多头注意力头的个数, dim_head则为每个头对应的embeding size；2、attn_bias实际上充当为在使用mask attention时的mask；3、p也就是dropout对应值；4、op为Tuple，用于指定优化self-attention计算所采用的算子。基本使用方式如下：
```python
import xformers.ops as xops
y = xops.memory_efficient_attention(q, k, v)
y = xops.memory_efficient_attention(q, k, v, p=0.2) # 使用dropout
y = xops.memory_efficient_attention(
    q, k, v,
    attn_bias=xops.LowerTriangularMask()
)# 使用casual 注意力
```
值得着重了解的就是其中`attn_bias`参数，简单直观的理解：用于控制注意力可见性和结构的统一接口，**既可以表示 mask，也可以表示稀疏/局部/因果等高级注意力模式**，并且以高性能方式融入 attention 内核。比如说：
1、`xops.LowerTriangularMask()`：常规的causal注意力也就是下三角mask
2、`xops.LocalAttentionFromBottomRightMask`：局部注意力，每个token只能看最近的window_size个token
### cache策略
cache指的是：**缓存通过存储和重用不同层（例如注意力层和前馈层）的中间输出来加速推理，而不是在每个推理步骤执行整个计算**。它以更多内存为代价显着提高了生成速度，并且不需要额外的训练。主要详细介绍两种：1、DeepCache；2、FORA。对于更加多的cache策略可以看[知乎](https://zhuanlan.zhihu.com/p/711223667)，**推荐直接使用**[CacheDit](#cachedit)来进行加速。
#### DeepCache策略
> Paper:[https://arxiv.org/pdf/2312.00858](https://arxiv.org/pdf/2312.00858)
> Code:[https://link.zhihu.com/?target=https%3A//github.com/horseee/DeepCache](https://link.zhihu.com/?target=https%3A//github.com/horseee/DeepCache)

**主要针对UNet架构**的Diffusion模型进行推理加速。DeepCache 是一种Training-free的扩散模型加速算法，核心思想是**利用扩散模型序列去噪步骤中固有的时间冗余来减少计算开销**。
![](https://s2.loli.net/2026/01/13/7fSrYDnbHFLu6iG.png)
基于 U-Net 结构特性，发现相邻去噪步骤的高层特征具有显著时间一致性（Adjacent steps in the denoising process exhibit significant temporal similarity in high-level features.），比如说上图中作者在测试上采用block $U_2$的特征和其它所有的采样步之间相似性计算（图b），因此缓存这些高层特征并仅以低成本更新低层特征，从而避免重复计算。具体方法为：
![](https://s2.loli.net/2026/01/13/eXRHCFcdxLi2z7K.png)
比如说在官方的使用中有参数：`helper.set_params(cache_interval=3,cache_branch_id=0,)`表示是每3个时间步进行一次完成forward然后刷新cache，而其中参数cache_branch_id值得是一般而言在UNet中会定义`branch 0 → early / down blocks`等就是选择哪些层的输出。具体过程如下：t=1进行计算缓存，t=2,3都直接使用缓存，t=4完整计算得到缓存。
#### FORA
> Paper: [https://arxiv.org/pdf/2407.01425](https://arxiv.org/pdf/2407.01425)
> Code: [https://github.com/prathebaselva/FORA](https://github.com/prathebaselva/FORA)

**主要是争对Dit架构**的Diffusion模型进行推理加速。利用 Diffusion Transformer 扩散过程的重复特性实现了可用于DiT的Training-free的Cache加速算法。
![](https://s2.loli.net/2026/01/13/UCOEAJDLZNHXFW5.png)
FORA的核心在于发现Dit在去噪过程中，**相邻时间步的Attn和MLP层特征存在显著重复性**（如上图所示:在layer0、9、18、27这些层以及250步采样中，随后采样步约往后特征之间相似性也就越高。）。通过Caching特征，FORA 将这些重复计算的中间特征保存并在后续时间步直接复用，避免逐步重新计算。
![](https://s2.loli.net/2026/01/13/dSp5Zy9zua3gjw4.png)
具体而言，模型以固定间隔 N 重新计算并缓存特征：当时间步 t 满足 t mod N=0 时，更新所有层的缓存；在后续 N-1 步中，直接检索cached的 Attn 和 MLP 特征，跳过重复计算。这种策略利用了 DiT 架构在邻近时间时间步的特征相似性，在不修改DiT模型结构的前提下实现加速。例如，在 250 步 DDIM 采样中，当 N=3 时，模型仅需在第 3、6、9... 步重新计算特征，其余步骤复用Cache，使计算量减少约 2/3。实验表明，FORA对后期去噪阶段的特征相似性利用更为高效，此时特征变化缓慢，缓存复用的性价比最高。
#### CacheDit
[cache-dit](https://github.com/vipshop/cache-dit)这个框架主要是适用于Dit结构的扩散模型使用，其具体原理如下，其中具体使用如下：[df_acceralate.ipynb](code/Python/DFModelCode/DF_acceralate/df_acceralate.ipynb)
### 量化技术概述
[量化技术](https://www.big-yellow-j.top/posts/2025/10/11/Quantized.html)是一种模型压缩的常见方法，将模型权重从高精度（如FP16或FP32）量化为低比特位（如INT8、INT4）。常见的量化策略可以分为PTQ和QAT两大类。量化感知训练（Quantization-Aware Training）：在**模型训练过程中进行量化**，一般效果会更好一些，但需要额外训练数据和大量计算资源。后量化（Post-Training Quantization, PTQ）：在**模型训练完成后，对模型进行量化**，无需重新训练。对于线性量化下，浮点数与定点数之间的转换公式如下：$Q=\frac{R}{S}+Z;R=(Q-Z)*S$，其中R 表示量化前的浮点数、Q 表示量化后的定点数、S（Scale）表示缩放因子的数值、Z（Zero）表示零点的数值。
比如说在LLM中常用的两种**后量化技术**：1、**GPTQ量化技术**：通过量化——补偿——量化迭代方法，首先量化$W_{:,j}$，而后去计算误差并且补充到 $W_{:,j:(i+B)}$而后进行迭代实现所有参数的量化；2、**AWQ量化技术**：模型计算过程中只有关键参数起作用因此对于关键参数保持原来的精度(FP16)，对其他权重进行低比特量化，但是这样不同进度参数会导致硬件问题，因此在AWQ中**对所有权重均进行低比特量化，但是，在量化时，对于显著权重乘以较大的scale，相当于降低其量化误差；同时，对于非显著权重，乘以较小的scale，相当于给予更少的关注。**
#### Bitsandbytes 量化
通过使用bitsandbytes量化来实现8-bit（int8）或者4-bit（int4、Qlora中一般就会使用）量化，不过区别上面提到的AWQ以及GPTQ量化，bitsandbytes属于量化感知训练，前者需要通过数据来保证量化精度（量化过程是离线、一次性过程），后者量化过程是即时的可逆的。其技术原理如下：$w≈s q$其中w表示原始的FP16权重，q代表int4/int8权重，s缩放因子，其量化过程为对每一个block权重计算：$\max(\text{abs}(w))$而后去计算scale：$s=\frac{amx(\| w\|)}{2^{b-1}-1}$而后代入公式就可以得到量化后权重，推理过程中进行：反量化 + 矩阵乘法融合在一个 CUDA kernel 中完成：$Y=X(sq)$。因此对于其使用也很简单，比如说在代码中：[cache_acceralate.py](code/Python/DFModelCode/DF_acceralate/cache_acceralate.py)
```python
# 在ZImagePipeline中参数为：
class ZImagePipeline(DiffusionPipeline, ZImageLoraLoaderMixin, FromSingleFileMixin):
    def __init__(,..,vae, text_encoder, tokenizr, transformer):
        ...
# 因此可以直接对里面的text_encoder使用量化处理

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
quantization_config = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,# 在模型加载阶段，将权重以 4-bit 量化形式加载
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,# 指定 反量化后参与计算的 dtype
    bnb_4bit_use_double_quant=True,#启用 Double Quantization（双重量化），也就是对block的scale在进行一次量化
    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],# 指定 不参与 bitsandbytes 量化的模块
)
transformer = AutoModel.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    mirror='https://hf-mirror.com'
)
```
去对你的`model_name`里面的transformer进行量化处理，除此之外还有使用例子就是进行优化器量化，比如说
```python
# 和使用adamw方式一样，使用qlora使用一般带上这个优化器
import bitsandbytes as bnb
optimizer_class = bnb.optim.AdamW8bit
```
#### SVDQuant量化
> https://github.com/nunchaku-ai/nunchaku

#TODO: 量化侯模型如何进行后训练可以直接使用 flux1-dev-kontext_fp8_scaled.safetensors 进行介绍
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTg3NDMyNDk4XX0=
-->