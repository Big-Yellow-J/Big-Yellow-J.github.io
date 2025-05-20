---
layout: mypost
title: 深入浅出了解生成模型-3：Diffusion模型原理以及代码
categories: 生成模型
extMath: true
images: true
address: 武汉🏯
show_footer_image: true
tags: [cv-backbone,生成模型,diffusion model]
show: true
description: 日常使用比较多的生成模型比如GPT/Qwen等这些大多都是“文生文”模型（当然GPT有自己的大一统模型可以“文生图”）但是网上流行很多AI生成图像，而这些生成图像模型大多都离不开下面三种模型：1、GAN；2、VAE；3、Diffusion Model。因此本文通过介绍这三个模型作为生成模型的入门。本文主要介绍三类Diffusion Model
---

前文已经介绍了VAE以及GAN这里介绍另外一个模型：Diffusion Model，除此之外介绍Conditional diffusion model、Latent diffusion model

## Diffusion Model
diffusion model（后续简称df）模型原理很简单：*前向过程*在一张图像基础上不断添加噪声得到一张新的图片之后，*反向过程*从这张被添加了很多噪声的图像中将其还原出来。原理很简单，下面直接介绍其数学原理：
![https://arxiv.org/pdf/2208.11970](https://s2.loli.net/2025/05/19/zofsq8ky7GnLjm9.png)

> 上图中实线代表：反向过程（去噪）；虚线代表：前向过程（加噪）

那么我们假设最开始的图像为 $x_0$通过不断添加噪声（添加噪声过程假设为$t$）那么我们的 **前向过程**：$q(x_1,...,x_T\vert x_0)=q(x_0)\prod_{t=1}^T q(x_t\vert x_{t-1})$，同理 **反向过程**：$p_\theta(x_0,...\vert x_{T})=p(x_T)\prod_{t=1}^Tp_\theta(x_{t-1}\vert x_t)$

### 前向过程
在df的前向过程中：

$$
q(x_1,...,x_T\vert x_0)=q(x_0)\prod_{t=1}^T q(x_t\vert x_{t-1})
$$

通常定义如下的高斯分布：$q(x_t\vert x_{t-1})=N(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_tI)$，其中参数$\beta$就是我们的 **噪声调度**参数来控制我们每一步所添加的噪声的“权重”（这个权重可以固定也可以时间依赖，对于时间依赖很好理解最开始图像是“清晰”的在不断加噪声过程中图像变得越来越模糊），于此同时随着不断的添加噪声那么数据$x_0$就会逐渐的接近标准正态分布 $N(0,I)$的 $x_t$，整个加噪过程就为：

$$
\begin{align*}
t=1 \quad & x_1 = \sqrt{1 - \beta_1} x_0 + \sqrt{\beta_1} \epsilon_1 \\
t=2 \quad & x_2 = \sqrt{1 - \beta_2} x_1 + \sqrt{\beta_2} \epsilon_2 \\
&\vdots \\
t=T \quad & x_T = \sqrt{1 - \beta_T} x_{T-1} + \sqrt{\beta_T} \epsilon_T
\end{align*}
$$

在上述过程中我们可以将$t=1$得到的 $x_1$代到下面 $t=2$的公式中，类似的我们就可以得到下面的结果：$x_2=\sqrt{(1-\beta_2)(1-\beta_1)}x_0+ \sqrt{1-(1-\beta_2)(1-\beta_1)}\epsilon$ （之所以用一个$\epsilon$是因为上面两个都是服从相同高斯分布就可以直接等同过来）那么依次类推就可以得到下面结果：

$$
\begin{align*}
    x_T=\sqrt{(1-\beta_1)\dots(1-\beta_T)}x_0+ \sqrt{1-(1-\beta_1)\dots(1-\beta_T)}\epsilon \\
\Rightarrow x_T=\sqrt{\bar{\alpha_T}}x_0+ \sqrt{1-\bar{\alpha_T}}\epsilon
\end{align*}
$$

其中：$\bar{\alpha_T}=\sqrt{(1-\beta_1)\dots(1-\beta_T)}$，那么也就是说对于前向过程（加噪过程）可以从$x_0$到 $x_T$一步到位，不需要说再去逐步计算中间状态了。

### 反向过程
**反向过程**：$p_\theta(x_0,...\vert x_{T})=p(x_T)\prod_{t=1}^Tp_\theta(x_{t-1}\vert x_t)$，也就是从最开始的标准正态分布的 $x_t$逐步去除噪声最后还原得到 $x_0$。仔细阅读上面提到的前向和反向过程中都是条件概率但是在反向传播过程中会使用一个参数$\theta$，这是因为前向过程最开始的图像和噪声我们是都知道的，而反向过程比如$p(x_{t-1}\vert x_t)$是难以直接计算的，需要知道整个数据分布，因此我们可以通过神经网路去近似这个分布，而这个神经网络就是我们的参数：$\theta$。于此同时反向过程也会建模为正态分布：$p_\theta(x_{t-1}\vert x_t)=N(x_{t-1};\mu_\theta(x_t,t),\sum_\theta(x_t,t))$，其中 $\sum_\theta(x_t,t)$为我们的方差对于在值可以固定也可以采用网络预测[^1]
> 在OpenAI的Improved DDPM中使用的就是使用预测的方法：$\sum_\theta(x_t,t)=\exp(v\log\beta_t+(1-v)\hat{\beta_t})$，直接去预测系数：$v$

回顾一下生成模型都在做什么。在[GAN](./2025-05-08-GAN.md)中是通过 *生成器网络* 来拟合正式的数据分布也就是是 $G_\theta(x)≈P(x)$，在 [VAE](./2025-05-11-VAE.md)中则是通过将原始的数据分布通过一个 低纬的**潜在空间**来表示其优化的目标也就是让 $p_\theta(x)≈p(x)$，而在Diffusion Model中则是直接通过让我们 去噪过程得到结果 和 加噪过程结果接近，什么意思呢？df就像是一个无监督学习我所有的GT都是知道的（每一步结果我都知道）也就是是让：$p_\theta(x_{t-1}\vert x_t)≈p(x_{t-1}\vert x_t)$ 换句话说就是让我们最后解码得到的数据分布和正式的数据分布相似：$p_\theta(x_0)≈p(x_0)$ 既然如此知道我们需要优化的目标之后下一步就是直接构建损失函数然后去优化即可。

### 优化过程
通过上面分析，发现df模型的优化目标和VAE的优化目标很相似，其损失函数也是相似的，首先我们的优化目标是最大化下面的边际对数似然：$\log p_\theta(x_0)=\log \int_{x_{1:T}}p_\theta(x_0,x_{1:T})dx_{1:T}$，对于这个积分计算是比较困难的，因此引入：$q(x_{1:T}\vert x_0)$ 那么对于这个公式有：

$$
\begin{align*}
    \log p_\theta(x_0)&=\log \int_{x_{1:T}}p_\theta(x_{0:T})dx_{1:T} \\
    &=\log \int_{x_{1:T}} q(x_{1:T}\vert x_0)\frac{p_\theta(x_{0:T})}{q(x_{1:T}\vert x_0)}dx_{1:T}\\
    &=\mathbb{E}_{q(x_{1:T|x_0})}[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\vert x_0)}]
\end{align*}
$$

## Conditional Diffusion Model

## Latent Diffusion Model

## DF模型
### Dit模型
将Transformer使用到Diffusion Model中，而Dit[^2]属于Latent Diffusion Model也就是在通过一个autoencoder来将图像压缩为低维度的latent，扩散模型用来生成latent，然后再采用autoencoder来重建出图像，比如说在Dit中使用KL-f8对于输入图像维度为：256x256x3那么压缩得到的latent为32x32x4。Dit的模型结构为：
![image.png](https://s2.loli.net/2025/05/19/K8frUqVY4la7Xeg.png)

模型输入参数3个分别为：1、低纬度的latent；2、标签label；3、时间步t。对于latent直接通过一个patch embed来得到不同的patch（得到一系列的token）而后将其和位置编码进行相加得到最后的embedding内容，直接结合代码[^3]来解释模型：
假设模型的输入为：

```python
#Dit参数为：DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6)
batch_size= 16
image = torch.randn(batch_size, 4, 32, 32).to(device)
t = torch.randint(0, 1000, (batch_size,)).to(device)
y = torch.randint(0, 1000, (batch_size,)).to(device)
```

那么对与输入分别都进行embedding处理：1、**Latent Embedding：得到（8，64，384）**，因为patchembedding直接就是假设我们的patch size为4那么每个patch大小为：4x4x4=64并且得到32/4* 32/4=64个patches，而后通过线linear处理将64映射为hidden_size=384；2、**Time Embedding和Label Embedding：得到（8，384）（8，384）**，因为对于t直接通过sin进行编码，对于label在论文里面提到使用 *classifier-free guidance*方式，具体操作就是在**训练过程中**通过`dropout_prob`来将输入标签**随机**替换为无标签来生成无标签的向量，在 **推理过程**可以通过 `force_drop_ids`来指定某些例子为无条件标签。将所有编码后的内容都通过补充位置编码信息（latent embedding直接加全是1，而label直接加time embedding），补充完位置编码之后就直接丢到 `DitBlock`中进行处理，对于`DitBlock`结构：
```python
def forward(self, x, c):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
    x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
    x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
    return x
```

在这个代码中不是直接使用注意力而是使用通过一个 `modulate`这个为了实现将传统的layer norm（$\gamma{\frac{x- \mu}{\sigma}}+ \beta$）改为动态的$\text{scale}{\frac{x- \mu}{\sigma}}+ \text{shift}$，直接使用动态是为了允许模型根据时间步和类标签调整 Transformer 的行为，使生成过程更灵活和条件相关，除此之外将传统的残差连接改为 权重条件连接 $x+cf(x)$。再通过线性层进行处理类似的也是使用上面提到的正则化进行处理，处理之后结果通过`unpatchify`处理（将channels扩展2倍而后还原到最开始的输入状态）
## DF训练
* **传统训练**

对于传统的DF训练（前向+反向）比较简单，直接通过输入图像而后不断添加噪声而后解噪。以huggingface[^4]上例子为例（测试代码: [Unet2Model.py]('Big-Yellow-J.github.io/code/Unet2Model.py.txt')），**首先**、对图像进行添加噪声。**而后**、直接去对添加噪声后的模型进行训练“去噪”（也就是预测图像中的噪声）。**最后**、计算loss反向传播。
> 对于加噪声等过程可以直接借助 `diffusers`来进行处理，对于diffuser：
> 1、schedulers：调度器
> 主要实现功能：1、图片的前向过程添加噪声（也就是上面的$x_T=\sqrt{\bar{\alpha_T}}x_0+ \sqrt{1-\bar{\alpha_T}}\epsilon$）；2、图像的反向过程去噪；3、时间步管理等。如果不是用这个调度器也可以自己设计一个只需要：1、前向加噪过程（需要：使用固定的$\beta$还是变化的、加噪就比较简单直接进行矩阵计算）；2、采样策略

```python
def get_beta_schedule(timesteps, start=beta_start, end=beta_end, schedule='linear'):
    if schedule == 'linear':
        betas = torch.linspace(start, end, timesteps, device=device)
    elif schedule == 'cosine':
        s = 0.008  # 余弦调度的平滑参数
        timesteps_tensor = torch.arange(timesteps, device=device, dtype=torch.float32)
        f_t = torch.cos((timesteps_tensor / timesteps + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = betas.clamp(min=1e-4, max=0.999)
    else:
        raise ValueError("Unsupported schedule type")
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

betas, alphas, alphas_cumprod = get_beta_schedule(timesteps, schedule='linear')

# 前向扩散：添加噪声
def q_sample(x0, t, noise=None):
    """
    在时间步 t 为图像 x0 添加噪声
    Args:
        x0: 干净图像，形状 (N, C, H, W)
        t: 时间步，形状 (N,)
        noise: 噪声张量，形状同 x0
    Returns:
        带噪图像 x_t
    """
    if noise is None:
        noise = torch.randn_like(x0).to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[t]).view(-1, 1, 1, 1)
    return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise

# DDIM 逆向采样
def ddim_step(x_t, t, pred_noise, t_prev, eta=0.0):
    """
    DDIM 去噪一步
    Args:
        x_t: 当前带噪图像，形状 (N, C, H, W)
        t: 当前时间步（整数）
        pred_noise: 模型预测的噪声
        t_prev: 下一时间步（t-1 或跳跃步）
        eta: 控制随机性的参数（0.0 表示确定性）
    Returns:
        x_{t-1}：去噪后的图像
    """
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    alpha_t_prev = alphas_cumprod[t_prev].view(-1, 1, 1, 1) if t_prev >= 0 else torch.tensor(1.0, device=device)
    
    # 预测 x_0
    pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
    pred_x0 = pred_x0.clamp(-1, 1)  # 防止数值溢出
    
    # 计算方向（噪声部分）
    sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
    noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
    
    # DDIM 更新
    x_t_prev = (torch.sqrt(alpha_t_prev) * pred_x0 + 
                torch.sqrt(1 - alpha_t_prev - sigma_t**2) * pred_noise + 
                sigma_t * noise)
    return x_t_prev
```

测试得到结果为：

![](https://cdn.z.wiki/autoupload/20250520/CHJj/1000X200/Generate-image.gif)


* Latent Diffusion Model训练

## 参考
1、https://www.tonyduan.com/diffusion/index.html
2、https://arxiv.org/pdf/2006.11239
3、https://arxiv.org/pdf/1503.03585
4、https://arxiv.org/pdf/2208.11970
5、https://arxiv.org/pdf/2102.09672

[^1]: https://arxiv.org/pdf/2102.09672
[^2]: https://arxiv.org/abs/2212.09748
[^3]:https://github.com/facebookresearch/DiT
[^4]:https://huggingface.co/docs/diffusers/en/tutorials/basic_training