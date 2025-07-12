---
layout: mypost
title: 深入浅出了解生成模型-6：常用基础模型与 Adapters等解析
categories: 生成模型
extMath: true
images: true
address: 长沙🌷
show_footer_image: true
tags:
- 生成模型
- diffusion model
- ControlNet
- T2I-Adapter
- SD
- SDVL
show: true
description: 对比Stable Diffusion SD 1.5与SDXL模型差异，SDXL采用双CLIP编码器（OpenCLIP-ViT/G+CLIP-ViT/L）提升文本理解，默认1024x1024分辨率并优化处理；介绍ControlNet（空间结构控制）、T2I-Adapter、DreamBooth（解决语言偏离）等Adapters，实现风格迁移与高效生成。
---

## 基座扩散模型
主要介绍SD以及SDXL两类模型，但是SD迭代版本挺多的（从1.2到3.5）因此本文主要重点介绍SD 1.5以及SDXL两个基座模型，以及两者之间的对比差异，除此之外还有许多闭源的扩散模型比如说Imagen、DALE等。

### SDv1.5 vs SDXL[^1]
> **SDv1.5**: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
> **SDXL**:https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

两者模型详细的模型结构：[SDv1.5--SDXL模型结构图](https://1drv.ms/u/c/667854cf645e8766/ESgZEHNEn3RJsKY0t1KQAgABYKHDhQtutJztw6OhEt9DPg?e=5SqEro)，其中具体模型参数的对比如下：
**1、CLIP编码器区别**：
在SD1.5中选择的是**CLIP-ViT/L**（得到的维度为：768）而在SDXL中选择的是两个CLIP文本编码器：**OpenCLIP-ViT/G**（得到的维度为：1280）以及**CLIP-ViT/L**（得到维度为：768）在代码中对于两个文本通过编码器处理之后SDXL直接通过cat方式拼接：`prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)` 也就是说最后得到的维度为：[..,..,1280+768]。最后效果很明显：**SDXL对于文本的理解能力大于SD1.5**
**2、图像输出维度区别**：
再SD1.5中的默认输出是：512x512而再SDXL中的默认输出是：1024x1024，如果希望将SD1.5生成的图像处理为1024x1024可以直接通过超分算法来进行处理，除此之外在SDXL中还会使用一个refiner模型（和Unet的结构相似）来强化base模型（Unet）生成的效果。
**3、SDXL论文中的技术细节**：
* 1、**图像分辨率优化策略**。

数据集中图像的尺寸图像利用率问题（选择512x512舍弃256x256就会导致图像大量被舍弃）如果通过超分辨率算法将图像就行扩展会放大伪影，这些伪影可能会泄漏到最终的模型输出中，例如，导致样本模糊。（The second method, on the other hand, usually introduces upscaling artifacts which may leak into the final model outputs, causing, for example, blurry samples.）作者做法是：**训练阶段**直接将原始图像的分辨率 $c=(h_{org},w_{org})$作为一个条件，通过傅里叶特征编码而后加入到time embedding中，**推理阶段**直接将分辨率作为一个条件就行嵌入，进而实现：**当输入低分辨率条件时，生成的图像较模糊；在不断增大分辨率条件时，生成的图像质量不断提升。**
![image.png](https://s2.loli.net/2025/07/09/pMcLmdHThu2CnNx.webp)

* 2、**图像裁剪优化策略**

直接统一采样裁剪坐标top和cleft（分别指定从左上角沿高度和宽度轴裁剪的像素数量的整数），并通过傅里叶特征嵌入将它们作为调节参数输入模型，类似于上面描述的尺寸调节
第1，2点代码中的处理方式为：
```python
def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)

    passed_add_embed_dim = (
        self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
    )
    expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
    ...
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids
```

> **推荐阅读**：
> 1、[SDv1.5-SDXL-SD3生成效果对比](https://www.magicflow.ai/showcase/sd3-sdxl-sd1.5)

### Imagen系列
> https://imagen.research.google/
> https://deepmind.google/models/imagen/
> 非官方实现：https://github.com/lucidrains/imagen-pytorch
> 类似Github，通过3阶段生成：https://github.com/deep-floyd/IF

Imagen[^6]论文中主要提出：1、纯文本语料库上预训练的通用大型语言模型（例如[T5](https://huggingface.co/collections/google/t5-release-65005e7c520f8d7b4d037918)、CLIP、BERT等）在编码图像合成的文本方面非常有效：在Imagen中增加语言模型的大小比增加图像扩散模型的大小更能提高样本保真度和Imagetext对齐。
![](https://s2.loli.net/2025/07/12/lCFNWwDmgGnZueE.webp)

2、通过提高classifier-free guidance weight（$\epsilon(z,c)=w\epsilon(z,c)+ (1-w)\epsilon(z)$ 也就是其中的参数 $w$）可以提高image-text之间的对齐，但会损害图像逼真度，产生高度饱和不自然的图像（论文里面给出的分析是：每个时间步中预测和正式的x都会限定在 $[-1,1]$这个范围但是较大的 $w$可能导致超出这个范围），论文里面做法就是提出 **动态调整方法**：在每个采样步骤中，我们将s设置为 $x_0^t$中的某个百分位绝对像素值，如果s>1，则我们将 $x_0^t$阈值设置为范围 $[-s,s]$，然后除以s。
![](https://s2.loli.net/2025/07/12/jAEBS7I1Ob6DPal.webp)

3、和上面SD模型差异比较大的一点就是，在imagen中直接使用多阶段生成策略，模型先生成64x64图像再去通过超分辨率扩散模型去生成256x256以及1024x1024的图像，在此过程中作者提到使用noise conditioning augmentation（NCA）策略（对输入的文本编码后再去添加随机噪声）
![](https://s2.loli.net/2025/07/12/HJm96oPr2AlXICs.webp)


## Adapters
> https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference

此类方法是在完备的 DF 权重基础上，额外添加一个“插件”，保持原有权重不变。我只需修改这个插件，就可以让模型生成不同风格的图像。可以理解为在原始模型之外新增一个“生成条件”，通过修改这一条件即可灵活控制模型生成各种风格或满足不同需求的图像。

### ControlNet
> https://github.com/lllyasviel/ControlNet
> 建议直接阅读：[https://github.com/lllyasviel/ControlNet/discussions/categories/announcements](https://github.com/lllyasviel/ControlNet/discussions/categories/announcements) 来了解更加多细节

![](https://s2.loli.net/2025/07/09/Tfji2LMv15tgr6d.webp)

ControlNet[^2]的处理思路就很简单，再左图中模型的处理过程就是直接通过：$y=f(x;\theta)$来生成图像，但是在ControlNet里面会将我们最开始的网络结构复制然后通过在其前后引入一个**zero-convolution**层来“指导”（$Z$）模型的输出也就是说将上面的生成过程变为：$y=f(x;\theta)+Z(f(x+Z(c;\theta_{z_1});\theta);\theta_{Z_2})$。通过冻结最初的模型的权重保持不变，保留了Stable Diffusion模型原本的能力；与此同时，使用额外数据对“可训练”副本进行微调，学习我们想要添加的条件。因此在最后我们的SD模型中就是如下一个结构：

![](https://s2.loli.net/2025/07/09/uVNAEnleRMJ6p4v.webp)

在论文里面作者给出一个实际的测试效果可以很容易理解里面条件c（条件 𝑐就是提供给模型的显式结构引导信息，**用于在生成过程中精确控制图像的空间结构或布局**，一般来说可以是草图、分割图等）到底是一个什么东西，比如说就是直接给出一个“线稿”然后模型来输出图像。

![](https://s2.loli.net/2025/07/09/rkWH3o1MOaNs6pg.webp)

> **补充-1**：为什么使用上面这种结构
> 在[github](https://github.com/lllyasviel/ControlNet/discussions/188)上作者讨论了为什么要使用上面这种结构而非直接使用mlp等（作者给出了很多测试图像），最后总结就是：**这种结构好**
> **补充-2**：使用0卷积层会不会导致模型无法优化问题？
> 不会，因为对于神经网络结构大多都是：$y=wx+b$计算梯度过程中即使 $w=0$但是里面的 $x≠0$模型的参数还是可以被优化的


#### ControlNet的代码操作
> Code: [https://github.com/shangxiaaabb/ProjectCode/tree/main/code/Python/DFModelCode/training_controlnet](https://github.com/shangxiaaabb/ProjectCode/tree/main/code/Python/DFModelCode/training_controlnet)

**首先**，简单了解一个ControlNet数据集格式，一般来说（）数据主要是三部分组成：1、image（可以理解为生成的图像）；2、condiction_image（可以理解为输入ControlNet里面的条件 $c$）；3、text。比如说以[raulc0399/open_pose_controlnet](https://huggingface.co/datasets/raulc0399/open_pose_controlnet)为例
![](https://s2.loli.net/2025/07/12/nphNm3OIebFGazr.webp)

**模型加载**，一般来说扩散模型就只需要加载如下几个：`DDPMScheduler`、`AutoencoderKL`（vae模型）、`UNet2DConditionModel`（不一定加载条件Unet模型），除此之外在ControlNet中还需要加载一个`ControlNetModel`。对于`ControlNetModel`中代码大致结构为，代码中通过`self.controlnet_down_blocks`来存储ControlNet的下采样模块（**初始化为0的卷积层**）。`self.down_blocks`用来存储ControlNet中复制的Unet的下采样层。在`forward`中对于输入的样本（`sample`）首先通过 `self.down_blocks`逐层处理叠加到 `down_block_res_samples`中，而后就是直接将得到结果再去通过 `self.controlnet_down_blocks`每层进行处理，最后返回下采样的每层结果以及中间层处理结果：`down_block_res_samples`，`mid_block_res_sample`

```python
class ControlNetModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(...):
        ...
        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])
        # 封装下采样过程（对应上面模型右侧结构）
        controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)
        for i, down_block_type in enumerate(down_block_types):
            # down_block_types就是Unet里面下采样的每一个模块比如说：CrossAttnDownBlock2D
            ...
            down_block = get_down_block(down_block_type) # 通过 get_down_block 获取uet下采样的模块
            self.down_blocks.append(down_block)
            for _ in range(layers_per_block):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)
    @classmethod
    def from_unet(cls, unet,...):
        ...
        # 通过cls实例化的类本身ControlNetModel
        controlnet = cls(...)
        if load_weights_from_unet:
            # 将各类权重加载到 controlnet 中
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            ...

        return controlnet
    def forward(...):
        ...
        # 时间编码
        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb, timestep_cond)
        if self.class_embedding is not None:
            ...
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        # 对条件进行编码
        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)
            elif self.config.addition_embed_type == "text_time":
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)
        emb = emb + aug_emb if aug_emb is not None else emb       

        sample = self.conv_in(sample)
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample = sample + controlnet_cond

        # 下采样处理
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if ...
                ...
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples
        # 中间层处理
        ...
        # 将输出后的内容去和0卷积进行叠加
        controlnet_down_block_res_samples = ()
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)
        ...
        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)
        ...
```

**模型训练**，训练过程和DF训练差异不大。将图像通过VAE处理、产生噪声、时间步、将噪声添加到（VAE处理之后的）图像中，而后通过 `controlnet`得到每层下采样的结果以及中间层结果：`down_block_res_samples, mid_block_res_sample = controlnet(...)`而后将这两部分结果再去通过unet处理
```python
model_pred = unet(
    noisy_latents,
    timesteps,
    encoder_hidden_states=encoder_hidden_states,
    down_block_additional_residuals=[
        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
    ],
    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
    return_dict=False,
)[0]
```

后续就是计算loss等处理

**模型验证**，直接就是使用`StableDiffusionControlNetPipeline`来处理了


### T2I-Adapter
> https://github.com/TencentARC/T2I-Adapter

![image.png](https://s2.loli.net/2025/07/09/gZLDtFSGr25kCwa.webp)

T2I[^3]的处理思路也比较简单（T2I-Adap 4 ter Details里面其实就写的很明白了），对于输入的条件图片（比如说边缘图像）:512x512，首先通过 pixel unshuffle进行下采样将图像分辨率改为：64x64而后通过一层卷积+两层残差连接，输出得到特征 $F_c$之后将其与对应的encoder结构进行相加：$F_{enc}+ F_c$，当然T2I也支持多个条件（直接通过加权组合就行）

### DreamBooth
> https://huggingface.co/docs/diffusers/v0.34.0/using-diffusers/dreambooth

论文[^4]里面主要出发点就是：1、解决**language drif**（语言偏离问题）：指的是模型通过后训练（微调等处理之后）模型丧失了对某些语义特征的感知，就比如说扩散模型里面，模型通过不断微调可能就不知道“狗”是什么从而导致模型生成错误。2、高效的生成需要的对象，不会产生：生成错误、细节丢失问题，比如说下面图像中的问题：
![](https://s2.loli.net/2025/07/12/mRaHPOtC23li9Fn.webp)

为了实现图像的“高效迁移”，作者直接将图像（比如说我们需要风格化的图片）作为一个特殊的标记，也就是论文里面提到的 `a [identifier] [class noun]`（其中class noun为类别比如所狗，identifier就是一个特殊的标记），在prompt中加入类别，通过利用预训练模型中关于该类别物品的先验知识，并将先验知识与特殊标记符相关信息进行融合，这样就可以在不同场景下生成不同姿势的目标物体。就比如下面的 `fine-tuning`过程通过几张图片让模型学习到 *特殊的狗*，然后再推理阶段模型可以利用这个 *特殊的狗*去生成新的动作。

![](https://s2.loli.net/2025/07/12/hYM1VdykDxALrGo.webp)

再论文里面作者设计如下的Class-specific Prior Preservation Loss（参考stackexchange）[^5]：

$$\begin{aligned}
 & \mathbb{E}_{x,c,\epsilon,t}\left[\|\epsilon-\varepsilon_{\theta}(z_{t},t,c)\|_{2}^{2}+\lambda\|\epsilon^{\prime}-\epsilon_{pr}(z_{t^{\prime}}^{\prime},t^{\prime},c_{pr})\|_{2}^{2}\right]
\end{aligned}$$

上面损失函数中后面一部分就是我们的先验损失，比如说$c+{pr}$就是对 "a dog"进行编码然后计算生成损失。在代码中：

```python
if args.with_prior_preservation:
    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
    target, target_prior = torch.chunk(target, 2, dim=0)
    # Compute instance loss
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    # Compute prior loss
    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
    # Add the prior loss to the instance loss.
    loss = loss + args.prior_loss_weight * prior_loss
else:
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
```

## 总结
对于不同的扩散（基座）模型（SD1.5、SDXL）等大部分都是采用Unet结构，当然也有采用Dit的，这两个模型之间的差异主要在于后者会多一个clip编码器再文本语义上比前者更加有优势。对于adapter而言，可以直接理解为再SD的基础上去使用“风格插件”，这个插件不去对SD模型进行训练（从而实现对参数的减小），对于ControNet就是直接对Unet的下采样所有的模块（前后）都加一个zero-conv而后将结果再去嵌入到下采用中，而T2I-Adapter则是去对条件进行编码而后嵌入到SD模型（上采用模块）中。对于deramboth就是直接通过设计的Class-specific Prior Preservation Loss来实现生成特例的风格化迁移

## 参考
[^1]:[https://arxiv.org/pdf/2307.01952](https://arxiv.org/pdf/2307.01952)
[^2]:[https://arxiv.org/pdf/2302.05543](https://arxiv.org/pdf/2302.05543)
[^3]:[https://arxiv.org/pdf/2302.08453](https://arxiv.org/pdf/2302.08453)
[^4]:[https://arxiv.org/pdf/2208.12242](https://arxiv.org/pdf/2208.12242)
[^5]:https://stats.stackexchange.com/questions/601782/how-to-rewrite-dreambooth-loss-in-terms-of-epsilon-prediction
[^6]: [https://arxiv.org/pdf/2205.11487](https://arxiv.org/pdf/2205.11487)