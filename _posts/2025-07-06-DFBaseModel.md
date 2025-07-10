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
description: 本文重点对比Stable Diffusion SD 1.5与SDXL基座模型，分析CLIP编码器差异（SDXL采用OpenCLIP-ViT/G与CLIP-ViT/L拼接，文本理解能力更强）、图像输出维度（SDXL默认1024x1024并使用refiner模型）及SDXL分辨率与裁剪优化策略；同时介绍Adapters中的ControlNet（通过zero-convolution指导输出）和T2I-Adapter（特征相加控制生成）。
---

## Stable Diffusion系列
主要介绍SD以及SDXL两类模型，但是SD迭代版本挺多的（从1.2到3.5）因此本文主要重点介绍SD 1.5以及SDXL两个基座模型，以及两者之间的对比差异。
### SDv1.5 vs SDXL[^1]
> **SDv1.5**: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
> **SDXL**:https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

两者模型详细的模型结构：[SDv1.5--SDXL模型结构图](../Dio.drawio)，其中具体模型参数的对比如下：
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

## Adapters
> https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference

此类方法是在完备的 DF 权重基础上，额外添加一个“插件”，保持原有权重不变。我只需修改这个插件，就可以让模型生成不同风格的图像。下面介绍的 ControlNet 和 T2I-Adapter，可以理解为在原始模型之外新增一个“生成条件”，通过修改这一条件即可灵活控制模型生成各种风格或满足不同需求的图像。

### ControlNet[^2]
> https://github.com/lllyasviel/ControlNet
> 建议直接阅读：[https://github.com/lllyasviel/ControlNet/discussions/categories/announcements](https://github.com/lllyasviel/ControlNet/discussions/categories/announcements) 来了解更加多细节

![](https://s2.loli.net/2025/07/09/Tfji2LMv15tgr6d.webp)

ControlNet的处理思路就很简单，再左图中模型的处理过程就是直接通过：$y=f(x;\theta)$来生成图像，但是在ControlNet里面会将我们最开始的网络结构复制然后通过在其前后引入一个**zero-convolution**层来“指导”（$Z$）模型的输出也就是说将上面的生成过程变为：$y=f(x;\theta)+Z(f(x+Z(c;\theta_{z_1});\theta);\theta_{Z_2})$。通过冻结最初的模型的权重保持不变，保留了Stable Diffusion模型原本的能力；与此同时，使用额外数据对“可训练”副本进行微调，学习我们想要添加的条件。因此在最后我们的SD模型中就是如下一个结构：

![](https://s2.loli.net/2025/07/09/uVNAEnleRMJ6p4v.webp)

在论文里面作者给出一个实际的测试效果可以很容易理解里面条件c（条件 𝑐就是提供给模型的显式结构引导信息，**用于在生成过程中精确控制图像的空间结构或布局**，一般来说可以是草图、分割图等）到底是一个什么东西，比如说就是直接给出一个“线稿”然后模型来输出图像。

![](https://s2.loli.net/2025/07/09/rkWH3o1MOaNs6pg.webp)

> **补充-1**：为什么使用上面这种结构
> 在[github](https://github.com/lllyasviel/ControlNet/discussions/188)上作者讨论了为什么要使用上面这种结构而非直接使用mlp等（作者给出了很多测试图像），最后总结就是：**这种结构好**
> **补充-2**：使用0卷积层会不会导致模型无法优化问题？
> 不会，因为对于神经网络结构大多都是：$y=wx+b$计算梯度过程中即使 $w=0$但是里面的 $x≠0$模型的参数还是可以被优化的

### T2I-Adapter[^3]
> https://github.com/TencentARC/T2I-Adapter

![image.png](https://s2.loli.net/2025/07/09/gZLDtFSGr25kCwa.webp)

T2I的处理思路也比较简单（T2I-Adap 4 ter Details里面其实就写的很明白了），对于输入的条件图片（比如说边缘图像）:512x512，首先通过 pixel unshuffle进行下采样将图像分辨率改为：64x64而后通过一层卷积+两层残差连接，输出得到特征 $F_c$之后将其与对应的encoder结构进行相加：$F_{enc}+ F_c$，当然T2I也支持多个条件（直接通过加权组合就行）


### ControlNet的代码操作
> Code: [https://github.com/shangxiaaabb/ProjectCode/tree/main/code/Python/DFModelCode/training_controlnet](https://github.com/shangxiaaabb/ProjectCode/tree/main/code/Python/DFModelCode/training_controlnet)

**首先**，简单了解一个ControlNet数据集格式，一般来说（）数据主要是三部分组成：1、image（可以理解为生成的图像）；2、condiction_image（可以理解为输入ControlNet里面的条件 $c$）；3、text。比如说以[raulc0399/open_pose_controlnet](https://huggingface.co/datasets/raulc0399/open_pose_controlnet)为例
![](https://s2.loli.net/2025/07/10/ywau8kjIlE1L7er.png)

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

## 参考
[^1]:https://arxiv.org/pdf/2307.01952
[^2]:https://arxiv.org/pdf/2302.05543
[^3]:https://arxiv.org/pdf/2302.08453