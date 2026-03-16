---
layout: mypost
title: 深入浅出了解生成模型-6：常用图像生成基座模型与 Adapters等解析
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
- SDXL
- Flux
- Z-image
show: true
stickie: true
description: 本文主要介绍基于Unet和Dit框架的基座扩散模型，重点对比SD1.5与SDXL的核心差异，包括CLIP编码器（SDXL采用双编码器拼接提升文本理解能力）、图像输出维度（SDXL默认1024x1024优于SD1.5的512x512）及技术优化策略。还涵盖Imagen的多阶段生成与动态调整方法，Dit模型的patch切分与adaLN模块，Hunyuan-DiT的双文本编码器与旋转位置编码，FLUX.1的VAE通道优化与旋转位置编码，以及SD3的三文本编码器与MM-Dit架构。同时涉及VAE模型重构表现对比、guidance_rescale参数对生成效果的影响，和Adapters技术如ControlNet（零卷积层条件控制）、DreamBooth（样本微调与类别先验损失）等插件式模型调整方法，旨在全面解析不同扩散模型的结构特性与应用技术。
---
## 基座扩散模型
主要介绍基于Unet以及基于Dit框架的基座扩散模型以及部分GAN和VAE模型，其中SD迭代版本挺多的（从1.2到3.5）因此本文主要重点介绍SD 1.5以及SDXL两个基座模型，以及两者之间的对比差异，除此之外还有许多闭源的扩散模型比如说Imagen、DALE等。对于Dit基座模型主要介绍：Hunyuan-DiT、FLUX.1等。对于各类模型评分网站（模型评分仁者见仁智者见智，特别是此类生成模型视觉图像生成是一个很主观的过程，同一张图片不同人视觉感官都是不同的）：[https://lmarena.ai/leaderboard](https://lmarena.ai/leaderboard)
### SD系列模型
#### SD1.5 vs SDXL
> **SDv1.5**: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
> **SDXL**[^1]:https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

两者模型详细的模型结构：[SDv1.5--SDXL模型结构图](https://drive.google.com/file/d/1-fiJKHVANsghaviyIha8MbbC2Yz4ld6L/view?usp=sharing)，其中具体模型参数的对比如下：
**1、CLIP编码器区别**：
在SD1.5中选择的是**CLIP-ViT/L**（得到的维度为：768）而在SDXL中选择的是两个CLIP文本编码器：**OpenCLIP-ViT/G**（得到的维度为：1280）以及**CLIP-ViT/L**（得到维度为：768）在代码中对于两个文本通过编码器处理之后SDXL直接通过cat方式拼接：`prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)` 也就是说最后得到的维度为：[..,..,1280+768]。最后效果很明显：**SDXL对于文本的理解能力大于SD1.5**
**2、图像输出维度区别**：
再SD1.5中的默认输出是：512x512而再SDXL中的默认输出是：1024x1024，如果希望将SD1.5生成的图像处理为1024x1024可以直接通过超分算法来进行处理，除此之外在SDXL中还会使用一个refiner模型（和Unet的结构相似）来强化base模型（Unet）生成的效果。
**3、SDXL论文中的技术细节**：
* 1、**图像分辨率优化策略**。

数据集中图像的尺寸图像利用率问题（选择512x512舍弃256x256就会导致图像大量被舍弃）如果通过超分辨率算法将图像就行扩展会放大伪影，这些伪影可能会泄漏到最终的模型输出中，例如，导致样本模糊。（The second method, on the other hand, usually introduces upscaling artifacts which may leak into the final model outputs, causing, for example, blurry samples.）作者做法是：**训练阶段**直接将原始图像的分辨率 $c=(h_{org},w_{org})$作为一个条件，通过傅里叶特征编码而后加入到time embedding中，**推理阶段**直接将分辨率作为一个条件就行嵌入，进而实现：**当输入低分辨率条件时，生成的图像较模糊；在不断增大分辨率条件时，生成的图像质量不断提升。**
![image.png](https://s2.loli.net/2025/07/09/pMcLmdHThu2CnNx.webp)
* 2、**图像裁剪优化策略**

直接统一采样裁剪坐标top和cleft（分别指定从左上角沿高度和宽度轴裁剪的像素数量的整数），并通过傅里叶特征嵌入将它们作为调节参数输入模型，类似于上面描述的尺寸调节。第1，2点代码中的处理方式为：

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

#### SD3模型
> SD3的diffusers官方文档：[StableDiffusion3Pipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_3#diffusers.StableDiffusion3Pipeline)

https://zhouyifan.net/2024/09/03/20240809-flux1/
SD3[^12]、FLUX对于这几组模型的前世今生不做介绍，主要了解其模型结构以及论文里面所涉及到到的一些知识点。首先介绍SD3模型在模型改进上[^16]：1、改变训练时噪声采样方法；2、将一维位置编码改成二维位置编码；3、提升 VAE 隐空间通道数（作者实验发现最开始VAE会将模型**下采样8倍数并且处理通道为4的空间**，也就是说 $512 \times 512 \times 3 \rightarrow 64\times 64 \times 4$，不过在 **SD3**中将通道数由**4改为16**）；4、对注意力 QK 做归一化以确保高分辨率下训练稳定。
![](https://s2.loli.net/2025/09/01/R5HI3yLPXBEbtzQ.webp)
其中SD3模型的整体框架如上所述:
**1、文本编码器处理**（[代码](https://github.com/huggingface/diffusers/blob/0f252be0ed42006c125ef4429156cb13ae6c1d60/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L972)），在text encoder上SD3使用三个文本编码器：`clip-vit-large-patch14`、 `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` 、 `t5-v1_1-xxl` ，对于这3个文本编码器对于文本的处理过程为：就像SDXL中一样首先3个编码器分别都去对文本进行编码，首先对于两个[CLIP的文本编码](https://github.com/huggingface/diffusers/blob/0f252be0ed42006c125ef4429156cb13ae6c1d60/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L289)处理过程为直接通过CLIP进行 `prompt_embeds = text_encoder(text_input_ids.to(device)...)` 而后去选择 `prompt_embeds.hidden_states[-(clip_skip + 2)]`（默认条件下 `clip_skip=None`也就是**直接选择倒数第二层**）那么最后得到文本编码的维度为：`torch.Size([1, 77, 768]) torch.Size([1, 77, 1280])` 而[T5的encoder](https://github.com/huggingface/diffusers/blob/0f252be0ed42006c125ef4429156cb13ae6c1d60/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L233)就比较检查直接通过encoder进行编码，那么其编码维度为：`torch.Size([1, 256, 4096])`，这样一来就会得到3组的文编码，对于CLIP的编码结果直接通过 `clip_prompt_embeds=torch.cat([prompt_embed, prompt_2_embed], dim=-1)` 即可，在将得到后的 `clip_prompt_embeds`结果再去和T5的编码结果进行拼接之前会首先 `clip_prompt_embeds=torch.nn.functional.pad(clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]))` 而后将T5的文本内容和 `clip_prompt_embeds`进行合并 `prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)`。由于使用T5模型导致模型的参数比较大进导致模型的显存占用过大（2080Ti等GPU上轻量化的部署推理SD 3模型，可以只使用CLIP ViT-L + OpenCLIP ViT-bigG的特征，此时需要**将T5-XXL的特征设置为zero**（不加载）[^14]），选择**不去使用T5模型会对模型对于文本的理解能力有所降低**。
![image.png](https://s2.loli.net/2025/09/01/RdQCOmXeMfwUYsh.webp)

> SD3使用T5-XXL模型。这使得以少于24GB的VRAM在GPU上运行模型，即使使用FP16精度。因此如果需要使用就需要：1、将部分模型[下放到CPU上](https://github.com/huggingface/diffusers/blob/0f252be0ed42006c125ef4429156cb13ae6c1d60/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L186)；2、直接取消T5的使用（`StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",text_encoder_3=None,tokenizer_3=None,torch_dtype=torch.float16)`）。
> 文本编码过程：1、CLIP编码分别得到：[1, 77, 768]和[1, 77, 1280]；2、T5编码得到：[1, 256, 4096]；3、CLIP文本编码拼接：[1, 77, 2048]在去将其通过pad填充到和T5一致得到最后CLIP编码器维度为：**[1, 77, 4096]**；4、最后文本编码维度：`[1, 333, 4096]`

**2、Flow Matching模式**（[原理](https://www.big-yellow-j.top/posts/2025/07/06/DFscheduler.html)）；
**3、MM-Dit模型架构**（[代码](https://github.com/huggingface/diffusers/blob/d03240801f2ac2b4d1f49584c1c5628b98583f6a/src/diffusers/models/transformers/transformer_sd3.py#L80)）：观察上面过程，扩散模型输入无非就是3个内容：1、时间步（$y$）；2、加噪处理的图像（$x$）；3、文本编码（$c$）。首先对于 **时间步**而言处理过程为：直接通过 Sin位置编码然后去和CLIP（两个合并的）进行组合即可对于另外两个部分直接通过[代码](https://github.com/huggingface/diffusers/blob/d03240801f2ac2b4d1f49584c1c5628b98583f6a/src/diffusers/models/transformers/transformer_sd3.py#L80)进行理解：

```python
def forward(
    self,
    hidden_states: torch.Tensor, # 加噪声的图片 (batch size, channel, height, width)
    encoder_hidden_states: torch.Tensor = None, # 条件编码比如说：文本prompt (batch size, sequence_len, embed_dims)
    pooled_projections: torch.Tensor = None, # 池化后的条件编码 (batch size, embed_dims)
    timestep: torch.LongTensor = None, # 时间步编码
    block_controlnet_hidden_states: List = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    skip_layers: Optional[List[int]] = None,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    ...
    height, width = hidden_states.shape[-2:]
    # Step-1 
    hidden_states = self.pos_embed(hidden_states) # 直接使用 2D的位置编码
    temb = self.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states) # 一层线性映射
    ...
    # Step-2
    for index_block, block in enumerate(self.transformer_blocks):
        is_skip = True if skip_layers is not None and index_block in skip_layers else False
        if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:
            ...
        elif not is_skip:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        ...
    # Step-3
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)
    patch_size = self.config.patch_size
    height = height // patch_size
    width = width // patch_size

    hidden_states = hidden_states.reshape(
        shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
    )
    ...
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
```

**Step-1**：首先去将图像 $x$使用2D 正弦-余弦位置编码进行处理，对于时间步直接sin位置编码，对于条件（文本prompt等）直接通过一层线性编码处理。
**Step-2**：然后就是直接去计算Attention：`encoder_hidden_states, hidden_states = block(hidden_states=hidden_states,encoder_hidden_states=encoder_hidden_states,temb=temb,joint_attention_kwargs=joint_attention_kwargs,)`，对于这个[block](https://github.com/huggingface/diffusers/blob/d03240801f2ac2b4d1f49584c1c5628b98583f6a/src/diffusers/models/attention.py#L570)的设计过程为：
![](https://s2.loli.net/2025/09/01/gLkSbrt9vfyQ5uJ.webp)

```python
def forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
):
    joint_attention_kwargs = joint_attention_kwargs or {}
    # Aeetntion Step-1
    if self.use_dual_attention:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
            hidden_states, emb=temb
        )
    else:
        ...

    if self.context_pre_only:
        ...
    else:
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
    # Attention Step-2
    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        **joint_attention_kwargs,
    )
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output
    ...
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    if self._chunk_size is not None:
        ...
    else:
        ff_output = self.ff(norm_hidden_states)
    ff_output = gate_mlp.unsqueeze(1) * ff_output
    hidden_states = hidden_states + ff_output
    if self.context_pre_only:
        ...

    return encoder_hidden_states, hidden_states
```

计算注意力过程中，首先 **Attention Step-1**：正则化处理（正如上面[Dit中](https://www.big-yellow-j.top/posts/2025/07/06/DFBaseModel.html#:~:text=%E5%9C%A8layernorm%E4%B8%AD%E4%B8%80%E8%88%AC%E5%BD%92%E4%B8%80%E5%8C%96%E5%A4%84%E7%90%86%E6%96%B9%E5%BC%8F%E4%B8%BA)的一样将条件拆分为几个参数，观察SD3图中的MMDit设计，会将 **加噪声处理的图片** 和 **条件编码**都去（处理方式相同）通过 “正则化”，在SD3中处理方式为，直接 `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = emb.chunk(9, dim=1)` 拆分之后去通过 `LayerNorm`处理之后得到 `norm_hidden_states` 而后在去计算 `norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]`）然后后面处理过程就比较简单和上面的流程图是一样的。
这样一来一个MMDit block就会返回两部分结果 `encoder_hidden_states`, `hidden_states`（区**别Dit之间在于，MMDit是将image和text两种模态之间的信息进行融合二Dit只是使用到imgae一种模态**）
**Step-3**就比较简单就是一些norm等处理。
**总的来说**MMDiT Block 的输入主要有三部分：**时间步嵌入** $y$：通过一个 MLP 投影，得到一组参数，用于调节 Block 内的 LayerNorm / Attention / MLP（类似 FiLM conditioning）。**图像 token** $x$：由加噪图像 latent patch embedding 得到，并加上 2D 正弦余弦位置编码。**文本 token** $c$：来自文本编码器的输出，一般带有 1D 位置编码。**Block 内部机制**：将 $x$ 和 $c$ 拼接在一起，作为 Transformer 的输入序列。在自注意力层中，$x$ token 能和 $c$ token 交互，从而实现 跨模态融合。$y$（timestep embedding）通过投影提供额外的条件控制。

> **2D 正弦-余弦位置编码**
> ![](https://s2.loli.net/2025/09/01/lwZns5H9vTpeOgU.webp)
> 左侧为一般的位置编码方式，但是有一个缺点：生成的图像的分辨率是无法修改的。比如对于上图，假如采样时输入大小不是4x3，而是4x5，那么0号图块的下面就是5而不是4了，模型训练时学习到的图块之间的位置关系全部乱套，因此就通过2D位置去代表每一块的位置信息。

简单补充介绍一下Flux.1模型（主要是文生图模型），对于**FLUX.1模型而言其结构如下**
![](https://s2.loli.net/2025/09/01/WTD97u3eFiQwr4d.webp)
区别SD3模型在于，FLUX.1在文本编码器选择上**只使用了2个编码器**（CLIPTextModel、T5EncoderModel）并且FLUX.1 VAE架构依然继承了SD 3 VAE的**8倍下采样和输入通道数（16）**。在FLUX.1 VAE输出Latent特征，并在Latent特征输入扩散模型前，还进行了 `_pack_latents`操作，一下子将Latent**特征通道数提高到64（16 -> 64）**，换句话说，FLUX.1系列的扩散模型部分输入通道数为64，是SD 3的四倍。对于 `_pack_latents`做法是会将一个 $2\times 2$的像素去补充到通道中。

```python
def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents
```

除去改变text的编码器数量以及VAE的通道数量之外，FLUX.1还做了如下的改进：FLUX.1 没有做 Classifier-Free Guidance (CFG)（对于CFG一般做法就是直接去将“VAE压缩的图像信息变量复制两倍” `torch.cat([latents] * 2)`，文本就是直接将negative_prompt的编码补充到文本编码中 `torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)`）而是把指引强度 guidance 当成了一个和时刻 t 一样的约束信息，传入去噪模型 transformer 中。在transformer模型结构设计中，SD3是**直接对图像做图块化，再设置2D位置编码** `PatchEmbed`，在FLUX.1中使用的是 `FluxPosEmbed`（旋转位置编码）

```python
# SD3
self.pos_embed = PatchEmbed(height=sample_size,width=sample_size,patch_size=patch_size,in_channels=in_channels,)
embed_dim=self.inner_dim,pos_embed_max_size=pos_embed_max_size,  # hard-code for now.)
# FLUX.1
self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
```
### Imagen
> https://imagen.research.google/
> https://deepmind.google/models/imagen/
> 非官方实现：https://github.com/lucidrains/imagen-pytorch
> 类似Github，通过3阶段生成：https://github.com/deep-floyd/IF

Imagen[^6]论文中主要提出：1、纯文本语料库上预训练的通用大型语言模型（例如[T5](https://huggingface.co/collections/google/t5-release-65005e7c520f8d7b4d037918)、CLIP、BERT等）在编码图像合成的文本方面非常有效：在Imagen中增加语言模型的大小比增加图像扩散模型的大小更能提高样本保真度和Imagetext对齐。
![](https://s2.loli.net/2025/07/12/lCFNWwDmgGnZueE.webp)
2、通过提高classifier-free guidance weight（$\epsilon(z,c)=w\epsilon(z,c)+ (1-w)\epsilon(z)$ 也就是其中的参数 $w$）可以提高image-text之间的对齐，但会损害图像逼真度，产生高度饱和不自然的图像（论文里面给出的分析是：每个时间步中预测和正式的x都会限定在 $[-1,1]$这个范围但是较大的 $w$可能导致超出这个范围），论文里面做法就是提出 **动态调整方法**：在每个采样步骤中，我们将s设置为 $x_0^t$中的某个百分位绝对像素值，如果s>1，则我们将 $x_0^t$阈值设置为范围 $[-s,s]$，然后除以s。
![](https://s2.loli.net/2025/07/12/jAEBS7I1Ob6DPal.webp)
3、和上面SD模型差异比较大的一点就是，在imagen中直接使用多阶段生成策略，模型先生成64x64图像再去通过超分辨率扩散模型去生成256x256以及1024x1024的图像，在此过程中作者提到使用noise conditioning augmentation（NCA）策略（**对输入的文本编码后再去添加随机噪声**）
![](https://s2.loli.net/2025/07/12/HJm96oPr2AlXICs.webp)
### Dit
> https://github.com/facebookresearch/DiT

![](https://s2.loli.net/2025/07/15/CUisy5TPE24kKaH.webp)
Dit[^11]模型结构上，1、**模型输入**，将输入的image/latent切分为不同patch而后去对不同编码后的patch上去添加位置编码（直接使用的sin-cos位置编码），2、**时间步以及条件编码**，对于时间步t以及条件c的编码而后将两部分编码后的内容进行相加，在 `TimestepEmbedder`上处理方式是：直接通过**正弦时间步嵌入**方式而后将编码后的内容通过两层liner处理；在 `LabelEmbedder`处理方式上就比较简单直接通过 `nn.Embedding`进行编码处理。3、使用Adaptive layer norm（adaLN）block以及adaZero-Block（对有些参数初始化为0，就和lora中一样初始化AB为0，为了保证后续模型训练过程中的稳定）

> 在[layernorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)中一般归一化处理方式为：$\text{Norm}(x)=\gamma \frac{x-\mu}{\sqrt{\sigma^2+ \epsilon}}+\beta$ 其中有两个参数 $\gamma$ 和 $\beta$ 是固定的可学习参数（比如说直接通过 `nn.Parameter` 进行创建），在模型初始化时创建，并在训练过程中通过梯度下降优化。但是在 adaLN中则是直接通过 $\text{Norm}(x)=\gamma(c) \frac{x-\mu}{\sqrt{\sigma^2+ \epsilon}}+\beta(c)$ 通过输入的条件c进行学习的
> 总结Dit训练过程：首先将图片通过VAE进行编码，而后将编码后的内容patch化然后输入到Ditblock（就是通过一些Attention结构进行堆叠）中去预测模型的噪声

### Hunyuan-DiT
> https://huggingface.co/Tencent-Hunyuan/HunyuanDiT

腾讯的Hunyuan-DiT[^8]模型整体结构
![](https://s2.loli.net/2025/07/15/Hum9FCtPbV7do1B.webp)
整体框架不是很复杂，1、文本编码上直接通过结合两个编码器：CLIP、T5；2、VAE则是直接使用的SD1.5的；3、引入2维的旋转位置编码；4、在Dit结构上（图片VAE压缩而后去切分成不同patch），使用的是堆叠的注意力模块（在SD1.5中也是这种结构）self-attention+cross-attention（此部分输入文本）。论文里面做了改进措施：1、借鉴之前处理，计算attention之前首先进行norm处理（也就是将norm拿到attention前面）。
简短了解一下模型是如何做数据的：
![](https://s2.loli.net/2025/07/15/dJZETbyHB6SQPKI.webp)
### PixArt
> https://pixart-alpha.github.io/

华为诺亚方舟实验室提出的 $\text{PixArt}-\alpha$模型整体框架如下：
![](https://s2.loli.net/2025/07/15/cWTtLdONRPC9fnz.webp)
相比较Dit模型论文里面主要进行的改进如下：
1、**Cross-Attention layer**，在DiT block中加入了一个多头交叉注意力层，它位于自注意力层（上图中的Multi-Head Self
-Attention）和前馈层（Pointwise Feedforward）之间，使模型能够灵活地引入文本嵌入条件。此外，为了利用预训练权重，将交叉注意力层中的输出投影层初始化为零，作为恒等映射，保留了输入以供后续层使用。
2、AdaLN-single，在Dit中的adaptive normalization layers（adaLN）中部分参数（27%）没有起作用（在文生图任务中）将其替换为adaLN-single
### Flux系列模型
对于Flux系列模型截至到2026-02-14，主要用的比较多的就是Context以及较新的Klein模型（都是文生图+图生图模型），这里简单介绍一下这两个模型。
#### Flux.1-Context模型
对于Flux-Context[^19]模型在官方hf上开源了多种系列模型，主要流行的有3个版本：1、MAX和PRO：商用闭源；2、DEV：开源-12B，对于dev其官方权重地址为：[https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)除此之外相对使用比较多的是其通过nunchakus量化后的模型权重：[mit-han-lab/nunchaku-flux.1-kontext-dev](https://huggingface.co/mit-han-lab/nunchaku-flux.1-kontext-dev/tree/main)。简单介绍一下其模型原理如下：
![1087X533/image.png](https://test.fukit.cn/autoupload/f/vkB-Pqb1HNqojGbyEgL65tiO_OyvX7mIgxFBfDMDErs/20260214/iVPX/1087X533/image.png)
仅从上面模型架构很容易发现，在整体架构上还是**采用的是DiT模型框架核心差异在于将原图进行补充输入**，按照论文中的描述其生成过程为：$p(x\vert y,c)$：1、提示词文本输入，对应公式中的c，2、原图输入（这个主要是对于图像编辑任务而言需要的）对应公式中的y，如果是文生图那么$y=\phi$；3、噪声输入（将噪声转换为我的目标输入图像）。区别其他的图像编辑模型，一般而言会直接对原图进行vae编码而后添加噪声，在context上保持原图进行输入，对于这两部分输入都会去通过vae进行编码而后去对编码后的内容直接进行拼接，比如说：
```python
if image_reference_latents is not None:
    latent_model_input = torch.cat([latents, image_reference_latents], dim=1)
elif image_latents is not None:
    latent_model_input = torch.cat([latents, image_latents], dim=1)
```
其中latents、image_reference_latents、image_latents分别表示添加噪声的原始图像、参考图像vae编码、原始图像vae编码。进一步对于其模型结构如下：
![478X715](https://test.fukit.cn/autoupload/f/vkB-Pqb1HNqojGbyEgL65tiO_OyvX7mIgxFBfDMDErs/20260214/kf6f/478X715/image.png)
按照论文中第2章描述：从头训练的16通道VAE潜在空间里。它混合使用了19个double-stream（前期分离）和 38个single-stream（后期融合）的 Transformer 结构，用fused FFN 提效，用3D RoPE做位置编码。堆区其中提到的**两种stream方式**简单总结如下：所谓的double以及single无非就是对特征分别使用单独的linear去分别处理文本以及图像特征，而对于single则是将两部分特征拼接起来再用linear处理。**3D rope**：指的是在上述提到的2D位置编码上引入时间特征t。
#### Flux.2-Klein
对于Klein模型官方分别开源了：9B、4B两种版本以及所对应的蒸馏版本，其中4B可以被使用与商业用途。Klein模型在模型的推理以及显存占用上都是极大的改善
![1741X665](https://test.fukit.cn/autoupload/f/vkB-Pqb1HNqojGbyEgL65tiO_OyvX7mIgxFBfDMDErs/20260214/Xy5c/1741X665/image.png)
### Qwen image系列模型
#### Qwen image
> 官方blog：[https://qwenlm.github.io/zh/blog/qwen-image/](https://qwenlm.github.io/zh/blog/qwen-image/)
> Qwen Image图片编辑：[https://huggingface.co/Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit)
> Qwen Image：[https://huggingface.co/Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
> Qwen Image Lora微调8步生图：[https://huggingface.co/lightx2v/Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning)
> Qwen Image图片编辑int4量化版本：[https://huggingface.co/nunchaku-tech/nunchaku-qwen-image](https://huggingface.co/nunchaku-tech/nunchaku-qwen-image)，[代码](https://github.com/nunchaku-tech/nunchaku/blob/main/examples/v1/qwen-image.py)

Qwen image[^18]无论是多行文字、段落布局，还是中英文等不同语种，Qwen-Image都能以极高的保真度进行渲染，尤其在处理复杂的中文（logographic languages）方面，表现远超现有模型（不过目前：2025.08.29模型全权重加载的话一般设备很难使用，不过又量化版本可以尝试）模型整体结构：
![](https://s2.loli.net/2025/09/01/U7HqQcJxZ96SN3A.webp)
整体框架上还是MMDit结构和上面的SD3都是一致的，不过模型的改进在于：1、区别之前的都是使用CLIP模型去对齐图片-文本之间信息，在Qwen Image中则是直接使用**Qwen2.5-VL**；2、对于VAE模型则是直接使用**Wan-2.1-VAE**（不过选择冻结encoder部分只去训练decoder部分）；3、模型的结构还是使用MMDit结构，知识将位置编码方式改为**Multimodal Scalable RoPE (MSRoPE)**，位置编码方式
![](https://s2.loli.net/2025/09/01/QuEY2gWZFzUMlCK.webp)
大致框架了解之后细看他的数据是如何收集的以及后处理的：
![](https://s2.loli.net/2025/09/01/nzrOe2yaBpL5iwF.webp)
对于收集到数据之后，论文里面通过如下操作进行后处理：**1、阶段一过滤数据**：模型预训练是在256x256的图片上进行训练的，因此，过滤掉256x256以外的图片还有一些低质量图片等；**2、阶段二图片质量强化**：主要还是过滤一些低质量图片如亮度纹理等；
#### Z-Image 以及 Z-Image-Turbo
对于两种模型权重地址如下：1、[Z-Image HuggingFace](https://huggingface.co/Tongyi-MAI/Z-Image)；2、[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)对于两种模型对比如下：
![image.png](https://test.fukit.cn/autoupload/f/vkB-Pqb1HNqojGbyEgL65tiO_OyvX7mIgxFBfDMDErs/default/image.png)
Turbo是对Z-Image进行蒸馏加速版本。对于Z-Image其具体模型结构如下所示（在模型结构上差异不大，关键点在于数据构建过程见：[扩散模型数据构建](https://www.big-yellow-j.top/posts/2026/02/15/DataBuild.html)）
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260219131344750.png)
对于上述模型结构，文本编码器：Qwen3-4B；VAE使用Flux VAE；位置编码上使用3D Unified RoPE编码方式，除此之外对于图像编辑任务上使用SigLIP2去提取图像特征。除此之外再上述结构中Q/K-Norm指的是在计算attention之前提前去对Q/K进行归一化处理（一般的注意力计算是对Q/K/V在计算attention之后进行归一化亦或者提前对3者都进行归一化）除此之外其在训练阶段使用策略是：
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260219131448332.png)
在预训练（pre-training）阶段都是使用flow-matching方法进行训练，对于上述过程中首先在低质量预训练（Low-Resolution Pre-training）中使用256分辨率图像进行训练，在Omni Pre-training中则是直接进行更加复制训练任意分辨率、文生图、图生图等预训练。后续的强化学习中先使用DPO做offline alignment在使用GRPO做online alignment。除此之外使用Distribution Matching Distillation（DMD分布匹配蒸馏方式）去蒸馏模型得到Turbo模型
### FireRed-Image-Edit模型
![开源模型技术总结-3————FireRed（小红书）开源模型](https://www.big-yellow-j.top/posts/2026/03/09/FireRedModel.html)
### 基座模型对比效果分析
具体使用测试服务器为3090-24G进行测试，得到[不同模型在文生图效果对比分析结果](https://xcnsumle7uuf.feishu.cn/wiki/Tu74wPeCEirZ5XkY3WncaET1nod?from=from_copylink)
### VAE基座模型
对于VAE模型在之前的[博客](https://www.big-yellow-j.top/posts/2025/05/11/VAE.html)有介绍过具体的原理，这里主要就是介绍几个常见的VAE架构模型（使用过程中其实很少会去修改VAE架构，一般都是直接用SD自己使用的）所以就简单对比一下不同的VAE模型在图片重构上的表，主要是使用此[huggingface](https://huggingface.co/spaces/rizavelioglu/vae-comparison)上的进行比较（比较的数值越小越好，就数值而言 **CogView4-6B**效果最佳），下面结果为随便挑选的一个图片进行测试结果：

| 模型名称                  | 数值        | 时间(s)          |
| ------------------------- | ----------- | ---------------- |
| stable-diffusion-v1-4     | 2,059       | 0.5908           |
| eq-vae-ema                | 1,659       | 0.0831           |
| eq-sdxl-vae               | 1,200       | 0.0102           |
| sd-vae-ft-mse             | 1,204       | 0.0101           |
| sdxl-vae                  | 929         | 0.0105           |
| playground-v2.5           | 925         | 0.0096           |
| stable-diffusion-3-medium | 24          | 0.1027           |
| FLUX.1                    | 18          | 0.0412           |
| **CogView4-6B**     | **0** | **0.1265** |
| FLUX.1-Kontext            | 18          | 0.0098           |

### 不同模型参数对生成的影响
> https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#stable-diffusion-20

* 参数 `guidance_rescale`对于生成的影响

引导扩散模型（如 Classifier-Free Guidance，CFG）中，用于调整文本条件对生成图像的影响强度。它的核心作用是控制模型在生成过程中对文本提示的“服从程度”。公式上，CFG 调整预测噪声的方式如下：

$$
\epsilon = \epsilon_{\text{uncond}} + \text{guidance\_scale} \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})
$$

其中：1、$\epsilon_{\text{cond}}$：基于文本条件预测的噪声。2、$\epsilon_{\text{uncond}}$：无条件（无文本提示）预测的噪声。3、guidance_scale：决定条件噪声相对于无条件噪声的权重。得到最后测试结果如下（参数分别为[1.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]，`prompt = "A majestic lion standing on a mountain during golden hour, ultra-realistic, 8k"`， `negative_prompt = "blurry, distorted, low quality"`），容易发现数值越大文本对于图像的影响也就越大。
![tmp-CFG.png](https://s2.loli.net/2025/08/06/2jk18UISnqKdPZf.webp)
其中代码具体操作如下，从代码也很容易发现上面计算公式中的 uncond代表的就是我的negative_prompt，也就是说**CFG做的就是negative_prompt对生成的影响**：

```python
if self.do_classifier_free_guidance:
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
    add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
    add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
prompt_embeds = prompt_embeds.to(device)
```
## Adapters
> https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference

此类方法是在完备的 DF 权重基础上，额外添加一个“插件”，保持原有权重不变。我只需修改这个插件，就可以让模型生成不同风格的图像。可以理解为在原始模型之外新增一个“生成条件”，通过修改这一条件即可灵活控制模型生成各种风格或满足不同需求的图像。
### ControlNet
> https://github.com/lllyasviel/ControlNet
> 建议直接阅读：[https://github.com/lllyasviel/ControlNet/discussions/categories/announcements](https://github.com/lllyasviel/ControlNet/discussions/categories/announcements) 来了解更加多细节

![](https://s2.loli.net/2025/07/09/Tfji2LMv15tgr6d.webp)
ControlNet[^2]的处理思路就很简单，再左图中模型的处理过程就是直接通过：$y=f(x;\theta)$来生成图像，但是在ControlNet里面会 **将我们最开始的网络结构复制** 然后通过在其前后引入一个 **zero-convolution** 层来“指导”（ $Z$ ）模型的输出也就是说将上面的生成过程变为：$y=f(x;\theta)+Z(f(x+Z(c;\theta_{z_1});\theta);\theta_{Z_2})$。通过冻结最初的模型的权重保持不变，保留了Stable Diffusion模型原本的能力；与此同时，使用额外数据对“可训练”副本进行微调，学习我们想要添加的条件。因此在最后我们的SD模型中就是如下一个结构：
![](https://s2.loli.net/2025/07/09/uVNAEnleRMJ6p4v.webp)
在论文里面作者给出一个实际的测试效果可以很容易理解里面条件c（条件 𝑐就是提供给模型的显式结构引导信息，**用于在生成过程中精确控制图像的空间结构或布局**，一般来说可以是草图、分割图等）到底是一个什么东西，比如说就是直接给出一个“线稿”然后模型来输出图像。
![](https://s2.loli.net/2025/07/09/rkWH3o1MOaNs6pg.webp)

> **补充-1**：为什么使用上面这种结构
> 在[github](https://github.com/lllyasviel/ControlNet/discussions/188)上作者讨论了为什么要使用上面这种结构而非直接使用mlp等（作者给出了很多测试图像），最后总结就是：**这种结构好**
> **补充-2**：使用0卷积层会不会导致模型无法优化问题？
> 不会，因为对于神经网络结构大多都是：$y=wx+b$计算梯度过程中即使 $w=0$但是里面的 $x≠0$模型的参数还是可以被优化的

#### ControlNet代码操作

> Code: [https://github.com/shangxiaaabb/ProjectCode/tree/main/code/Python/DFModelCode/training_controlnet](https://github.com/shangxiaaabb/ProjectCode/tree/main/code/Python/DFModelCode/training_controlnet)

**首先**，简单了解一个ControlNet数据集格式，一般来说数据主要是三部分组成：1、image（可以理解为生成的图像）；2、condiction_image（可以理解为输入ControlNet里面的条件 $c$）；3、text。比如说以[raulc0399/open_pose_controlnet](https://huggingface.co/datasets/raulc0399/open_pose_controlnet)为例
![](https://s2.loli.net/2025/07/12/nphNm3OIebFGazr.webp)
**模型加载**，一般来说扩散模型就只需要加载如下几个：`DDPMScheduler`、`AutoencoderKL`（vae模型）、`UNet2DConditionModel`（不一定加载条件Unet模型），除此之外在ControlNet中还需要加载一个 `ControlNetModel`。对于 `ControlNetModel`中代码大致结构为，代码中通过 `self.controlnet_down_blocks`来存储ControlNet的下采样模块（**初始化为0的卷积层**）。`self.down_blocks`用来存储ControlNet中复制的Unet的下采样层。在 `forward`中对于输入的样本（`sample`）首先通过 `self.down_blocks`逐层处理叠加到 `down_block_res_samples`中，而后就是直接将得到结果再去通过 `self.controlnet_down_blocks`每层进行处理，最后返回下采样的每层结果以及中间层处理结果：`down_block_res_samples`，`mid_block_res_sample`

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

后续就是计算loss等处理。**模型验证**，直接就是使用 `StableDiffusionControlNetPipeline`来处理了。最后随机测试的部分例子（controlnet微调效果不是很好）：
![output.jpg](https://s2.loli.net/2025/07/22/SNfEiTVXpeZgOIP.webp)

### T2I-Adapter
> https://github.com/TencentARC/T2I-Adapter

![image.png](https://s2.loli.net/2025/07/09/gZLDtFSGr25kCwa.webp)
T2I[^3]的处理思路也比较简单（T2I-Adap 4 ter Details里面其实就写的很明白了），对于输入的条件图片（比如说边缘图像）:512x512，首先通过 pixel unshuffle进行下采样将图像分辨率改为：64x64而后通过一层卷积+两层残差连接，输出得到特征 $F_c$之后将其与对应的encoder结构进行相加：$F_{enc}+ F_c$，当然T2I也支持多个条件（直接通过加权组合就行）
### DreamBooth
> https://huggingface.co/docs/diffusers/v0.34.0/using-diffusers/dreambooth

DreamBooth 针对的使用场景是，期望生成同一个主体的多张不同图像， 就像照相馆一样，可以为同一个人或者物体照多张不同背景、不同姿态、不同服装的照片（和ControlNet不同去添加模型结构，仅仅是在文本 Prompt）。在论文[^4]里面主要出发点就是：1、解决**language drif**（语言偏离问题）：指的是模型通过后训练（微调等处理之后）模型丧失了对某些语义特征的感知，就比如说扩散模型里面，模型通过不断微调可能就不知道“狗”是什么从而导致模型生成错误。2、高效的生成需要的对象，不会产生：生成错误、细节丢失问题，比如说下面图像中的问题：
![](https://s2.loli.net/2025/07/12/mRaHPOtC23li9Fn.webp)
为了实现图像的“高效迁移”，作者直接将图像（比如说我们需要风格化的图片）作为一个特殊的标记，也就是论文里面提到的 `a [identifier] [class noun]`（其中class noun为类别比如所狗，identifier就是一个特殊的标记），在prompt中加入类别，通过利用预训练模型中关于该类别物品的先验知识，并将先验知识与特殊标记符相关信息进行融合，这样就可以在不同场景下生成不同姿势的目标物体。就比如下面的 `fine-tuning`过程通过几张图片让模型学习到 *特殊的狗*，然后再推理阶段模型可以利用这个 *特殊的狗*去生成新的动作。**换言之**就是（以下面实际DreamBooth代码为例）：首先通过几张 *狮子狗* 图片让模型知道 *狮子狗*张什么样子，然后再去生成 *狮子狗*的不同的动作。
![](https://s2.loli.net/2025/07/12/hYM1VdykDxALrGo.webp)
在论文里面作者设计如下的Class-specific Prior Preservation Loss（参考stackexchange）[^5]：

$$
\begin{aligned}
 & \mathbb{E}_{x,c,\epsilon,t}\left[\|\epsilon-\varepsilon_{\theta}(z_{t},t,c)\|_{2}^{2}+\lambda\|\epsilon^{\prime}-\epsilon_{pr}(z_{t^{\prime}}^{\prime},t^{\prime},c_{pr})\|_{2}^{2}\right]
\end{aligned}
$$

上面损失函数中后面一部分就是我们的先验损失，比如说$c_{pr}$就是对 "a dog"进行编码然后计算生成损失。在代码中：

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
#### DreamBooth代码操作
> 代码：[https://github.com/shangxiaaabb/ProjectCode/tree/main/code/Python/DFModelCode/training_dreambooth_lora/](https://github.com/shangxiaaabb/ProjectCode/tree/main/code/Python/DFModelCode/training_dreambooth_lora/)
> 权重：[https://www.modelscope.cn/models/bigyellowjie/SDXL-DreamBooth-LOL/files](https://www.modelscope.cn/models/bigyellowjie/SDXL-DreamBooth-LOL/files)

在介绍DreamBooth代码之前，简单回顾DreamBooth原理，我希望我的模型去学习一种画风那么我就需要准备**样本图片**（如3-5张图片）这几张图片就是专门的模型需要学习的，但是为了防止模型过拟合（模型只学习了我的图片内容，但是对一些细节丢掉了，比如说我提供的5张油画，模型就学会了我的油画画风但是为了防止模型对更加多的油画细节忘记了，那么我就准备 `num_epochs * num_samples` 张油画**类型图片**然后通过计算 `Class-specific Prior Preservation Loss`）需要准备 **类型图片**来计算Class-specific Prior Preservation Loss。代码处理（SDXL+Lora）：
DreamBooth中**数据处理过程**：结合上面描述我需要准备两部分数据集（如果需要计算 `Class-specific Prior Preservation Loss`）分别为：`instance_data_dir`（与之对应的 `instance_prompt`）以及 `class_data_dir`（与之对应的 `class_prompt`）而后需要做的就是将两部分数据组合起来构成：

```python
batch = {
    "pixel_values": pixel_values,
    "prompts": prompts,
    "original_sizes": original_sizes,
    "crop_top_lefts": crop_top_lefts,
}
```
模型训练过程**首先是lora处理模型**：在基于transformer里面的模型很容易使用lora，比如说下面代码使用lora包裹模型并且对模型权重进行保存：
```python
from peft import LoraConfig
def get_lora_config(rank, dropout, use_dora, target_modules):
    '''lora config'''
    base_config = {
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": dropout,
        "init_lora_weights": "gaussian",
        "target_modules": target_modules,
    }
    return LoraConfig(**base_config)
# 包裹lora模型权重
unet_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
unet_lora_config = get_lora_config(
    rank= config.rank,
    dropout= config.lora_dropout,
    use_dora= config.use_dora,
    target_modules= unet_target_modules,
)
unet.add_adapter(unet_lora_config)
```
一般的话考虑SD模型权重都比较大，而且我们使用lora微调模型没必要对所有的模型权重进行存储，那么一般都会定义一个 `hook`来告诉模型那些参数需要保存、加载，这样一来使用 `accelerator.save_state(save_path)` 就会先去使用 `hook`处理参数然后进行保存。：
```python
def save_model_hook(models, weights, output_dir):
    if accelerator.is_main_process:
        unet_lora_layers_to_save = None
      
        for model in models:
            if isinstance(model, type(unwrap_model(unet))):
                unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            ...
            weights.pop() # 去掉不需要保存的参数

        StableDiffusionXLPipeline.save_lora_weights(
            output_dir,
            unet_lora_layers= unet_lora_layers_to_save,
            ...
        )
def load_model_hook(models, input_dir):
    unet_ = None

    while len(models) > 0:
        model = models.pop()

        if isinstance(model, type(unwrap_model(unet))):
            unet_ = model

    lora_state_dict, network_alphas = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)

    unet_state_dict = {f"{k.replace('unet.', '')}": v for k, v in lora_state_dict.items() if k.startswith("unet.")}
    unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
    incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
    ...
accelerator.register_save_state_pre_hook(save_model_hook)
accelerator.register_load_state_pre_hook(load_model_hook)
```
**其次模型训练**：就是常规的模型训练（直接在样本图片：`instance_data_dir`以及样本的prompt：`instance_prompt`上进行微调）然后计算loss即可，如果涉及到 `Class-specific Prior Preservation Loss`（除了上面两个组合还需要：`class_data_dir`以及 `class_prompt`）那么处理过程为（以SDXL为例），不过需要事先了解的是在计算这个loss之前会将两个数据集以及prompt都**组合到一起成为一个数据集**（`instance-image-prompt` 以及 `class-image-prompt`之间是匹配的）：

```python
# 样本内容编码
instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(config.instance_prompt, text_encoders, tokenizers)
# 类型图片内容编码
if config.with_prior_preservation:
    class_prompt_hidden_states, class_pooled_prompt_embeds = compute_text_embeddings(config.class_prompt, text_encoders, tokenizers)
...
prompt_embeds = instance_prompt_hidden_states
unet_add_text_embeds = instance_pooled_prompt_embeds
if not config.with_prior_preservation:
    prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
    unet_add_text_embeds = torch.cat([unet_add_text_embeds, class_pooled_prompt_embeds], dim=0)
...
model_pred = unet(...)
if config.with_prior_preservation:
    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
    target, target_prior = torch.chunk(target, 2, dim=0)
    ...
    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
...
loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
...
loss = loss + config.prior_loss_weight * prior_loss
accelerator.backward(loss)
```
在这个里面之所以用 `chunk`是因为如果计算 `Class-specific Prior Preservation Loss`里面的文本prompt是由两部分拼接构成的 `torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)`那么可以直接通过chunk来分离出两部分（这个过程和使用参数 `guidance_rescale`很相似）
最后测试的结果为（`prompt: "A photo of Rengar the Pridestalker in a bucket"`，模型[代码](https://github.com/shangxiaaabb/ProjectCode/tree/main/code/Python/DFModelCode/training_dreambooth_lora/)以及[权重下载](https://www.modelscope.cn/models/bigyellowjie/SDXL-DreamBooth-LOL/files)）：

![image.png](https://s2.loli.net/2025/07/15/7xIPMW6SJ1degZj.webp)

<!-- ## 简易Demo代码
通过总结上面代码在“微调”DF模型中一个简易的代码流程（以微调SDXL模型为例）为（SDXL模型可以直接参考[training_dreambooth_lora](https://github.com/shangxiaaabb/ProjectCode/tree/main/code/Python/DFModelCode/training_dreambooth_lora)）：
**1、基础模型加载**
SDXL区别SD1.5其存在两个文本编码器因此在加载过程中需要加载两个文本编码器，并且基础模型加载主要是加载如下几个模型（如果*不涉及到文本可能就不需要文本编码器*）：1、文本编码器；2、VAE模型；3、Unet模型；4、调度器。除此之外对于所有的模型都会不去就行参数更新。
**2、精度设置** -->

## 总结
对于不同的扩散（基座）模型（SD1.5、SDXL、Imagen）等大部分都是采用Unet结构，当然也有采用Dit的，这两个模型（SD1.5、SDXL）之间的差异主要在于后者会多一个clip编码器再文本语义上比前者更加有优势。对于adapter而言，可以直接理解为再SD的基础上去使用“风格插件”，这个插件不去对SD模型进行训练（从而实现对参数的减小），对于ControNet就是直接对Unet的下采样所有的模块（前后）都加一个zero-conv而后将结果再去嵌入到下采用中，而T2I-Adapter则是去对条件进行编码而后嵌入到SD模型（上采用模块）中。对于deramboth就是直接通过给定的样本图片去生“微调”模型，而后通过设计的Class-specific Prior Preservation Loss来确保所生成的样本特里不会发生过拟合。
## 参考
[^1]: [https://arxiv.org/pdf/2307.01952](https://arxiv.org/pdf/2307.01952)
[^2]: [https://arxiv.org/pdf/2302.05543](https://arxiv.org/pdf/2302.05543)
[^3]: [https://arxiv.org/pdf/2302.08453](https://arxiv.org/pdf/2302.08453)  
[^4]: [https://arxiv.org/pdf/2208.12242](https://arxiv.org/pdf/2208.12242)
[^5]: https://stats.stackexchange.com/questions/601782/how-to-rewrite-dreambooth-loss-in-terms-of-epsilon-prediction
[^6]: [https://arxiv.org/pdf/2205.11487](https://arxiv.org/pdf/2205.11487)
[^8]: [https://arxiv.org/pdf/2405.08748](https://arxiv.org/pdf/2405.08748)
[^9]: [https://arxiv.org/pdf/2506.15742](https://arxiv.org/pdf/2506.15742)
[^10]: [https://arxiv.org/pdf/2310.00426](https://arxiv.org/pdf/2310.00426)
[^11]: [Scalable Diffusion Models with Transformers](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf)
[^12]: [https://arxiv.org/pdf/2403.03206](https://arxiv.org/pdf/2403.03206)
[^13]: [https://arxiv.org/pdf/2109.07161](https://arxiv.org/pdf/2109.07161)
[^14]: [https://zhuanlan.zhihu.com/p/684068402](https://zhuanlan.zhihu.com/p/684068402)
[^15]: [https://zhouyifan.net/2024/09/03/20240809-flux1/](https://zhouyifan.net/2024/09/03/20240809-flux1/)
[^16]: [https://zhouyifan.net/2024/07/14/20240703-SD3/](https://zhouyifan.net/2024/07/14/20240703-SD3/)
[^17]: [https://stability.ai/news/stable-diffusion-3-research-paper](https://stability.ai/news/stable-diffusion-3-research-paper)
[^18]: [https://arxiv.org/pdf/2508.02324](https://arxiv.org/pdf/2508.02324)
[^19]: [https://arxiv.org/pdf/2506.15742](https://arxiv.org/pdf/2506.15742)