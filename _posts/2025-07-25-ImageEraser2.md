---
layout: mypost
title: 图像消除论文-2：RORem、ObjectClear
categories: 图像消除
address: 武汉🏯
extMath: true
show_footer_image: true
tags:
- diffusion model
- 图像消除
special_tag: 更新中
description: 
---

本文主要介绍几篇图像擦除论文模型：RORem、ObjectClear
## RORem
> https://arxiv.org/pdf/2501.00740
> https://github.com/leeruibin/RORem


## ObjectClear
> https://arxiv.org/pdf/2505.22636
> https://github.com/zjx0101/ObjectClear
> **基座模型**：**SDXL-Inpainting**

本文出发点主要为两个：1、创建数据集；2、通过引入注意力机制（attention-mask）去引导模型消除
### 数据集构建
![image.png](https://s2.loli.net/2025/07/25/bKsqRJNEf3DcH2a.png)

主要为两部分数据集：1、拍摄数据集（2875张图片）；2、模型数据集。对于 **拍摄数据集**：首先将图片处理为512x512而后，直接通过DIBO+SAM去识别然后切割实体得到mask（$M_o$）然后去结合“mask相关的语义特征”（比如说阴影 $M_e$ 等）得到$M_{fg}=[M_o,M_e]$

### 模型结构
![image.png](https://s2.loli.net/2025/07/25/XUlB5YM8zNqS2hZ.png)

将基座模型（**SDXL-Inpainting**）的输入（1、噪声分布$z_t$；2、masked image $I_m$这个一般就是直接将mask从图中扣除；3、mask：$M_o$；4、文本prompt：$c$）中的mask image替换为原始的图像：$I_{in}$（这和之前的SmartEraser处理方式类似），除此之外对于输入DF模型中的处理思路和SmartEraser也是相似的：**图像（需要消除的）和文本分别通过Clip不同编码器处理然后组合**。而后输入到DF的Attention计算当中，对于特征组合部分代码处理方式为：
```python
text_object_embed = self.fuse_fn(object_embeds)#两层mlp+norm
text_embeds_new = text_embeds.clone()
text_embeds_new[:, fuse_index, :] = text_object_embed.squeeze(1)
```
将得到的`text_object_embed`作为文本特征编码特征输入到unet中进行计算。除此之外论文中添加了一个mask loss计算。
除此之外使用`Attention-Guided Fusion`具体代码处理方式为：
```python3
def unet_store_cross_attention_scores(self, unet, attention_scores):
    from diffusers.models.attention_processor import (
        Attention,
        AttnProcessor,
        AttnProcessor2_0,
    )
    import types

    UNET_LAYER_NAMES = [
        "down_blocks.0",
        "down_blocks.1",
        "down_blocks.2",
        "mid_block",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
    ]

    start_layer = 0
    end_layer = 2
    applicable_layers = UNET_LAYER_NAMES[start_layer:end_layer]

    def make_new_get_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(
                query, key, attention_mask
            )
            attention_scores[name] = attention_probs
            return attention_probs

        return new_get_attention_scores

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in applicable_layers):
                continue
            if isinstance(module.processor, AttnProcessor2_0):
                module.set_processor(AttnProcessor())
            module.old_get_attention_scores = module.get_attention_scores
            module.new_get_attention_scores = types.MethodType(
                make_new_get_attention_scores_fn(name), module
            )
            module.get_attention_scores = module.new_get_attention_scores

    return unet
....
fuse_index = 5
if self.config.apply_attention_guided_fusion:
    if i == len(timesteps) - 1:
        attn_key, attn_map = next(iter(self.cross_attention_scores.items()))
        attn_map = self.resize_attn_map_divide2(attn_map, mask, fuse_index)
        init_latents_proper = image_latents
        if self.do_classifier_free_guidance:
            _, init_mask = attn_map.chunk(2)
        else:
            init_mask = attn_map
        attn_map = init_mask
    self.clear_cross_attention_scores(self.cross_attention_scores)
...
attn_pils = []
if output_type == "pil" and attn_map is not None:
    for i in range(len(attn_map)):
        attn_np = attn_map[i].mean(dim=0).cpu().numpy() * 255.
        attn_pil = PIL.Image.fromarray(attn_np.astype(np.uint8)).convert("L")
        attn_pils.append(attn_pil)
    
    original_pils = self.image_processor.postprocess(init_image, output_type="pil")

    generated_pils = image

    fused_images = []
    for i in range(len(generated_pils)):
        ori_pil = original_pils[i]
        gen_pil = generated_pils[i]
        attn_pil = attn_pils[i]

        fused_np = attention_guided_fusion(np.array(ori_pil), np.array(gen_pil), np.array(attn_pil))
        fused_pil = PIL.Image.fromarray(fused_np.astype(np.uint8)).resize(ori_pil.size)

        fused_images.append(fused_pil)

    image = fused_images
```
**首先**对于`unet_store_cross_attention_scores`主要是处理如下两个步骤：1、搜集down_blocks.0和down_blocks.1中attn2模块的注意力分数；2、将down_blocks.0和down_blocks.1中attn2模块处理器从AttnProcessor2_0替换为AttnProcessor