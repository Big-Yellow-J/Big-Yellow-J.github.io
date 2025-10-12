---
layout: mypost
title: 多模态算法QwenVL、KimiVL等算法原理
categories: 多模态
extMath: true
images: true
address: 武汉🏯
tags:
- cv-backbone
- 多模态
- multimodal
show_footer_image: true
description: 多模态大语言模型通用框架通过视觉编码器（如ViT）、文本编码器及映射层对齐维度输入LLM。QwenVL系列（QwenVL、QwenVL2、QwenVL2.5）为典型实现，QwenVL采用ViT-bigG视觉编码器，经可学习query的Cross-Attention压缩视觉token至256长度，融合二维绝对位置编码；QwenVL2改进为动态分辨率（无需固定尺寸，2x2
  token拼接+MLP）及多模态旋转位置编码（M-RoPE，含时序、高度、宽度信息），提升处理性能。
---

对于多模态系列模型大致的多模态大语言模型的通用模型框架和每个模块的一些实现方法[^1]：
![](https://s2.loli.net/2025/09/21/JF9YdeEAhuMyzkZ.webp)
基本上就是对于图片/视频等通过不同的视觉编码器（Vit/Clip等）进行编码，对于text通过编码器进行编码，而后将视觉模态信息通过映射层（q-former/mlp等）将两部分维度对齐而后丢到LLM中输出结果。简单总结常用的多模态模型。
## QwenVL系列
目前QwenVL迭代更新迭代到3（**截至2025.10.10**）主要介绍QwenVL、QwenVL2、QwenVL2.5、QwenVL3
### QwenVL
在QwenVL[^4]中在论文里面作者提到的其模型的整个训练过程如下：
![](https://s2.loli.net/2025/09/21/HEhlRPFJBMKpjoZ.webp)
> 仅从提供的不同阶段还是很容易发现QwenVL还是是采用和BLIP相似的使用 learned-query来对齐模态信息
> 语言模型使用（7.7B）：Qwen-7B
> 视觉编码器（1.9B）：Vit-bigG
> 融合器（0.08B）：Learnable Query

不过论文里面对于模型细节介绍不是很多，从代码角度出发窥其模型结构：
**模型视觉编码器**：视觉编码器使用的是ViT架构（Vision Transformer），ViT的网络设置和初始化参数使用了OpenCLIP预训练好的**ViT-bigG模型**。具体的代码处理过程（[代码](https://huggingface.co/Qwen/Qwen-VL/blob/main/visual.py)），其中模型输出维度变化过程：1x3x448x448-->1x1664x32x32（首先卷积处理）-->1x1024x1664（拉平交换维度）
**特征融合器**：上述ViT处理后，对于$448\times 448$分辨率的图像，生成一个 **[1024, 1664]**的序列，也就是向量维度为1664的长度为1024的序列。为了压缩视觉token的输入长度，Qwen-VL引入了一个Adapter来压缩图像特征。这个Adaper就是一个随机初始化的单层Cross-Attention模块。该模块使用一组可学习的query向量，将来自ViT的图像特征作为Key向量。通过Cross-Attention操作后将视觉特征序列压缩到固定的256长度（也就是将视觉特征压缩到 **256 1644**）
此外，考虑到位置信息对于精细图像理解的重要性，Qwen-VL将二维绝对位置编码（三角位置编码）整合到Cross-Attention的 $q,k$中，以减少压缩过程中可能丢失的位置细节。随后将长度为256的压缩图像特征序列输入到大型语言模型中。
### QwenVL-2
对于QwenVL-2[^3]其模型的基本结构如下：
![](https://s2.loli.net/2025/09/21/5c1jovnLVOaS62H.webp)
**1、使用动态分辨率**（也就是说输入图像不需要再去改变图像尺寸到一个固定值），于此同时为了减少 **visual-token**数量，将**2x2的的相邻的token进行拼接**到一个token而后通过MLP层进行处理。
![](https://s2.loli.net/2025/09/21/w3agENHmLVcoSdt.webp)
**动态分辨率**处理如上，通过指定`[mix_pixels, max_pixels]`范围然后将图像保持原始的纵横比去缩减图像到上面的范围中（[处理过程](https://github.com/QwenLM/Qwen2.5-VL/blob/c15045f8829fee29d4b3996e068775fe6a5855db/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L59)，首先计算原始图像的像素数量，而后判断和上面指标的范围，如果超出范围就去计算需要修改的比例，在将整个比例去处理到分辨率上）
在通过使用动态分辨率处理图像之后会在单一**图片增加时间维度**也就是将：CHW-->TCHW（这点是为了和视频处理过程进行对齐），在源码中T选择数值为2也就是将图片“复制一次”，而后对帧序列进行Patchification操作
```python
def _preprocess(): 
    ......   
    channel = patches.shape[1]
    grid_t = patches.shape[0] // self.temporal_patch_size
    grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
    patches = patches.reshape(
        grid_t,                            # 0
        self.temporal_patch_size, channel, # 1 2
        grid_h // self.merge_size,         # 3
        self.merge_size, self.patch_size,  # 4 5
        grid_w // self.merge_size,         # 6
        self.merge_size, self.patch_size,  # 7 8
    ) # self.merge_size=2 self.patch_size=14 self.temporal_patch_size=2
    ### 将2x2的邻域Patch放到一起，方便后续做领域的Patch过Projector层做聚合压缩
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    ### Patch序列化，并保留Patch位置信息（时间，高，宽）
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
    )
```
上面过程也就是进行所谓的“2x2的相邻token拼接”，最后得到`[grid_t * grid_h * grid_w, channel * temporal_patch_size(2) * patch_size(14) * patch_size(14)]`（其中`grid_h=resized_height // self.patch_size(14)`）
2、**多模态的旋转位置编码（M-RoPE）**,也就是将原来位置编码所携带的信息处理为：时序（temporal）、高度（height）、宽度（width）。比如下图中对于文本处理直接初始化为：$(i,i,i)$。但是对于图片而言就是：$(i,x,y)$ 其中 $i$ 是恒定的，而对于视频就会将 $i$ 换成视频中图像的顺序
**总结处理过程**：动态分辨率处理-->复制时间维度-->将序列切割为patch。这样一来就会直接将图像处理为：`[grid_t * grid_h * grid_w, channel * temporal_patch_size(2) * patch_size(14) * patch_size(14)]`（其中`grid_h=resized_height // self.patch_size(14)`）除此之外而后去计算 3d-RoPE最后通过一层线性层处理就得到最后的视觉token。
### QwenVL-2.5
在QwenVL2.5中[^6]模型具体的代码处理过程参考Blog[^5]具体模型结构：
![](https://s2.loli.net/2025/09/21/R8yLfVqpznvkgZw.webp)
在图像处理过程上和QwenVL2差异不大都是直接：动态分辨率处理-->复制时间维度-->将序列切割为patch，对比两个模型差异：
![](https://s2.loli.net/2025/09/22/NvKgQqhC36WAkjU.webp)
1、采用 RMSNorm 替换了所有 LayerNorm；2、ViT中每一个VisionBlock中的MLP换成了SwiGLU 结构。只从模型结构上差异不到，在QwenVL2.5中主要进行改动：1、使用window-attention（对应上述结构中的`Qwen2_5_VLVisionAttention`）对于具体的划分window方法（[代码](https://github.com/huggingface/transformers/blob/41925e42135257361b7f02aa20e3bbdab3f7b923/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L465)）：根据输入的图像大小 (gird_t, grid_h, grid_w)去得到窗口索引 (window_index) 和 累积序列长度 (cu_window_seqlens)。具体例子如下：
```python
# 数据数据特征
[ [ 0,  1,  2,  3,  4,  5],
  [ 6,  7,  8,  9, 10, 11],
  [12, 13, 14, 15, 16, 17],
  [18, 19, 20, 21, 22, 23],
  [24, 25, 26, 27, 28, 29],
  [30, 31, 32, 33, 34, 35] ]
# 保证可以被window_size划分需要进行填充
[ [ 0,  1,  2,  3,  4,  5, X, X],
  [ 6,  7,  8,  9, 10, 11, X, X],
  [12, 13, 14, 15, 16, 17, X, X],
  [18, 19, 20, 21, 22, 23, X, X],
  [24, 25, 26, 27, 28, 29, X, X],
  [30, 31, 32, 33, 34, 35, X, X],
  [ X,  X,  X,  X,  X,  X, X, X],
  [ X,  X,  X,  X,  X,  X, X, X] ]
# 而后直接更具window大小得到每个需要计算注意力的window
# window-0
[ 0,  1,  2,  3]
[ 6,  7,  8,  9] 
[12, 13, 14, 15]
[18, 19, 20, 21]
# 展平重新排列得到：
# window-0
[0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21]
# window-1 
[4, 5, 10, 11, 16, 17, 22, 23]
# 计算累计长度
seqlens = (index_padded != -100).sum([2, 3]) # 计算有效长度：window-0：16 window-1：8.....
cu_seqlens_tmp = seqlens.cumsum(0) * 4 + cu_window_seqlens[-1]
cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
# [0, 64, 96, 128, 144]
# 得到最后返回结果window_index, cu_window_seqlens
```
在得到window_index和cu_window_seqlens之后就是[计算注意力过程](https://github.com/huggingface/transformers/blob/41925e42135257361b7f02aa20e3bbdab3f7b923/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L267C9-L275C100)
```python
for i in range(1, len(cu_seqlens)):
  attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

q = q.transpose(0, 1)
k = k.transpose(0, 1)
v = v.transpose(0, 1)
attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
attn_weights = attn_weights + attention_mask
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
```
### QwenVL-3
在官方Blog[^7]的介绍中
![](https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vl_arc.jpg#center)
对于模型架构的更新为：1、**MRoPE-Interleave**: 改进位置编码，采用时间(t)、高度(h)、宽度(w)交错分布形式，提升对长视频的理解能力。2、**DeepStack 技术**: 融合 ViT 多层次特征，将视觉特征注入 LLM 的多层中，实现更精细化的视觉理解和图文对齐精度。3、**文本时间戳对齐机制 (T-RoPE 升级)**: 采用“时间戳-视频帧”交错输入形式，实现帧级别时间信息与视觉内容的细粒度对齐，提升视频事件定位精度。整体模型结构在区别上一代QwenVL-2.5改进点在于：patch_embed的patch_size变大了（14->16），embed使用的三维卷积里加了bias，ViT的隐层维度hiddeen_dim从1280->1152，而后使用DeepStack、MRoPE-Interleave。对于具体源码（[代码](https://github.com/huggingface/transformers/blob/0419ff881d7bb503f4fc0f0a7a5aac3d012c9b91/src/transformers/models/qwen3_vl/modular_qwen3_vl.py)）分析整体模型处理过程如下（[代码](https://github.com/huggingface/transformers/blob/0419ff881d7bb503f4fc0f0a7a5aac3d012c9b91/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L885)）
> **值得注意的是在输入数据预处理阶段QwenVL-3和2.5的处理是相同的通过smart_resize去修改分辨率**

```python
class Qwen3VLModel(Qwen3VLPreTrainedModel):
  ...
  def __iniit__(...):
    super().__init__(config)
    self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
    self.language_model = Qwen3VLTextModel._from_config(config.text_config)
    self.rope_deltas = None  # cache rope_deltas here
    self.post_init()
  def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
    ):
    ...
    # 图像处理过程
    if pixel_values is not None:
      image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
      image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
      image_mask, _ = self.get_placeholder_mask(
          input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
      )
      inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```
### `get_image_features`处理过程
> [代码](https://github.com/huggingface/transformers/blob/0419ff881d7bb503f4fc0f0a7a5aac3d012c9b91/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L1048C1-L1062C52)

通过视觉编码处理得到`image_embeds`和 `deepstack_image_embeds`而后再去对 `image_embeds`进行裁剪，裁剪的逻辑为：
`split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist();image_embeds = torch.split(image_embeds, split_sizes)` 回到`self.visual`中模型具体处理过程如下（[代码](https://github.com/huggingface/transformers/blob/0419ff881d7bb503f4fc0f0a7a5aac3d012c9b91/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L701)）：
```python
def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
  # 1、视觉patch处理
  hidden_states = self.patch_embed(hidden_states)
  # 2、添加绝对位置编码
  pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
  hidden_states = hidden_states + pos_embeds

  rotary_pos_emb = self.rot_pos_emb(grid_thw)

  seq_len, _ = hidden_states.size()
  hidden_states = hidden_states.reshape(seq_len, -1)
  rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
  emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
  position_embeddings = (emb.cos(), emb.sin())

  cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
      dim=0,
      dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
  )
  cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
  # Vit处理
  deepstack_feature_lists = []
  for layer_num, blk in enumerate(self.blocks):
      hidden_states = blk(
          hidden_states,
          cu_seqlens=cu_seqlens,
          position_embeddings=position_embeddings,
          **kwargs,
      )
      if layer_num in self.deepstack_visual_indexes:
          deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
              hidden_states
          )
          deepstack_feature_lists.append(deepstack_feature)

  hidden_states = self.merger(hidden_states)

  return hidden_states, deepstack_feature_lists
```
`patch_embed`就是直接使用3维卷积（bias为True）：`Conv3d(3, 1152, kernel_size=(2, 16, 16), stride=(2, 16, 16))`（维度上对应：`(grid_t*grid_h*grid_w, hiddend_size)`）；
`fast_pos_embed_interpolate`用于处理任意分辨率的输入。在一个**固定分辨率**的网格上（`self.pos_embed` 对应的维度为 `(2304, 1152)` ）预定义位置编码，然后对任意分辨率的输入进行快速插值，得到对应位置的插值位置编码。
> 在 ViT 模型的预训练阶段，通常使用固定的输入分辨率（例如 224×224），并将其划分为固定数量的 patch（例如 14×14，共 196 个 patch）。这意味着模型内部的 pos_embed 是一个固定长度的可学习参数矩阵，模型在训练过程中已经隐式地学习到了这些位置编码之间的空间关系。
> 当推理阶段输入的分辨率发生变化时，如果直接重新计算或生成新的位置编码，就会破坏模型在预训练阶段学到的空间语义信息，从而导致性能下降。
> 因此，QwenVL-3 等模型的做法是：**固定一套在预训练阶段学习到的位置编码**，在输入新的分辨率时，不重新生成编码，而是通过 **双线性插值** 将原始位置编码映射到新的空间尺度上，从而在保持预训练空间结构的前提下，适配不同输入尺寸。换句话说，新的 patch 位置不再重新计算 embedding，而是通过插值在原有位置编码上“找到”其对应的空间位置。

`deepstack_merger_list`处理

### 总结
从QwenVL到QwenVL2.5视觉编码器处理过程：
**QwenVL**：将图像转化为**固定的分辨率**而后将输入到Vit-bigG进行处理得到视觉特征之后再去使用类似Q-former处理过程（QwenVL中使用的是*一个随机初始化的单层Cross-Attention模块*）使用learned-query（压缩到**固定的256长度的token**）将视觉token进行压缩而后输入到LLM中。
**QwenVL2**：首先使用**动态分辨率**（将图像**除以固定的factor而后保持横纵比**将其缩减到 `[mix_pixels, max_pixels]`中）去处理图像而后将其输入到视觉编码器中，而后将**2x2的的相邻的token进行拼接**（也就是将图像补充一个时间帧得到TCHW，而后再去在THW三个维度划分得到不同的patch：grid_t,grid_h,grid_w）到一个token而后通过MLP层进行处理。
**QwenVL2.5**：整体框架上和QwenVL2差异不大，区别在于使用了window-attention以及2D-RoPE
## KimiVL系列
## 参考
[^1]: [https://arxiv.org/abs/2504.07491](https://arxiv.org/abs/2504.07491)
[^2]: [https://zhuanlan.zhihu.com/p/25267823390](https://zhuanlan.zhihu.com/p/25267823390)
[^3]: [http://arxiv.org/abs/2409.12191](http://arxiv.org/abs/2409.12191)
[^4]: [https://arxiv.org/pdf/2308.12966](https://arxiv.org/pdf/2308.12966)
[^5]: [https://www.big-yellow-j.top/posts/2025/08/29/QwenVLCode.html](https://www.big-yellow-j.top/posts/2025/08/29/QwenVLCode.html)
[^6]: [https://arxiv.org/abs/2502.13923](https://arxiv.org/abs/2502.13923)
[^7]: [QwenVL-3-Blog](https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef&from=research.latest-advancements-list)
