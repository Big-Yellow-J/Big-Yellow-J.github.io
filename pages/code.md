---
layout: mypost
title: Code
show_footer_image: false
---
> **所有代码**：[https://github.com/Big-Yellow-J/Big-Yellow-J.github.io/tree/master/code/Python](https://github.com/Big-Yellow-J/Big-Yellow-J.github.io/tree/master/code/Python)

**部分代码说明**

| 文件名称               | 实现功能      | 文件地址    |
|:--------------------:|:------------:|:----------:|
| `MHA` | 多头注意力模块,支持 `flash_attn`,输入数据格式为：`x:(B,T,C),atten_mask:(B,T)`   | [🔗](../code/MultiHeadAttention.py.txt)   |
| `GQA` | 分组注意力模块,支持 `flash_attn`,输入数据格式为：`x:(B,T,C),atten_mask:(B,T)`  | [🔗](../code/GroupedQueryAttention.py.txt) |
| `MQA` | 多查询注意力模块,支持 `flash_attn`,输入数据格式为：`x:(B,T,C),atten_mask:(B,T)` | [🔗](../code/MultiHeadAttention.py.txt)   |
| `SWA` | 滑动窗口注意力模块,支持 `flash_attn`,输入数据格式为：`x:(B,T,C),atten_mask:(B,T)` | [🔗](../code/WindowAttention.py.txt)     |
| `MoBA` | Kimi MoBA论文稀疏注意力计算,输入数据格式为：`x:(B,T,C),atten_mask:(B,T)` | [🔗](../code/MoBAAttention.py.txt)     |
| `PosEncoding` | 位置编码,`RotaryPositionalEncoding`,`AbsolutePositionEmbedding`,`LearnedPositionEmbedding`。输入：`x:(B,T,C)`             | [🔗](../code/PositionalEncoding.py.txt) |
| `Norm`        | 归一化操作,`LayerNorm`,`BatchNorm`,`RMSNorm`,`InstanceNorm`,`GlobalResponseNorm`。输入：`(B,T,C)` 或者 `(B,C,H,W)` | [🔗](../code/Norm.py.txt)
| `ResNet`          | 视觉编码器,`ResNet50`, `ResNet101`, `ResNet152`系列  | [🔗](../code/CVBackbone/ResNet.py.txt)          |
| `ConvNeXt`        | 视觉编码器,`ConvNeXt v1`系列                         | [🔗](../code/ConvNeXt.py.txt)                   |
| `Vit`             | 视觉编码器,`Vit`                                    | [🔗](../code/CVBackbone/Vit.py.txt)             |
| `SwinTransformer` | 视觉编码器,`SwinTransformer`                        | [🔗](../code/CVBackbone/SwinTransformer.py.txt) |