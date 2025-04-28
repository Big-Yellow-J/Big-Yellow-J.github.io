---
layout: mypost
title: Code
show_footer_image: false
---

| 文件名称               | 实现功能      | 文件地址    |
|:--------------------:|:------------:|:----------:|
| `MHA` | 多头注意力模块，支持 `flash_attn`，输入数据格式为：`x:(B,T,C),atten_mask:(B,T)`   | [🔗](../code/MultiHeadAttention.py.txt)   |
| `GQA`| 分组注意力模块，支持 `flash_attn`，输入数据格式为：`x:(B,T,C),atten_mask:(B,T)`  | [🔗](../code/GroupedQueryAttention.py.txt) |
| `MQA` | 多查询注意力模块，支持 `flash_attn`，输入数据格式为：`x:(B,T,C),atten_mask:(B,T)` | [🔗](../code/MultiHeadAttention.py.txt)   |
| `SWA` | 滑动窗口注意力模块，支持 `flash_attn`，输入数据格式为：`x:(B,T,C),atten_mask:(B,T)` | [🔗](../code/WindowAttention.py.txt)      |
| `PosEncoding` | 位置编码，`RotaryPositionalEncoding`、`AbsolutePositionEmbedding`、`LearnedPositionEmbedding`。输入：`x:(B,T,C)`             | [🔗](../code/PositionalEncoding.py.txt) |
| `Norm`               | 正则化操作，`LayerNorm`、`BatchNorm`、`RMSNorm`。输入：`(B,T,C)` 或者 `(B,C,H,W)` | [🔗](../code/Norm.py.txt)
| `ResNet`          | 视觉编码器，`ResNet50`, `ResNet101`, `ResNet152`。参数：`num_classes:预测类别, channel_ratio:通道裁剪比率` | [🔗](../code/CVBackbone/ResNet.py.txt)    |
| `Vit`             | 视觉编码器，`Vit`                                | [🔗](../code/CVBackbone/Vit.py.txt)          |
| `SwinTransformer` | 视觉编码器，`SwinTransformer`                    | [🔗](../code/CVBackbone/SwinTransformer.py.txt) |