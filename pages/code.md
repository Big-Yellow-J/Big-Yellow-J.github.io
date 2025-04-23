---
layout: mypost
title: Code
show_footer_image: false
---

| 文件名称               | 实现功能      | 文件地址    |
|:--------------------:|:------------:|:----------:|
| `MultiHeadAttention` | 多头注意力模块，支持 `flash_attn`第三方以及pytorch官方实现，输入数据格式为：`x:(B,T,C),atten_mask:(B,T)`                               | [🔗](../code/MultiHeadAttention.py.txt) |
| `PositionalEncoding` | 位置编码，支持：`RotaryPositionalEncoding`、`AbsolutePositionEmbedding`、`LearnedPositionEmbedding`。输入：`x:(B,T,C)`             | [🔗](../code/PositionalEncoding.py.txt) |
| `Norm`               | 正则化操作，支持：`LayerNorm`、`BatchNorm`、`RMSNorm`。输入：`(B,T,C)` 或者 `(B,C,H,W)`                                             | [🔗](../code/Norm.py.txt)
| `CVBackbone`         | 常用视觉编码器，支持：`ResNet`、`Vit`、`Swin-Transformer`等    | [🔗-ResNEt](../code/CVBackbone/ResNet.py.txt)  [🔗-Vit](../code/CVBackbone/Vit.py.txt)  [🔗-SwinTransformer](../code/CVBackbone/SwinTransformer.py.txt)|