---
layout: mypost
title: Code
show_footer_image: false
---

<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Link Card</title>
  <style>
    .link-card {
      display: flex;
      align-items: center;
      max-width: 600px;
      border: 1px solid #e0e0e0;
      border-radius: 8px;
      padding: 16px;
      margin: 16px auto;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      text-decoration: none;
      color: inherit;
      transition: transform 0.2s;
    }
    .link-card:hover {
      transform: translateY(-2px);
    }
    .link-card img {
      width: 32px;
      height: 32px;
      margin-right: 16px;
    }
    .link-card-content {
      flex: 1;
    }
    .link-card-title {
      font-size: 1.2em;
      font-weight: bold;
      margin: 0 0 8px;
    }
    .link-card-description {
      font-size: 0.9em;
      color: #555;
      margin: 0;
    }
    .link-card-url {
      font-size: 0.8em;
      color: #888;
      margin-top: 4px;
    }
  </style>
</head>
<body>
  <a id="link-card" class="link-card" href="" target="_blank">
    <img id="link-favicon" src="" alt="Favicon">
    <div class="link-card-content">
      <div id="link-title" class="link-card-title"></div>
      <div id="link-description" class="link-card-description"></div>
      <div id="link-url" class="link-card-url"></div>
    </div>
  </a>

  <script>
    // Configuration for the link to convert
    const linkConfig = {
      url: "{{ include.url | default: 'https://github.com/Big-Yellow-J/Big-Yellow-J.github.io/tree/master/code/Python' }}",
      title: "{{ include.title | default: 'Code！！😁😁😁' }}",
      description: "{{ include.description | default: 'All Code in my blog！' }}"
    };

    // Update card elements with provided or default values
    document.getElementById('link-card').href = linkConfig.url;
    document.getElementById('link-favicon').src = `https://www.google.com/s2/favicons?domain=${linkConfig.url}`;
    // document.getElementById('link-favicon').src = `https://api.faviconkit.com/${new URL(linkConfig.url).hostname}/144`;
    document.getElementById('link-title').textContent = linkConfig.title;
    document.getElementById('link-description').textContent = linkConfig.description;
    document.getElementById('link-url').textContent = linkConfig.url;
  </script>
</body>
</html>

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