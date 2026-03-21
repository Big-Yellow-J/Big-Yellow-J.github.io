---
layout: mypost
title: 残差连接————Kimi注意力残差/字节混合注意力
categories: paper
address: 武汉🏯
extMath: true
show_footer_image: true
tags:
- 论文
- kimi
description: 传统残差连接通过跳跃连接缓解深度模型梯度消失与退化问题，但存在各层贡献权重一致、浅层信息随层数叠加逐渐被稀释的缺陷，过往门控、加权类改进效果有限。针对该痛点，Kimi提出注意力残差连接，对前序所有block输出计算softmax注意力权重做加权融合，分别在单block计算后、MLP处理前执行两次融合。字节推出混合深度注意力方案，基于GQA的历史KV缓存，同步计算序列维度常规注意力与深度维度历史信息注意力，融合更新输出，解决大模型深度增加后的信号衰减问题。
---

本文主要介绍最新的Kimi的注意力残差连接以及字节的“残差”连接两篇论文，在最开始的残差连接方案[^3]中：
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260321105833272.png)
核心过程就是 $x = x+ f(x)$，随着不断的叠加卷积层数，那么就容易导致 梯度消失以及 退化问题，残差连接就是通过跳跃连接（skip connection），允许输入信息绕过若干层直接传递到后面的层。后续也有很多去对这个过程进行改进比如说使用门控残差连接、加权残差连接、修改连接位置等。不过影响都不是很大，因此对于残差连接过程就一直没有变化还是保持最开始的计算方式了。在kimi以及字节最近新发表两篇论文都是对这个过程做的改进具体解释如下。
## Kimi注意力残差连接
首先按照论文中逻辑出发，在标准的残差计算中：$h_l=h_{l-1}+f_{l-1}(h_{l-1})$ 对于这个计算方式在计算梯度传播过程中会直接将**每一层的贡献是相同**（直接计算上公式梯度）因此后续论文就提出做一个门控的残差连接方式 $h_l=\alpha_l \cdot h_{l-1}+\beta_l \cdot f_{l-1}(h_{l-1})$，对于上述两种残差注意力方式带来最大的问题就是：**所有层的贡献都是一致的，除此之外后续层只能获取前层的信息导致更加前面层的信息被稀释**（比如说l层只能获取l-1层信息，虽然l-2的信息会融合到l-1层但是l-2还是对l层的作用有限）。因此kimi的attention-residual出发点就是让后续层可以看到更加前面层的信息以及通过一个合适权重去控制残差连接，基于这个论文[^2]里面提出方案如下图c中描述：
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260321111201650.png)
对于*第n个block我将前面的几层的输出都进加权融合作为第n层的输入*，具体融合方式为：
$$
h_l= \alpha_{0\rightarrow l}\cdot h_1+ \sum_{i=1}^{l-1}\alpha_{i\rightarrow l}\cdot f_i(h_i)
$$
其中 $\sum_{i=1}^{l-1}\alpha_{i\rightarrow l}=1$那么对于权重系数 $\alpha_{i\rightarrow l}$ 的计算方式为：
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260321213723236.png)
其实也就是计算softmax的注意力权重，里面的 $q_l=w_l$ 通过一个学习的向量以及历史层的输出去计算softmax值去控制权重特征融合。去看代码具体过程：
```python
def block_attn_res(blocks: list[Tensor], partial_block: Tensor, proj: Linear, norm: RMSNorm) -> Tensor:
    V = torch.stack(blocks + [partial_block]) # [N+1, B, T, D]
    K = norm(V)
    logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)
    h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
    return h

def forward(self, blocks: list[Tensor], hidden_states: Tensor) -> tuple[list[Tensor], Tensor]:
    partial_block = hidden_states          # 进入当前层的初始 hidden_states（通常是上一层的输出）
    
    # 在 Attention 子层前，先做一次 Block AttnRes
    h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)
    
    # 如果当前层是 Block 的边界层 → 把当前 partial_block 作为完整 Block 保存下来
    if self.layer_number % (self.block_size // 2) == 0:
        blocks.append(partial_block)       # blocks 列表增长，新增一个完成的 Block rep
        partial_block = None               # 重置 partial（新 Block 从零开始？代码这里有小问题，实际可能要用 h 或重置逻辑）
    
    # 自注意力子层（标准 Transformer attention）
    attn_out = self.attn(self.attn_norm(h))
    partial_block = partial_block + attn_out if partial_block is not None else attn_out
    # ↑ 标准残差：partial_block += attn_out   （Block 内部用经典 +）

    # 在 MLP 子层前，再做一次 Block AttnRes（用不同的 proj 和 norm）
    h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)
    
    # MLP 子层
    mlp_out = self.mlp(self.mlp_norm(h))
    partial_block = partial_block + mlp_out               # 再次标准残差累加
    
    return blocks, partial_block    # 返回更新后的 blocks 列表 + 当前 Block 的 partial sum
```
其实通过代码很容易发现在block计算过程就是，输入前将前n层的block特征进行attention-residual方式特征融合，在计算完毕之后进行一个普通的残差连接，而后在将输出进行mlp处理之前再次通过一次attention-residual连接处理。
## 字节混合注意力
在字节论文[^1]中提出混合注意力去解决：**随着 LLM 的深度增加，它们往往会遭遇信号衰减的问题**：在浅层形成的有用特征会因反复的残差更新而逐渐被稀释，使得它们在更深的层中更难恢复（出发点和kimi的attention-residual相同）。
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260321220611829.png)
对于上图中提到的read以及write分别表示的是残差连接方式 $x=x+f(x)$里面分别对于x以及连接方式，比如说对于最开始残差连接我的read就是x（不去对x进行其他处理因此论文里面将其标记为identity）而我的连接方式是add因此将write处理为add。在上图b中选择直接将所有的信息进行拼接（比如说第i层计算输出就行和输入就行concat操作），虽然在信息传播过程是无损的，可以解决上面的信号衰减问题，但是这样会带来显存占用过高。那么论文里面提出Depth Attention处理过程为，对于输入通过相面方式处理：
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260321224003992.png)
其中对于 $K_i$ 以及 $V_i$ 表示的GQA过程中我的历史缓存的kv值而 $Q_{l-1}$ 则是上一层的Q结果，通过注意力融合方式得到最终的输入 $X_l^{in}$ 直接将这个结果解析attention的注意力计算得到 $X_l^{out}$，在得到结果之后通过：
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260321224445649.png)
又可以得到新的一层的输出结果（相当于替代了之前的残差连接通过相加为线性层处理方式）。除此之外进一步提出升级的 Mixture-of-Depth Attention方式：
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260321232204028.png)
对于上述过程中depth表示所有前面层的深度 KV cache（**对应深度部分**），而QKV则是表示当前层的结果（**对应序列部分**），10-23行处理序列部分注意力（就是比较常规的注意力计算过程），24-29行处理处理深度部分注意力，在计算注意力过程中会用softmax去更新同一个（m, acc, o），相当于将cache部分信息融入到注意力中。
## 总结
两篇论文中都是为了解决随着层数的叠加带来的“信息遗忘”问题，Kimi中选择直接将“历史block”信息通过注意力融合方式进行加权残差连接（attention-residual）也就是 $y=\alpha \cdot h_l+ \sum_{i=1}^{l-1} \alpha h_i$，具体过程为**将历史所有的block结果和用一个可学习的向量之间计算softmax作为权重** $\alpha$ 具体残差发生在：1、mlp处理前；2、每一个block处理之后。在字节的mixture-of-depth attention处理方式则是直接将GQA中的kv-cache中的KV值用来计算注意力去弥补信息损失，具体过程为在序列部分直接计算常规注意力，**在深度部分（KV cache历史结果）部分通过历史结果去更新在序列部分计算得到的注意力结果**。
## 参考
[^1]: [https://arxiv.org/abs/2603.15619](https://arxiv.org/abs/2603.15619)
[^2]: [https://arxiv.org/abs/2603.15031](https://arxiv.org/abs/2603.15031)
[^3]: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)