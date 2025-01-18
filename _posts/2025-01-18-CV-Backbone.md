---
layout: mypost
title: 深度学习基础理论————CV中常用Backbone(Resnet/Unet/Vit系列等)
categories: 深度学习基础理论
extMath: true
images: true
address: changsha
---

主要介绍在CV中常用的Backbone,参考论文中的表格,对不同的任务所使用的backbone如下:
![image](https://s2.loli.net/2025/01/15/xKEOXT6hBdL4ziG.png)

针对上面内容分为两块内容：1、基于卷积神经网络的CV Backbone：1.`Resnet`系列;2.`Unet`系列等；2、基于Transformer的 CV Backbone：1.`Vit`系列等;

## 一、基于卷积神经网络的CV Backbone：
### 1. `Resnet`系列

主要有[何凯明大佬](https://arxiv.org/pdf/1512.03385)提出，主要有`resnet18`，`resnet34`，`resnet50`，`resnet101`，`resnet152`，这几种区别主要就在于卷积层数上存在差异（18：18个卷积后面依次类推）,对于`Resnet`论文中最重要的一个就是`残差连接`：
![残差连接](https://s2.loli.net/2025/01/18/cqUbe39QZTjwC5f.png)

因为随着不断的叠加卷积层数，那么就容易导致 **梯度消失**以及 **退化**问题，残差连接就是通过跳跃连接（skip connection），允许输入信息绕过若干层直接传递到后面的层：

$$
h^{l+1} = h^l + F(h^l, W^l)
$$

其中$x$表示我们输入，$h^l$第$l$层的输入，$F(h^l, W^l)$残差分支的非线性变换。对于上面提到的两个问题，残差连接之所以能够缓解，是因为：
1、对于**梯度消失问题**（对于一个神经网络结构，由于反向传播时梯度不断地被链式法则的多个小梯度乘积缩小，最终在靠近输入层的地方梯度变得接近于零，导致参数无法有效更新），残差连接在反向传播时引入了一个 **恒等映射（identity mapping）**，使得梯度可以沿着跳跃路径直接传递给前层。这避免了梯度完全依赖深层网络中的权重进行传播。数学上表述就是，对于残差网络连接而言**梯度传递**为：

$$
\frac{\partial h^{L}}{\partial h^{l}} = I + \sum_{k=l}^{L-1} \prod_{j=k+1}^{L-1} \frac{\partial F(h^j, W^j)}{\partial h^j}
$$

$I$: 恒等映射项，确保梯度具有直接路径传播至浅层。$\sum_{k=l}^{L-1}$: 累积的非线性变换贡献。对于前馈神经网络而言**梯度传递**为：

$$
\frac{\partial h^L}{\partial h^l} = \prod_{k=l}^{L-1} \frac{\partial F(h^k, W^k)}{\partial h^k}
$$

对比很容易发现，如果某一层出现梯度值很大/小问题，那么就会导致这个效果被不断的扩大，但是残差连接就可以较好的避免这个问题，从另外一种角度而言，**对于第$l+1$层的输入不仅仅只考虑$l$层的信息，还要去结合输入$l$层的信息$x$**（就好比：**传递口号，在A这里错了，后面（B，C）可能就都是错了，但是A传递正确给B，B如果传递错误给C但是C还要听一下A怎么给B讲的，这样就可以很好的保证后面口令都不错误**）

2、对于 **退化问题**：网络深度增加时，传统深层网络会因优化难度增大而导致训练误差不降反升。残差连接引入恒等映射，使网络每层只需学习输入与目标之间的 **差值（residual）**，降低了优化的难度：
- 如果某层优化失败，跳跃连接仍能保留输入特征，从而避免性能下降。
- 在极端情况下，残差网络等价于浅层网络（当 $F(x) = 0$时）。

简易代码`Demo`（直接用`torch`）:

```python
import torch
import torchvision.models as models

model = models.resnet50(pretrained=False) # true就会下载权重
```

具体修改某一层参数，可以先将模型`print`出来然后直接进行修改。比如说修改`resnet50`中最后的线性层：

```
# 原始结构：(fc): Linear(in_features=2048, out_features=1000, bias=True)
model.fc = nn.Linear(2048, 10) # 预测10个类别
```

总结一下上面提到的`Resnet`的输出，假设输入图片为：$1 \times 3 \times 512 \times 512$

| 模型         | `Layer1` (blocks)               | `Layer2` (blocks)               | `Layer3` (blocks)               | `Layer4` (blocks)               | 总 block 数量 |
|:--------------:|:-------------------------------:|:-------------------------------:|:-------------------------------:|:-------------------------------:|:-------------:|
| **ResNet-50**  | 3:`1, 256, 128, 128`            | 4:`1, 512, 64, 64`              | 6:`1, 1024, 32, 32`             | 3:`1, 2048, 16, 16`             | 16 (Bottleneck)|
| **ResNet-101** | 3:`1, 256, 128, 128`            | 4:`1, 512, 64, 64`              | 23:`1, 1024, 32, 32`            | 3:`1, 2048, 16, 16`             | 33 (Bottleneck)|
| **ResNet-152** | 3:`1, 256, 128, 128`            | 8:`1, 512, 64, 64`              | 36:`1, 1024, 32, 32`            | 3:`1, 2048, 16, 16`             | 50 (Bottleneck)|
| **ResNet-18**  | 2:`1, 64, 256, 256`             | 2:`1, 128, 128, 128`            | 2:`1, 256, 64, 64`              | 2:`1, 512, 32, 32`              | 8             |
| **ResNet-34**  | 3:`1, 64, 256, 256`             | 4:`1, 128, 128, 128`            | 6:`1, 256, 64, 64`              | 3:`1, 512, 32, 32`              | 16            |

**Bottleneck 层的具体结构**：

假设我们有一个输入张量 \( X \)，其通道数为 \( C_{in} \)，输出通道数为 \( C_{out} \)，通过 Bottleneck 结构后，网络的计算可以分为以下几个步骤：
1. **第一层卷积**（瓶颈）：\( 1 \times 1 \) 卷积，将输入通道数从 \( C_{in} \) 降低到一个较小的中间通道数 \( C_{mid} \)（通常 \( C_{mid} < C_{in} \)）。
   * 输出形状：$$ H \times W \times C_{mid} $$
2. **第二层卷积**：\( 3 \times 3 \) 卷积，进行特征提取。
   * 输出形状：$$ H \times W \times C_{mid} $$

3. **第三层卷积**：再次使用 \( 1 \times 1 \) 卷积，将中间通道数 \( C_{mid} \) 恢复到输出通道数 \( C_{out} \)。
   * 输出形状：$$ H \times W \times C_{out} $$

最终，输入 \( X \) 和输出 \( Y \) 通过残差连接相加，形成一个新的输出。残差连接使得信息能够直接流经网络的不同层，从而避免梯度消失问题。

### 2.`Unet`系列

`Unet`主要3种：`Unet1`，`Unet2`，`Unet3`

![2](https://s2.loli.net/2025/01/18/1gfAhQxrO4DbIcv.png)

### 3.其他

对于传统的`AlexNet`，`LeNet`，`GoogleNet`可以去看之前写的内容：
1、https://www.big-yellow-j.top/posts/2024/01/01/alexnet.html
2、https://www.big-yellow-j.top/posts/2024/01/01/LeNet.html
3、https://www.big-yellow-j.top/posts/2024/01/01/GoogleNet.html

## 二、基于Transformer的CV Backbone

主要介绍两种：`Vit`和`MAE`

![2](https://s2.loli.net/2025/01/18/csRmbCPaGz7yA3e.png)

# 参考:
1、https://arxiv.org/pdf/2206.08016
2、https://arxiv.org/pdf/1512.03385
3、https://arxiv.org/pdf/2010.11929
4、https://arxiv.org/pdf/2111.06377
5、https://arxiv.org/pdf/1505.04597
6、https://arxiv.org/pdf/2311.17791
7、https://arxiv.org/pdf/2004.08790

![1](https://s2.loli.net/2025/01/18/hZzmJaRBQukPLC2.png)