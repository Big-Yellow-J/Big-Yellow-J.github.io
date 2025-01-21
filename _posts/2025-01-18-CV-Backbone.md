---
layout: mypost
title: 深度学习基础理论————CV中常用Backbone(Resnet/Unet/Vit系列等)
categories: 深度学习基础理论
extMath: true
images: true
address: changsha
show_footer_image: true
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


`Unet`主要介绍3种：`Unet1`，`Unet++`，`Unet3`，主要应用在医学影像分割（当然图像分割领域都适用）

![2](https://s2.loli.net/2025/01/20/8BrmtOAKH6ETc5W.png)

对比上面三种结构，主体结构上并无太大差异，都是首先通过下采样（左边），然后通过上采样（右边）+特征融合。主要差异就在于**如何进行特征融合**。以`Unet`进行理解：

![2](https://s2.loli.net/2025/01/20/2Ip9ZOBF7mCq5uv.png)

**左侧encoder操作**：首先通过两层$3 \times3$卷积进行处理，然后通过一个 **池化**处理
**右侧decoder操作**：一个上采样的卷积层（去卷积层）+特征拼接concat（上图中白色部分就是要拼接的encoder内容）+两个3x3的卷积层（ReLU）反复构成。
`Unet`相比更早提出的`FCN`网络，使用拼接来作为特征图的融合方式。`FCN`是通过特征图对应像素值的**相加**来融合特征的；`U-net`通过**通道数的拼接**。
**`Unet`好处就在于，因为是逐层的去累加卷积操作，随着卷积的“深入”，越往下的卷积就拥有更加大的 *感受野*，但局部细节可能会逐渐丢失。为了解决这个问题，通过 *上采样*操作来恢复这些细节。上采样操作将低分辨率的特征图尺寸恢复到较高分辨率，从而保留更多的局部特征，弥补下采样过程中丢失的细节。最后将两部分内容继续融合（里面的skip-connection操作）相互进行弥补实现较好性能**

> **感受野**：可以简单理解：比如说一个512x512图像，最开始用卷积核（假设为3x3）去“扫”，那么这个卷积核就会把其“扫”的内容“汇总”起来，比如说某一个值是汇聚了他周围其他的值，这样一来**细节的感知就很多**，但是随着网络层数叠加，这些细节内容就会越来越少，但是计算得到的每个值却是“了解”到了更加“全局”的内容，如下图展示一样
> ![3](https://s2.loli.net/2025/01/20/CwfnKalUu1zNgox.png)
>
> **上采样**：可以简单理解为：将图片给“扩大”，既然要扩大，那么就会需要对内容进行填补，因此就会有不同的插值方式：'nearest', 'linear', 'bilinear', 'bicubic'（`pytorch`提供的）
> ![4](https://s2.loli.net/2025/01/20/UQMEFlPKks8DRLt.webp)
>
> 补充一点： **亚像素上采样 (Pixel Shuffle)**：普通的上采样采用的临近像素填充算法，主要考虑空间因素，没有考虑channel因素，上采样的特征图人为修改痕迹明显，图像分割与GAN生成图像中效果不好。为了解决这个问题，ESPCN中提到了亚像素上采样方式。[具体原理](https://www.cnblogs.com/zhaozhibo/p/15024928.html)如下
> ![](https://s2.loli.net/2025/01/21/5oYgfqvnFswNR7X.png)
>
> 根据上图，可以得出将维度为$[B,C,H,W]$的 feature map 通过亚像素上采样的方式恢复到维度$[B,C,sH,sW]$的过程分为两步：
> 1.首先通过卷积进行特征提取，将$[B,C,H,W]=>[B,s^2C,H,W]$
> 2.然后通过Pixel Shuffle 的操作，将$[B,s^2C,H,W]=>[B,C,sH,sW]$
> Pixel Shuffle的主要功能就是将这$s^2$个通道的特征图组合为新的$[B,C,sH,sW]$的上采样结果。具体来说，就是将原来一个低分辨的像素划分为$s^2$个更小的格子，利用$s^2$个特征图对应位置的值按照一定的规则来填充这些小格子。按照同样的规则将每个低分辨像素划分出的小格子填满就完成了重组过程。在这一过程中模型可以调整$s^2$个shuffle通道权重不断优化生成的结果。

```python
class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))	# 最终将输入转换成 [32, 9, H, W]
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)	# 通过 Pixel Shuffle 来将 [32, 9, H, W] 重组为 [32, 1, 3H, 3W]
    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x
    
if __name__ == "__main__":
    model = Net(upscale_factor=3)
    input = torch.arange(1, 10, dtype = torch.float32).view(1,1,3,3)
    output = model(input)
    print(output.size())

# 输出结果为：
torch.Size([1, 1, 9, 9])

```

对比三种`UNet`操作，从$1\rightarrow 3$进行特征融合的程度更加多，1：将**同“水平”** 的特征以及“下面”特征进行使用；3：将“左侧”所有的特征以及“下面”特征都进行使用；`Unet++`:不是直接使用简单的`skip-connection`而是回去结合 **邻近水平**和 **邻近水平下**的特征，比如说下面图c通过卷积操作结合$X^{0,0}\text{和}X^{1,0}$的特征。

总结上面三种网络结构改进在于：1、`Skip-connection`方式上区别（也就是**如何进行特征连接过程**）

> `Unet++`网络结构
> ![](https://s2.loli.net/2025/01/21/lPIWTdUvpyKfco5.png)


### 3.其他

对于传统的`AlexNet`，`LeNet`，`GoogleNet`可以去看之前写的内容：
1、https://www.big-yellow-j.top/code/AlexNet.html
2、https://www.big-yellow-j.top/code/LeNet.html
3、https://www.big-yellow-j.top/code/googlenet.html

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
8、https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
9、https://www.cnblogs.com/zhaozhibo/p/15024928.html
10、https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf
11、https://arxiv.org/pdf/1807.10165v1