---
layout: mypost
title: 🔥Pytorch使用-1：Pytorch计算图等概念
categories: pytorch
address: 武汉🏯
extMath: true
show_footer_image: true
tags:
- pytorch
- torch.compile
description: PyTorch计算图用于记录张量运算关系，节点分为张量、运算两类，边代表数据流与依赖关系，作为动态图框架，每次前向时即时构建，反向传播沿图从输出节点计算叶子张量梯度，显存占用含数据本身、计算图中间激活两部分，可通过.detach()分离无梯度需求张量、调整精度降低显存开销。PyTorch2.0新增torch.compile，经TorchDynamo抓图生成FX
  Graph、AOTAutograd提前生成反向图、TorchInductor完成算子融合等优化并生成Triton/C++代码实现计算提速，主流训练框架均支持开启。同时梳理reshape、view等张量变形方法的特性差异，明确model.train()、model.eval()、torch.no_grad()的适用场景与功能边界。
---

1、介绍torch基本知识，比如说：计算图等底层原理
推荐阅读torch官方：https://docs.pytorch.org/docs/stable/notes.html
## torch计算图概念
### 基础过程
![](https://s2.loli.net/2025/08/14/gIftdlM7KTw2Yak.webp)

参考上图中给出的例子[^1]，对于（pytorch中的）计算图而言主要起到的作用就是：**用来记录张量之间的运算关系**。涉及到的几个概念：1、节点（Node）：张量（Tensor）或者运算（Function）。2、边（Edge）：表示数据流和依赖关系，指明一个张量是由哪些运算生成的，或一个运算的输入来源于哪个张量。3、叶子节点（Leaf Tensor）：通常是用户创建的、需要梯度的张量（`requires_grad=True`）。4、动态计算图：PyTorch 是 动态图框架，计算图会在每次 forward 运行时即时构建，执行完一次计算后，默认图会释放（除非使用`retain_graph=True`）。当通过调用 `.backward()` 时，**PyTorch 会沿着这个计算图从输出节点反向传播，依次计算每个叶子节点的梯度**。比如说对于上面的过程（上面过程中每一个圆圈节点就会对应一个节点，那么反向传播就可以去计算这些节点梯度去对参数进行更新）：$z=w=y_1\times y_2= \log(a) \times \sin(x_2)=\log(x_1\times x_2)+ \sin(x_2)$

```python
import torch

X1 = torch.tensor(2.0, requires_grad=True)
X2 = torch.tensor(3.0, requires_grad=True)

a = X1 * X2                  # a = X1 * X2
y1 = torch.log(a)            # y1 = log(a)
y2 = torch.sin(X2)           # y2 = sin(X2)
w = y1 * y2                  # w = y1 * y2
z = w                        # z = w

a.retain_grad()
y1.retain_grad()
y2.retain_grad()
w.retain_grad()
z.retain_grad()

print("Forward: z =", z.item())
z.backward()
print(f"dz/dz = 1 (输出对自己的梯度永远是1) -> {z.grad.item()}")
print(f"dz/dw = dz/dz * ∂z/∂w = 1 * 1 = {w.grad.item()}")
print(f"dz/dy1 = dz/dw * ∂w/∂y1 = {w.grad.item()} * y2 = {w.grad.item()} * {y2.item()} = {y1.grad.item()}")
print(f"dz/dy2 = dz/dw * ∂w/∂y2 = {w.grad.item()} * y1 = {w.grad.item()} * {y1.item()} = {y2.grad.item()}")
print(f"dz/da  = dz/dy1 * ∂y1/∂a = {y1.grad.item()} * (1/a) = {y1.grad.item()} * (1/{a.item()}) = {a.grad.item()}")
print(f"dz/dX1 = dz/da * ∂a/∂X1 = {a.grad.item()} * X2 = {a.grad.item()} * {X2.item()} = {X1.grad.item()}")
print(f"dz/dX2 = dz/da * ∂a/∂X2 + dz/dy2 * ∂y2/∂X2\n"
      f"       = {a.grad.item()} * X1 + {y2.grad.item()} * cos(X2)\n"
      f"       = {a.grad.item()} * {X1.item()} + {y2.grad.item()} * {torch.cos(X2).item()}\n"
      f"       = {X2.grad.item()}")
```

输出结果为：
```python
Forward: z = 0.2528530955314636
dz/dz = 1 (输出对自己的梯度永远是1) -> 1.0
dz/dw = dz/dz * ∂z/∂w = 1 * 1 = 1.0
dz/dy1 = dz/dw * ∂w/∂y1 = 1.0 * y2 = 1.0 * 0.14112000167369843 = 0.14112000167369843
dz/dy2 = dz/dw * ∂w/∂y2 = 1.0 * y1 = 1.0 * 1.7917594909667969 = 1.7917594909667969
dz/da  = dz/dy1 * ∂y1/∂a = 0.14112000167369843 * (1/a) = 0.14112000167369843 * (1/6.0) = 0.023520000278949738
dz/dX1 = dz/da * ∂a/∂X1 = 0.023520000278949738 * X2 = 0.023520000278949738 * 3.0 = 0.07056000083684921
dz/dX2 = dz/da * ∂a/∂X2 + dz/dy2 * ∂y2/∂X2
       = 0.023520000278949738 * X1 + 1.7917594909667969 * cos(X2)
       = 0.023520000278949738 * 2.0 + 1.7917594909667969 * -0.9899924993515015
       = -1.7267885208129883
```

对于计算图就是对于你的输入数据进行了那种计算方式进行记录，后续梯度反向传播时候通过上面计算图（**计算图保存了所有中间变量和梯度信息**）来计算梯度更新参数，那么进一步了解一下这些概念与显存的分析，运行过程中数据的显存占用主要如下几个部分：1、数据本身显存占用；2、数据中间激活（计算图）显存占用。对于这两部分可以直接通过[checkpoint](https://www.big-yellow-j.top/posts/2025/01/03/DistributeTraining.html#:~:text=%E8%A1%A5%E5%85%851%EF%BC%9Agradient%2Dcheckpoint%E6%96%B9%E6%B3%95)以及改变精度来减小显存占用。结果在后续计算中不再需要梯度，可以直接使用 `.detach()` 将其从计算图中分离，以减少显存占用。
### torch.compile
**值得注意的是**：在torch>2.0之后引入一个新的概念 `torch.compile`[^3] 在传统的计算过程中，如 `x+y`那么pytorch就会执行Python 解释器调用函数、检查类型、分配内存、调用 GPU/CPU 操作等操作，这样以来过程就会比较慢，在compile中过程是（From ChatGPT）：

**第一步：首先通过TorchDynamo —— “动态录音机”（抓图）**
当你第一次运行被 torch.compile 装饰的函数时，Dynamo 会“偷偷”接管 Python 的执行。 它不是静态看代码，而是一边模拟运行，一边录音： 把所有 PyTorch 操作（加、乘、卷积、ReLU 等）记录下来，画成一张 FX Graph（一张计算流程图）。 python 的普通代码（if 判断、for 循环、打印等）如果太复杂，就产生 Graph Break（图断开），这部分还是用原来的慢方式运行。 它还会记录“假设”：比如输入 tensor 的形状是 [32, 3, 224, 224]、类型是 float32 等。这些假设叫 Guards（守卫）。 为什么动态录音？因为 PyTorch 代码经常有动态形状、控制流，静态分析太难了。

**第二步：AOTAutograd —— “提前准备反向传播”**
如果是训练（需要 backward），Dynamo 只抓了前向（forward）。 AOTAutograd 会提前从前向图生成反向图（不用等到真正做 backward 时才临时建图）。 它还会把复杂操作分解成更基础的操作（PrimTorch），让后续优化更容易。 好处：前向+反向可以一起优化，节省内存（不用保存所有中间结果）。

**第三步：TorchInductor（默认后端）—— “优化工厂 + 代码生成器”**
拿到干净的计算图后，Inductor 开始大改造： 融合操作：把能合并的算子合成一个内核（例如 conv + batchnorm + relu 变成一个 GPU 内核，减少内存读写）。 布局优化、内存复用、循环优化等。 生成代码： GPU 上主要生成 Triton 代码（一种简单却高效的语言，比手写 CUDA 容易，性能接近官方）。 CPU 上生成 C++ 代码。
最后把这些优化好的内核打包成一个可直接调用的函数。从而实现计算加速，比如说简单的计算：
```python
import torch
def fun1(a, b):
    return a+b
fun_compile = torch.compile(fun1)

@torch.compile
def fun2(a, b):
    return a+b
```

[//]: # (测试代码为直接使用 `DPOTrainer`（代码：[ppo_trainer.py]&#40;https://github.com/shangxiaaabb/ProjectCode/blob/main/code/Python/RL-TRL/ppo_trainer.py&#41;）)

基本上只需要对涉及到计算的函数用 `torch.compile`处理即可（第一次编译速度比较慢，后续计算就快了），直接测试使用compile再模型训练过程中的表现，值得注意的是在使用Trl框架进行强化学习过程中，训练参数直接支持使用 `compile`（具体位置为：transformers/training_args.py，直接在DPOConfig中进行指定即可），如果是其它训练过程（假设使用accelerator框架进行）：
```python
if compile:
    s_compile_time = time.time()
    model = torch.compile(model, mode="reduce-overhead")
    accelerator.print(f"Compile Time: {time.time() - s_compile_time:.2f}s")
...
model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
```
在测试resnet50在CIFAR10数据集上表现如下（[代码](https://github.com/shangxiaaabb/ProjectCode/blob/main/code/Python/Pytorch-Learning/torch_compile.ipynb)）：
```python
# 使用compile
Compile Time: 0.89s
Epoch 00 | Train Time: 12.17s | Batch Time: 0.14764126466245067Train ACC: 11.19% | Test ACC: 11.89%
Epoch 05 | Train Time: 5.72s | Batch Time: 0.046339944917328506Train ACC: 15.49% | Test ACC: 15.54%
Epoch 10 | Train Time: 5.71s | Batch Time: 0.04499245176509935Train ACC: 18.26% | Test ACC: 19.50%
Epoch 15 | Train Time: 5.27s | Batch Time: 0.047159457693294604Train ACC: 20.47% | Test ACC: 21.51%
Epoch 20 | Train Time: 5.33s | Batch Time: 0.044611001501278Train ACC: 22.09% | Test ACC: 23.85%
Epoch 25 | Train Time: 5.30s | Batch Time: 0.04631211319748236Train ACC: 23.77% | Test ACC: 24.83%
Epoch 30 | Train Time: 5.57s | Batch Time: 0.04546777082949269Train ACC: 25.45% | Test ACC: 27.15%
Epoch 35 | Train Time: 6.04s | Batch Time: 0.04685194638310647Train ACC: 26.64% | Test ACC: 27.61%
Epoch 40 | Train Time: 5.78s | Batch Time: 0.04502055109763632Train ACC: 27.74% | Test ACC: 29.68%
Epoch 45 | Train Time: 5.89s | Batch Time: 0.04582952966495436Train ACC: 29.10% | Test ACC: 30.91%
Epoch 50 | Train Time: 5.03s | Batch Time: 0.046522539489123285Train ACC: 30.10% | Test ACC: 31.79%
Epoch 55 | Train Time: 5.26s | Batch Time: 0.045492488510754645Train ACC: 31.21% | Test ACC: 33.37%
Epoch 60 | Train Time: 5.41s | Batch Time: 0.04656180556939573Train ACC: 32.18% | Test ACC: 34.68%
Epoch 65 | Train Time: 5.60s | Batch Time: 0.04664086322395169Train ACC: 33.27% | Test ACC: 35.17%
Epoch 70 | Train Time: 5.81s | Batch Time: 0.04666753691069934Train ACC: 34.34% | Test ACC: 35.94%
Epoch 75 | Train Time: 5.69s | Batch Time: 0.04741260956744758Train ACC: 35.30% | Test ACC: 37.51%
Epoch 80 | Train Time: 5.99s | Batch Time: 0.049882051896075814Train ACC: 35.83% | Test ACC: 38.80%
Epoch 85 | Train Time: 5.60s | Batch Time: 0.04556511859504544Train ACC: 37.01% | Test ACC: 39.24%
Epoch 90 | Train Time: 5.70s | Batch Time: 0.04652732245776118Train ACC: 37.74% | Test ACC: 40.30%
Epoch 95 | Train Time: 5.66s | Batch Time: 0.04582754933104223Train ACC: 38.30% | Test ACC: 41.03%

# 不使用compile
Epoch 00 | Train Time: 5.71s | Batch Time: 0.05050786174073511Train ACC: 11.29% | Test ACC: 11.41%
Epoch 05 | Train Time: 5.25s | Batch Time: 0.047670155155415436Train ACC: 15.84% | Test ACC: 16.45%
Epoch 10 | Train Time: 6.11s | Batch Time: 0.04854408575564015Train ACC: 18.56% | Test ACC: 19.64%
Epoch 15 | Train Time: 6.33s | Batch Time: 0.04916523427379375Train ACC: 20.40% | Test ACC: 22.29%
Epoch 20 | Train Time: 6.09s | Batch Time: 0.051163284146055886Train ACC: 22.42% | Test ACC: 23.61%
Epoch 25 | Train Time: 5.79s | Batch Time: 0.04896752201780981Train ACC: 23.90% | Test ACC: 25.22%
Epoch 30 | Train Time: 5.78s | Batch Time: 0.05085692113759566Train ACC: 25.14% | Test ACC: 26.61%
Epoch 35 | Train Time: 5.12s | Batch Time: 0.04880750422575036Train ACC: 26.75% | Test ACC: 28.13%
Epoch 40 | Train Time: 5.38s | Batch Time: 0.04720686406505351Train ACC: 27.84% | Test ACC: 29.21%
Epoch 45 | Train Time: 5.10s | Batch Time: 0.047691140856061666Train ACC: 28.90% | Test ACC: 30.45%
Epoch 50 | Train Time: 5.18s | Batch Time: 0.046876946274115115Train ACC: 29.81% | Test ACC: 31.63%
Epoch 55 | Train Time: 6.57s | Batch Time: 0.04712670676562251Train ACC: 31.35% | Test ACC: 33.11%
Epoch 60 | Train Time: 6.29s | Batch Time: 0.048398008151930204Train ACC: 32.10% | Test ACC: 34.14%
Epoch 65 | Train Time: 6.31s | Batch Time: 0.047911264458481144Train ACC: 33.10% | Test ACC: 34.88%
Epoch 70 | Train Time: 5.41s | Batch Time: 0.049120401849552076Train ACC: 34.41% | Test ACC: 36.21%
Epoch 75 | Train Time: 5.09s | Batch Time: 0.04761015152444645Train ACC: 35.20% | Test ACC: 37.11%
Epoch 80 | Train Time: 5.44s | Batch Time: 0.04731477523336605Train ACC: 36.20% | Test ACC: 38.30%
Epoch 85 | Train Time: 6.00s | Batch Time: 0.048004476391539284Train ACC: 36.92% | Test ACC: 39.01%
Epoch 90 | Train Time: 5.05s | Batch Time: 0.04837274064823073Train ACC: 37.71% | Test ACC: 39.21%
Epoch 95 | Train Time: 5.30s | Batch Time: 0.04843420398478605Train ACC: 38.82% | Test ACC: 40.20%
```
从上述结果上看，最后ACC差异不大，但是在每个epoch以及batch_time上还是有差异的。

[//]: # (同时对比DPOTrainer（Qwen2-0.5B+vicgalle/OpenHermesPreferences-roleplay数据集）不同表现如下：)

**建议**：通过上述结果对比发现模型训练可以提前去开启compile
## torch数据形状改变方式
在`torch`中涉及到数据形状改变函数，总结如下：

| 方法          | 功能描述                                               | 是否拷贝数据 | 注意事项 |
|---------------|--------------------------------------------------------|--------------|----------|
| `reshape()`   | 返回指定形状的新张量，可能会返回原数据的视图，也可能复制数据 | 视情况而定   | 当原张量在内存中不连续时会复制数据 |
| `view()`      | 返回与原数据共享内存的新张量，形状可变                   | 否           | 仅适用于内存连续的张量，否则需先 `.contiguous()` |
| `unsqueeze()` | 在指定维度插入一个大小为 1 的维度                        | 否           | 常用于增加 batch 维度或通道维度 |
| `squeeze()`   | 删除大小为 1 的维度                                     | 否           | 默认删除所有为 1 的维度，可指定 `dim` |
| `expand()`    | 扩展张量的某个维度，不复制数据，使用广播                 | 否           | 扩展的维度只能是 1，否则报错；共享内存需注意修改风险 |
| `expand_as()` | 将张量扩展为与另一个张量形状相同                         | 否           | 同 `expand()`，但形状由另一张量决定 |
| `transpose()` | **交换**两个维度位置                                        | 否           | 常用于矩阵转置，返回视图 |
| `permute()`   | 按指定顺序**重新排列**所有维度                              | 否           | 返回视图，但会改变 strides |
| `contiguous()`| 返回一个内存连续的张量                                  | 是（如必要） | 常与 `view()` 搭配使用 |
| `clone()`     | 复制张量数据并返回一个新张量                            | 是           | 独立内存，与原张量不共享存储 |
| `detach()`    | 返回与原数据共享内存但不参与计算图的张量                 | 否           | 常用于切断梯度计算链条 |

## torch多进程
代码：[https://github.com/shangxiaaabb/ProjectCode/blob/main/code/Python/DFDataBuild/instance_pipeline/instance_split.py](https://github.com/shangxiaaabb/ProjectCode/blob/main/code/Python/DFDataBuild/instance_pipeline/instance_split.py)
## model.train()、model.eval()、torch.no_grad()
`model.train()`：把整个模型设为训练模式；如 Dropout 开启、BatchNorm 用小批量统计并更新滑动均值/方差。不影响是否计算梯度。
`model.eval()`：把模型设为评估/推理模式；如 Dropout 关闭、BatchNorm 使用已累计的运行统计，不再更新。不影响是否计算梯度。
`torch.no_grad()`：在其上下文中关闭 autograd 记录，从而不构建计算图、不产生 .grad，省显存、加速前向。不改变模型里层的训练/评估行为
## 总结

## 参考
[^1]: https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/
[^2]: https://docs.pytorch.org/docs/stable/notes.html
[^3]: [https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
[^4]: [https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile)