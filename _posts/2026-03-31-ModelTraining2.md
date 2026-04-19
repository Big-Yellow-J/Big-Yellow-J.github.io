---
layout: mypost
title: 🔥Pytorch使用-2：dataloader处理过程及模型训练性能分析
categories: pytorch
extMath: true
images: true
address: changsha
show_footer_image: true
tags:
- 模型训练
- Pytorh学习
description: PyTorch训练推理场景性能瓶颈分为CPU、GPU计算、I/O、多卡通信、框架开销五类，对应不同判别特征：GPU利用率波动跳变对应CPU瓶颈，训练起步慢后续提速对应I/O瓶颈，NCCL
  AllReduce耗时占比超30%对应多卡通信瓶颈。宏观可通过bpytop、nvidia-smi、iotop快速排查CPU、GPU、磁盘占用，微观借助torch.profiler的概览、算子、追踪等视图精确定位耗时节点。可调整Dataloader的num_workers、pin_memory等参数优化数据加载，超大数据用IterableDataset、WebDataset或Hugging
  Face流式加载，针对不同瓶颈可采取算子融合、混合精度训练、梯度压缩等方案优化。
---

在模型训练过程中，通过分析模型损失、准确率这些基础指标去判别模型优化效果，通过flash-attn、混合精度训练等去优化模型训练速度，但是训练过程中对于设备性能瓶颈分析似乎做的比较少，比如说CPU、GPU使用率等，下面内容系统分析一下如何去分析训练/推理过程中的性能瓶颈。在介绍工具使用之前首先了解在使用pytroch进行训练过程中设备之间处理顺序是什么：`磁盘 → 内存 → CPU → GPU（前向）→ GPU（反向）→ GPU（参数更新）→ 内存 → 磁盘（可选）`，一般而言对于数据处理（**主要是通过CPU进行数据处理，如数据增强等**），这个过程主要是 `磁盘 → 内存 → CPU`，而后就是将处理后的数据交给GPU进行计算。**训练过程中瓶颈分析**[^1]：

**CPU 瓶颈**比较好认。GPU 利用率像心电图一样上下跳动，高的时候在算，低的时候在等数据。htop 一看，CPU 某几个核打满了，其他的闲着，DataLoader 的 worker 数量没配对。**GPU 计算瓶颈**的表现是利用率高，但实际吞吐量低。这时候得看 MFU（Model FLOPs Utilization），如果 MFU 很低，说明 GPU 算力没被喂饱。可能是算子实现效率差，也可能是 kernel 太碎，调度开销太大。**I/O 瓶颈**有个很典型的症状：训练刚开始特别慢，跑几个 step 之后速度才上来。因为第一批数据要从磁盘读，后面的数据可能已经缓存到内存里了。iotop 一看，磁盘读写爆高，CPU 反而不怎么忙。**多卡训练的通信瓶颈**也好判断。看 nvidia-smi，某几张卡利用率明显比其他的低，它们在等梯度同步。在 profiler 里看 NCCL 相关操作，如果 AllReduce 的时间占到 30% 以上，就是通信在拖。还有个容易被忽略的：框架开销。Python 解释器本身、GIL 锁、过多的 Python 层函数调用，这些都会吃掉时间。在 `torch.profiler `的 CPU trace 里，如果看到大量时间花在 Python 调度上而不是实际计算上，就是这个问题。
> 绝大部分时间 kernel算子一般都是优化比较好的（除非你自己去写算子），绝大多数情况下优化dataloader过程基本可以满足需求

## dataloader过程
平时写代码过程中对于 `dataloader`过程处理比较简单：
```python
from torch.utils.data import DataLoader, Dataset
class CustomDataset(Dataset):
    def __init__(self, ...):
        ...
    def __len__(self):
        # 一般就是直接返回数据数量
        ...
    def __getitem__(self, idx):
        # 一般就是对数据进行处理如标准化等
        ...
        # 如果处理报错就可以直接去下一个数据处理
        # next_index = (index + 1) % len(self)
        # return self.__getitem__(next_index)
    def collate_fn(self, batches):
        batch_size = len(batches)
        # 解包 batches 在 __getitem__ 中返回什么就解包得到什么
        _, _ = zip(*batches)
train_dataset = CustomDataset(xxx)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,                  # 根据你的 GPU 显存调整，越大越好
    shuffle=True,
    num_workers=8,                  # 根据 CPU 核心数和实验调整（起始建议）
    pin_memory=True,                # 强烈推荐
    prefetch_factor=4,              # 可选，加速预取
    persistent_workers=True,        # 可选，推荐
    drop_last=True                  # 可选，避免小 batch
)
```
介绍dataloader原理之前先去看里面**参数含义**：1、**batch_size**（int）：一次处理多少数据；2、**shuffle**（bool）：是否对数据进行打乱（一般对val数据不打乱）；3、**sampler**：从样本中的采样策略；4、**num_workers**（int）：数据加载进程数量；5、**collate_fn**：对batch数据进行处理操作；6、**pin_memory_device**：指定 pin memory 操作的目标[设备](https://docs.pytorch.org/docs/stable/torch.html#accelerators)（通常配合 pin_memory=True 使用）；7、**pin_memory**（bool）：_数据提前复制到cuda显存中_。一般而言除去常用几个参数即使去调整 `pin_memory` 去加快数据加载速度，一般而言对于 `num_workers` 并非越大越好（习惯用8），可以通过实则调节而后通过 `torch.profile`去分析数据加载费时。

**这里再去介绍一些Dataloader原理**（图像来自[知乎](https://zhuanlan.zhihu.com/p/1936349147797120821)）：

![20260417195427](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image/20260417195427.png)

`Dataset` 是整个数据加载流程的基础。它提供了**一个索引访问接口**，主要定义了两个方法：`__len__()`：返回数据集总样本数，`__getitem__(index)`：**根据索引返回第 $i$ 个样本**（样本可能来自磁盘读取、内存加载，或实时生成）。Sampler 与 BatchSampler 的作用如下：如果没有手动指定 sampler，且 shuffle=True，DataLoader 会自动使用 RandomSampler，在每个 epoch 开始时对所有样本的索引进行一次性随机打乱。BatchSampler 则负责将打乱后的索引列表，按照 batch_size 分组，形成一个个 batch 的索引列表（例如 [ [45, 7, 23, 12], ... ]）。**当 num_workers > 0 时**，DataLoader 会启用多进程数据加载机制，具体流程如下：

主进程通过 BatchSampler 生成所有已打乱的 batch 索引列表。主进程将每个 batch 的索引（batch_indices）通过 index_queue 分发给不同的 worker 进程。**每个 worker 进程 独立完成**以下工作：接收一个 batch_indices（例如 [45, 7, 23, 12]）多次调用 dataset[i] 获取对应样本在 worker 进程内执行 collate_fn，**将多个样本组装成一个完整的 batch**将组装好的 batch 通过 worker_result_queue 发送回主进程，主进程接收到 batch 后，若设置了 pin_memory=True，则通过后台线程将其转换为 pinned memory（页锁定内存），最后将 batch 返回给训练循环使用。

对于 `Dataloader` 中可以做的修改内容不多，比如有比较常见的几种情况：1、我的数据（dataset）中我的数据有1T，设备处不了如何做？2、如何取定义自己的采样器？
**1、处理大批量数据**：一般在定义 dataset（pytorch中提供两种数据类型 `Map-Style Datasets` 以及 `Iterable-Style Datasets`） 过程中是直接通过 `__len__` 以及 `__getiem__` 去获取数据信息，如果数据有1T直接打开然后 `len` 势必导致问题，比较简单方式直接通过pytorch原生的 `IterableDataset` 进行处理将数据转化为数据流，除此之外还可以通过使用huggingface datasets中流式模式 `dataset = load_dataset("json", data_files="data/1T.jsonl", streaming=True)`，如果对于图像等可以直接使用 `WebDataset` 方式，以使用 `IterableDataset` 为例：
```python
import json
import torch
from torch.utils.data import IterableDataset, DataLoader

class IterableDatasetJsonl(IterableDataset):
  def __init__(self, file_path, file_type: str='jsonl', shard_rank=0, num_shards=1):
    self.file_path = file_path
    self.file_type = file_type
    self.shard_rank = shard_rank
    self.num_shards = num_shards

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
      shard_rank = self.shard_rank * worker_info.num_workers + worker_info.id
      num_shards = self.num_shards * worker_info.num_workers
    else:
      shard_rank, num_shards = self.shard_rank, self.num_shards

    with open(self.file_path, 'r', encoding='utf-8') as f:
      for i, line in enumerate(f):
        if i % num_shards == shard_rank:
          sample = json.loads(line)
          '''
          继续后处理
          '''
          yield sample
```
对于 `IterableDataset` 使用区别 datatset 最大差异就是直接在 iter 中对数据进行处理

**2、使用不同的采样器**，最常见的情况就是，1、对于图像进行分桶输入（比如有1024x1024也有1024x768），3、控制数据分布，如数据自身具有一定特征需要控制这些特征在一个batch中分布相对一致。
> 对于文本情况，如果需要输入 bs>1 一般而言直接定义 `collate_fn` 提前去对文本进行 padding 并且返回 mask（一般模型都可以接受mask作为输入），padding方式： `padded_input_ids = pad_sequence(truncated, batch_first=True, padding_value=pad_token_id)`（`from torch.nn.utils.rnn import pad_sequence `）

```python
import torch
import random
from collections import defaultdict
from torch.utils.data import DataLoader, Sampler, Dataset
class BucketBatchSampler(Sampler):
  def __init__(self, dataset, batch_size, drop_last=True, shuffle=True):
    self.dataset = dataset
    self.batch_size = batch_size
    self.drop_last = drop_last
    self.shuffle = shuffle

    self.buckets = defaultdict(list)
    # 直接去便利所有数据 在效率上存在一定欠缺
    for idx in range(len(dataset)):
      h, w = dataset.get_image_size(idx)
      self.buckets[(h, w)].append(idx)
    self.bucket_keys = list(self.buckets.keys())

  def __iter__(self):
    batches = []
    for key in self.bucket_keys:
      indices = self.buckets[key][:]
      if self.shuffle:
        random.shuffle(indices)

      for i in range(0, len(indices), self.batch_size):
        batch = indices[i:i + self.batch_size]
        if self.drop_last and len(batch) < self.batch_size:
          continue
        if batch:
          batches.append(batch)

    if self.shuffle:
      random.shuffle(batches)

    for batch in batches:
      yield batch

  def __len__(self):
    total = 0
    for indices in self.buckets.values():
      total += len(indices) // self.batch_size
      if not self.drop_last and len(indices) % self.batch_size != 0:
        total += 1
    return total
```
对于上述代码中在sampler中提前遍历了所有的数据并且根据分辨率进行分组（实际可能需要对图像进行分辨率计算-->根据计算预估分组） ，`__iter__`：定义如何生成采样顺序（在使用 `for batch in dataloader` 中就会使用这个方法），最后通过 `yield` 进行返回（因为 `__iter__` 必须返回一个迭代器也可以使用 `return`），`__len__`：返回 该采样器在当前配置下会产生多少个 batch（或多少个样本）

## 宏观指标分析
最简单分析方法直接在模型运行过程中使用 `time` 去记录时间就可以快速了解每一个阶段耗时统计，除此之外还可以直接基于linux（假设服务器为linux Ubuntu系统）的基础命令进行分析，主要是分析CPU内存使用情况、GPU使用情况、磁盘io使用情况。**CPU性能分析**，一般而言可以直接使用htop、top、bytop等工具直接去看，这里直接使用**bytop**工具进行性能分析，首先安装bytop[^2]（`pip3 install bpytop --upgrade` 或者直接使用 `sudo apt install bpytop`），而后就可以直接终端使用 `bpytop`就可以看到各项性能分析
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260401153039800.png)
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260401153015796.png)
使用方法比较简单直接通过数字选择（直接键盘输入数字）需要看到的面板：
```
1：显示/关闭 CPU性能分析
2：显示/关闭 内存/存储性能分析
3：显示/关闭 网络分析
4：显示/关闭 各项进程进行分析
```
首先通过上述 `bpytop`就可以简单了解各项进程上在内存上使用情况如何、CPU使用情况如何。**GPU性能分析**，对于GPU性能分析最简单工具直接使用 `watch -n 0.1 nvidia-smi` 每0.1s刷新nvidia-smi情况，主要是去看GPU利用率、显存占用情况
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260401153751296.png)
**值得注意的是**，**有些时候即使将所有的在跑的程序都关闭但是发现显存还是被占用（利用率是0）**[^3]使用`ps -ef`命令
![](https://www.autodl.com/docs/qa4.assets/image-20220713171325500.png)
可以看到PID、PPID、CMD 3列重要信息，分别是进程ID、父进程ID、进程的启动命令。通过命令可以判断哪些进程是自己程序启动的进程，比如上方的python train.py就是我启动的进程，其他的均为系统进程或无关显存占用的进程。接下来杀死进程：从截图中看到python train.py程序的进程ID是594 和797，那么可以使用`kill -9 594 797`命令来结束进程。 

但是常常占用显存的进程会很多，特别是在多卡并行时，按此方法会比较繁琐，以下介绍一种更强大的方式结束进程：通过`ps -ef`能看出，我自己的进程都包含了train关键字（并且其他无关的系统进程没有包含，防止误杀），那么使用grep命令可以过滤出我自己的进程，例如：
![](https://www.autodl.com/docs/qa4.assets/image-20220713172143285.png)
接下来是获取进程的ID，此时可以使用awk命令，awk命令用法复杂，这里简单记住以下命令即可：
![](https://www.autodl.com/docs/qa4.assets/image-20220713172301267.png)
最后再通过kill命令，即可完整的结束进程。完整命令为`ps -ef | grep train | awk '{print $2}' | xargs kill -9`
![](https://www.autodl.com/docs/qa4.assets/image-20220713172428298.png)
以上输出中会多出来一个No such process的错误，可以忽略，出现原因是grep train也会产生一个进程，被自己过滤出来。
## 微观指标分析
上面介绍了宏观指标去看CPU/GPU/磁盘/内存之间的使用情况，最好的情况就是这几项的指标都要上去保证在一个较好的情况下，下面进一步介绍更加微观的指标
### 基于torch profiler分析
直接使用torch原生工具[^4]进行性能分析可以帮助我们分析和优化模型的执行时间、GPU 利用率、内存带宽等性能指标。通过 torch.profiler，你可以了解每一层模型在设备上的执行情况，分析 GPU 资源的利用率（**再了解每一块的耗时之后就可以直接再去争对耗时较长的内容进一步分析优化了**），具体代码测试过程中使用方法比较简单：
```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
# 首先初始化 profile 
# 如果要使用tensorboard需要额外安装 pip install torch-tb-profiler
...

self.accelerator = Accelerator(...,log_with='tensorboard',project_dir=args.log_dir)
...
log_root = self.args.log_dir
if self.accelerator.is_main_process:
    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,      # 等待步数
            warmup=1,    # 预热步数
            active=3,    # 活跃步数
            repeat=2     # 重复次数
        ),
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,            # 记录张量形状
        profile_memory=True,           # 记录内存使用
        with_stack=True,               # 记录调用栈
        with_flops=True                # 计算 FLOPs
    )
    profiler.start()
...
for epoch in range(1, self.num_epochs + 1):
    for batch_idx, (images, labels) in enumerate(self.train_loader):
        ...
        if profiler and self.accelerator.is_main_process:
            profiler.step()
if prof:
    profiler.export_chrome_trace("trace.json")
    prof.stop()
```
torch profile使用比较简单就是先初始化而后`start()`启动记录器、`step()`记录结果、`stop()`停止记录，而后直接通过 `tensorboard --logdir logs/` 即可（ **值得注意的是**，上面代码只会记录少数步，当 `repeat=0`时候就会一直记录，不需要频繁记录那么多），上述过程中需要注意tensorboard和profile的存储的最终的文件夹要保持一致，对于启动后的在tensorboard中视图中各项结果分析如下[^5]：
**Overview（概览）**：这个页面能帮你快速判断性能瓶颈在哪。
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260401215011996.png)
主要关注红框中内容，它会将每个Step（迭代）的时间拆分成 **Kernel**（计算）、**Memcpy**（数据传输）、**Memset**（GPU内存设置时间）、**DataLoader**（数据加载） 和 **CPU Exec**（CPU计算） 等几部分。如果"Kernel"占比低而"DataLoader"很高，说明数据加载是瓶颈；如果"CPU Exec"很高，则说明CPU侧的算子或逻辑存在优化空间。
**Operator（算子）**：这个表格是所有PyTorch操作（如aten::convolution）的性能数据。
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260401215618602.png)

主要关注红框中内容，Calls（运行过程中被使用次数）、Device xxx Duration（在 GPU 上花费的累计时间）、Host xxx Duration（在主机上花费时间），分析过程中主要是去更具耗时最长的就是优化重点。如果开启了with_stack=True，点击"Call Stack"还能直接跳转到你代码中调用该算子的位置
**Trace（追踪）**：这个时间线视图最直观，能让你看到每个算子和CUDA Kernel的精确起止时间。使用方法：在Chrome浏览器打开 chrome://tracing，然后加载生成的JSON文件。或者直接在TensorBoard的Trace页面分析。你可以通过鼠标滚轮缩放，并利用右上角的 Flow Events 按钮，查看是哪个CPU算子启动了一个GPU Kernel，这对于定位CUDA Kernel的启动延迟问题非常有帮助。

**Memory（内存）**：这个视图展示了内存随时间的分配和释放情况，帮你发现内存泄漏或不必要的显存占用。
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260401220618523.png)

**Kernel（内核）**：这是GPU上执行的底层函数视图。
![](https://ghfast.top/https://raw.githubusercontent.com/Big-Yellow-J/BlogImage/main/image20260401220329364.png)

主要是去查看GPU利用率（GPU Utilization）、SM效率（Est. SM Efficiency）以及Tensor Core的使用情况。如果这些指标偏低，说明GPU并没有被充分利用。
## 调节参数优化
CPU 瓶颈（一般就去修改数据处理过程，如数据增强等操作）可以直接调 num_workers、增大 prefetch_factor、把预处理卸载到 GPU等处理操作。
```python
DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
```
I/O 瓶颈就换 SSD、预加载数据到内存、用 NVIDIA DALI。GPU 计算瓶颈就做算子融合、上 torch.compile、开混合精度。通信瓶颈就上梯度压缩、通信计算重叠。框架开销就减少 Python 调用、用 TorchScript 或者 C++ 扩展
## 参考
[^1]: [https://www.zhihu.com/question/1927112862976972744/answer/2016593596803986385?utm_psn=2022387930946126849](https://www.zhihu.com/question/1927112862976972744/answer/2016593596803986385?utm_psn=2022387930946126849)
[^2]: [https://github.com/aristocratos/bpytop](https://github.com/aristocratos/bpytop)
[^3]: [https://www.autodl.com/docs/qa4/](https://www.autodl.com/docs/qa4/)
[^4]: [https://pytorch-cn.com/tutorials/recipes/recipes/profiler_recipe.html](https://pytorch-cn.com/tutorials/recipes/recipes/profiler_recipe.html)
[^5]: [https://github.com/pytorch/kineto/blob/main/tb_plugin/README.md](https://github.com/pytorch/kineto/blob/main/tb_plugin/README.md)
[^6]: [https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
[^7]: [https://docs.pytorch.org/docs/stable/data.html#map-style-datasets](https://docs.pytorch.org/docs/stable/data.html#map-style-datasets)
[^8]: [https://www.aidoczh.com/speechbrain/tutorials/advanced/data-loading-for-big-datasets-and-shared-filesystems.html](https://www.aidoczh.com/speechbrain/tutorials/advanced/data-loading-for-big-datasets-and-shared-filesystems.html)