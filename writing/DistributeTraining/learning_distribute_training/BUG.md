# MERaLiON-SER 模型训练兼容性修复说明

## 问题背景

MERaLiON-SER-v1 模型的自定义代码 (`modeling_ser_whisper_ecapa.py`) 存在两个导致训练失败的问题：

### 问题 1: Meta Tensor 错误
`LearnableMultiResolutionPooling.__init__` 使用 `torch.logspace()` + `int(k)`，新版 PyTorch 中 `from_pretrained` 阶段 tensor 在 meta device 上，调用 `.item()` 报错：
> RuntimeError: Tensor.item() cannot be called on meta tensors

### 问题 2: 模型无法训练（无梯度）
模型的 `forward` 方法被 `@torch.no_grad()` 装饰器包裹，导致所有输出没有 grad_fn，backward 失败：
> RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

### 问题 3: Gradient Checkpointing 不兼容
`SERWhisperECAPAForAudioClassification` 不支持 `gradient_checkpointing_enable`。

### 问题 4: 内存爆炸
原始代码用 `dataset.map` 预处理所有音频数据，导致一次性加载全部音频到内存。

### 问题 5: DDP 未使用参数报错（find_unused_parameters）
DDP 默认 `find_unused_parameters=False`，要求模型中所有参数都参与 loss 计算。但 MERaLiON-SER 模型 `forward` 返回的 dict 中除了 `logits`，还可能包含 `hidden_states`、`attentions` 等中间输出，部分参数（如 Whisper encoder 的某些层）只贡献给未被使用的输出，导致这些参数没有梯度，DDP 的 all-reduce 梯度同步失败：
> RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel` ...
> Parameter indices which did not receive grad for rank 1: 511 512 513 ... 610 ...

## 修复方案

### Fix 1: torch.logspace → 纯 Python（源文件修改）
```python
# 旧代码
kernel_sizes = [int(k) for k in torch.logspace(
    math.log10(min_kernel), math.log10(max_kernel), num_resolutions
)]
# 修复后（等价纯 Python，不依赖 tensor）
kernel_sizes = [
    int(10 ** (math.log10(min_kernel) + i * (math.log10(max_kernel) - math.log10(min_kernel)) / max(1, num_resolutions - 1)))
    for i in range(num_resolutions)
]
```

### Fix 2: 移除 @torch.no_grad()（源文件修改）
```python
# 旧代码
@torch.no_grad()
def forward(self, ...):
# 修复后
# PATCHED: removed @torch.no_grad() to enable training
def forward(self, ...):
```

### Fix 3: 禁用 Gradient Checkpointing（配置修改）
```python
gradient_checkpointing: bool = False  # MERaLiON-SER 不支持
```

### Fix 4: collate_fn 替代 dataset.map（代码重构）
数据只存路径和标签，音频由 `collate_fn` 在 DataLoader 取 batch 时实时加载，避免一次性加载上万条音频到内存。

### Fix 5: DDP 添加 find_unused_parameters + 关闭多余输出（代码修改）
**两处修改：**

**① `torchDDP_training.py` — DDP 包装时启用未使用参数检测：**
```python
# 旧代码
self.model = DDP(
    self.model,
    device_ids=[self.local_rank] if self.device.type == "cuda" else None,
    output_device=self.local_rank if self.device.type == "cuda" else None,
)
# 修复后
self.model = DDP(
    self.model,
    device_ids=[self.local_rank] if self.device.type == "cuda" else None,
    output_device=self.local_rank if self.device.type == "cuda" else None,
    find_unused_parameters=True,  # 允许部分参数不参与 loss 计算
)
```

**② `ddp_meralionser.py` — compute_loss / evaluate 中关闭不必要的模型输出：**
```python
# 旧代码
outputs = self.model(
    input_features=batch.get("input_features"),
    attention_mask=batch.get("attention_mask"),
)
# 修复后（从源头减少未使用参数的产生）
outputs = self.model(
    input_features=batch.get("input_features"),
    attention_mask=batch.get("attention_mask"),
    output_hidden_states=False,   # 不计算 hidden_states
    output_attentions=False,      # 不计算 attentions
    return_dict=True,
)
```

## 涉及文件

| 文件 | 修改 |
|------|------|
| `ModelParameterCache/.../snapshots/<hash>/modeling_ser_whisper_ecapa.py` | Fix 1 + Fix 2（源头修复） |
| `torchDDP_training.py` | Fix 5（DDP `find_unused_parameters=True`） |
| `ddp_meralionser.py` | Fix 3 + Fix 4 + Fix 5 + `_patch_modeling_file()` 自动修复 |
| `meralion_ser_model.py` | 优先使用本地 snapshot 路径 |

## 自动化保护机制

`MeralionSERTrainer._patch_modeling_file()` 在每次加载前自动检测并修复 snapshot 文件：
- 检测 `torch.logspace` → 替换为纯 Python
- 检测 `@torch.no_grad()` 装饰器 → 移除
- 已修复则跳过（幂等操作）

## 运行命令

```bash
# DDP 训练
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 ddp_meralionser.py
```
