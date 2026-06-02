import os
import random
from datetime import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

@dataclass
class BasicConfig:
    cache_dir: str = "./outputs"
    store_dir: str = "./outputs"
    project_name: str = "Training-BasicConfig"
    current_date: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d"))
    special_num: int = field(default_factory=lambda: random.randint(0, 9999))
    tracker_project_name: str = field(init=False)
    output_dir: str = field(init=False)

    evaluate_epochs: int = 1 # 每多少个 epoch 进行一次模型评估，<0 表示不进行评估
    checkpointing_steps: int = 0
    checkpoints_total_limit: int = 10
    resume_from_checkpoint: Optional[str] = ""

    backend: str = "nccl"
    distributed_strategy: str = "ddp"  # "ddp" | "fsdp" | "fsdp2"
    seed: int = 10086
    epoch: int = 10
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None
    train_dataset_shuffle: bool = True
    eval_dataset_shuffle: bool = False

    max_train_steps: int = 0
    learning_rate: float = 2e-5
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict | None = None
    lr_scheduler: str = "cosine"
    lr_warmup_steps: float = 0.1
    optim_name: str = "adamw"
    logging_steps: int = 10
    task_type: str = "llm"
    gradient_accumulation_steps: int = 1

    # FSDP options
    fsdp_sharding_strategy: str = "FULL_SHARD"  # "FULL_SHARD" | "SHARD_GRAD_OP" | "NO_SHARD"
    fsdp_use_orig_params: bool = True
    fsdp_limit_all_gathers: bool = True
    fsdp_sync_module_states: bool = False

    # Best model checkpoint: save when eval metric improves
    # metric_name: 对应 evaluate() 返回 dict 中的 key，如 "Eval/eval_loss" / "Eval/accuracy"
    # metric_mode: "min" 表示越小越好（loss），"max" 表示越大越好（accuracy）
    # 设为 None 或空字符串则关闭最佳模型保存
    best_metric_name: str = ""
    best_metric_mode: str = "min"  # "min" | "max"

    torch_compile: bool = False
    torch_profile: bool = False
    compile_config: Dict[str, Any] = field(default_factory=lambda: {
        "backend": "inductor", "mode": "default"
    })

    # logging: "tensorboard" | "wandb" | "tensorboard+wandb" | "none"
    log_with: str = "tensorboard"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"

    def __post_init__(self) -> None:
        self.tracker_project_name = (
            f"{self.current_date}-{self.project_name}-{self.special_num:04d}"
        )
        self.output_dir = os.path.join(self.store_dir, self.tracker_project_name)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RayBaseConfig:
    # 训练入口与运行方式
    train_script: str = ""
    train_cwd: Optional[str] = None
    python_executable: str = "python3"
    use_torchrun: bool = True
    nproc_per_node: int = 1

    # Ray 运行配置
    experiment_name: str = "ray_ddp_tune"
    storage_path: str = "./outputs/ray_results"
    local_dir: Optional[str] = None
    seed: int = 42
    num_samples: int = 10
    max_concurrent_trials: Optional[int] = None
    resources_per_trial: Dict[str, float] = field(
        default_factory=lambda: {"CPU": 4, "GPU": 1}
    )
    verbose: int = 1
    fail_fast: bool = False

    # 优化目标
    metric: str = "metric"
    mode: str = "max"  # "max" | "min"

    # 搜索空间与固定参数（固定参数会与 trial 参数合并）
    param_space: Dict[str, Any] = field(default_factory=dict)
    fixed_params: Dict[str, Any] = field(default_factory=dict)

    # 搜索算法:
    # "random" | "optuna" | "hyperopt" | "bayesopt" | "bohb"
    search_alg: str = "random"
    search_alg_kwargs: Dict[str, Any] = field(default_factory=dict)

    # 调度器:
    # "fifo" | "asha" | "median" | "hyperband" | "pbt" | "bohb"
    scheduler: str = "asha"
    scheduler_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"max_t": 50, "grace_period": 5, "reduction_factor": 3}
    )

    # 环境变量注入
    env_vars: Dict[str, str] = field(default_factory=dict)

    # 自定义 Ray init 参数，例如 {"address": "auto"}
    ray_init_kwargs: Dict[str, Any] = field(default_factory=dict)

    # trial 完成后读取 metric 的文件名（位于 trial 目录）
    result_filename: str = "ray_result.json"

    # 是否把 trial 的 stdout/stderr 回传到 ray 日志
    echo_subprocess_log: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
