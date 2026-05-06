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

    checkpointing_steps: int = 0
    checkpoints_total_limit: int = 10
    resume_from_checkpoint: Optional[str] = ""

    backend: str = "nccl"
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
    lr_scheduler: str = "cosine"
    lr_warmup_steps: float = 0.1
    optim_name: str = "adamw"
    logging_steps: int = 10
    task_type: str = "llm"
    gradient_accumulation_steps: int = 1

    torch_compile: bool = True
    torch_profile: bool = True
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
        os.makedirs(self.output_dir, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)