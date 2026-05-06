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

    seed: int = 10086
    epoch: int = 10
    batch_size: int = 1
    max_train_steps: int = 0
    learning_rate: float = 2e-5
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    lr_warmup_steps: float = 0.1
    optim_name: str = "adamw"
    logging_steps: int = 10
    task_type: str = "llm"  # llm | classification

    torch_compile: bool = False
    torch_profile: bool = False
    compile_config: Dict[str, Any] = field(default_factory=lambda: {
        "backend": "inductor", "mode": "default"
    })
    accelerator_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "mixed_precision": "bf16",
            "log_with": "tensorboard",
            "project_config": None,
        }
    )
    deepspeed_plugin: Dict[str, Any] = field(default_factory=dict)
    fsdp2_plugin: Dict[str, Any] = field(default_factory=dict)
    gradient_plugin: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_steps": 1,
        }
    )

    def __post_init__(self) -> None:
        self.tracker_project_name = f"{self.current_date}-{self.project_name}-{self.special_num:04d}"
        self.output_dir = os.path.join(self.store_dir, self.tracker_project_name)
        os.makedirs(self.store_dir, exist_ok=True)

    @property
    def gradient_accumulation_steps(self) -> int:
        return int(self.gradient_plugin.get("num_steps", 1))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
