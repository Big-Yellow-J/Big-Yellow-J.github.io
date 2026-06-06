import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from ray import tune

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from learning_distribute_training.torchDDP_config import RayBaseConfig
    from learning_distribute_training.torchDDP_ray import RayDDPTuner
except ModuleNotFoundError:
    from torchDDP_config import RayBaseConfig
    from torchDDP_ray import RayDDPTuner


@dataclass
class ResNet50RayConfig(RayBaseConfig):
    train_script: str = str((CURRENT_DIR / "ddp_resnet50.py").resolve())
    train_cwd: str = str(CURRENT_DIR)
    storage_path: str = str((PROJECT_DIR / "outputs" / "ray_results").resolve())

    experiment_name: str = "ray_tune_resnet50_cifar10_ddp"
    metric: str = "Eval/ACC"
    mode: str = "max"
    use_torchrun: bool = True
    num_samples: int = 20

    search_alg: str = "bayesopt"
    scheduler: str = "hyperband"
    scheduler_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "time_attr": "training_iteration",
            "max_t": 100,
            "reduction_factor": 3,
        }
    )

    nproc_per_node: int = 1       # 每个train过程使用卡数量
    max_concurrent_trials: int= 4 # 最大并发
    resources_per_trial: Dict[str, float] = field(default_factory=
                                                  lambda: {"CPU": 10, "GPU": 0.5}) # 每个进程占用GPU/CPU

    param_space: Dict[str, Any] = field(
        default_factory=lambda: {
            "learning_rate": tune.loguniform(1e-5, 5e-3),
            "batch_size": tune.choice([128, 256, 512]),
            "optim_name": tune.choice(["adamw", "adamw_8bit"]),
            "lr_scheduler": tune.choice(["cosine", "linear"]),
            "lr_warmup_steps": tune.uniform(0.0, 0.1),
            "mixed_precision": tune.choice(["bf16", "fp16", "fp32"]),
        }
    )
    fixed_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "project_name": "Training-ResNet50-RayTune",
            "epoch": 20,
            "checkpointing_steps": -1,
            "checkpoints_total_limit": 1,
            "disable_checkpointing": False,
        }
    )
    # 关闭 Ray 中间日志输出
    verbose: int = 0
    echo_subprocess_log: bool = False
    env_vars: Dict[str, str] = field(
        default_factory=lambda: {
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
            "RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS": "0",
        }
    )

def run_ray_tune_resnet50(config: ResNet50RayConfig | None = None):
    config = config or ResNet50RayConfig()
    os.makedirs(config.storage_path, exist_ok=True)
    results = RayDDPTuner(config).run()
    best = results.get_best_result(metric=config.metric, mode=config.mode)
    print("=" * 60)
    print(f"Best metric {config.metric}: {best.metrics.get(config.metric)}")
    print(f"Best config: {best.config}")
    print(f"Best path: {best.path}")
    print("=" * 60)
    return results

if __name__ == "__main__":
    # 不输出日志： CUDA_VISIBLE_DEVICES=0,1 python3 training/ray_ddp_resnet50.py
    # 输出日志：CUDA_VISIBLE_DEVICES=0,1 python3 ray_ddp_resnet50.py --search_alg bayesopt --scheduler hyperband
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_alg",type=str,default=None,help="random/optuna/hyperopt/bayesopt/bohb",)
    parser.add_argument("--scheduler",type=str,default=None,help="fifo/asha/median/hyperband/pbt/bohb",)
    parser.add_argument("--verbose", type=int, default=None, help="Ray verbosity, 0~3.")
    parser.add_argument("--show_trial_logs",action="store_true",help="打印每个 trial 的训练 stdout/stderr。",)
    args = parser.parse_args()

    ray_cfg = ResNet50RayConfig()
    if args.search_alg:
        ray_cfg.search_alg = args.search_alg
    if args.scheduler:
        ray_cfg.scheduler = args.scheduler
    if args.verbose is not None:
        ray_cfg.verbose = args.verbose
    if args.show_trial_logs:
        ray_cfg.echo_subprocess_log = True
    run_ray_tune_resnet50(ray_cfg)
