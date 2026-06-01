"""
Ray Tune — MERaLiON-SER DDP 双卡超参搜索 (GPU 4,5)

用法:
  pip install "ray[tune]" optuna
  export HF_ENDPOINT=https://hf-mirror.com && CUDA_VISIBLE_DEVICES=4,5 python ray_tune_meralion.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

RAY_STORAGE = "/home/huangjie/MdiriCode/CodeLearning/ModelTrainingResult/ray_tune_results"
TRAIN_SCRIPT = str(CURRENT_DIR / "ddp_meralionser.py")
NPROC = 2
GPU_PER_TRIAL = 2

SEARCH_SPACE = {
    "learning_rate":            tune.loguniform(1e-6, 5e-4),
    "batch_size":               tune.choice([32, 64, 128]),
    # "lora_r":                   tune.choice([8, 16, 32]),
    # "lora_alpha":               tune.choice([16, 32, 64]),
    # "lora_dropout":             tune.uniform(0.0, 0.3),
    "finetune_strategy":        tune.choice(["classifier_only"]),
    "gradient_accumulation_steps": tune.choice([1, 2, 4]),
    "lr_warmup_steps":          tune.uniform(0.01, 0.1),
}


def train_fn(trial_config: dict) -> None:
    """每个 trial 用 subprocess + torchrun 启动 DDP 双卡训练。"""
    print(f"[Trial] config: {json.dumps(trial_config, indent=2)}")
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["TRIAL_PARAMS"] = json.dumps(trial_config)
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(NPROC),
        TRAIN_SCRIPT,
    ]
    p = subprocess.run(cmd, env=env, cwd=str(CURRENT_DIR))
    if p.returncode != 0:
        raise RuntimeError(f"Trial failed, exit={p.returncode}")


def main():
    print("=" * 60)
    print("  Ray Tune — MERaLiON-SER DDP 双卡 (GPU 4,5)")
    print("=" * 60)

    ray.init(ignore_reinit_error=True)
    gpus = int(ray.cluster_resources().get("GPU", 1))
    concurrent = max(1, gpus // GPU_PER_TRIAL)
    print(f"[Ray] GPU={gpus}, GPU/trial={GPU_PER_TRIAL}, concurrent={concurrent}")

    analysis = tune.run(
        tune.with_resources(train_fn, resources={"CPU": 8, "GPU": GPU_PER_TRIAL}),
        config=SEARCH_SPACE,
        metric="metric_uar",
        mode="max",
        num_samples=15,
        max_concurrent_trials=concurrent,
        scheduler=ASHAScheduler(max_t=10, grace_period=3, reduction_factor=3),
        search_alg=OptunaSearch(metric="metric_uar", mode="max"),
        storage_path=RAY_STORAGE,
        name="meralion_ddp_tuning",
        verbose=1,
    )

    print("\n" + "=" * 60)
    print(f"  BEST: UAR={analysis.best_result.get('metric_uar')}")
    print(f"  Config: {analysis.best_config}")
    print(f"  Logdir: {analysis.best_logdir}")
    print("=" * 60)
    ray.shutdown()


if __name__ == "__main__":
    main()
