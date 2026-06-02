import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import ray
from ray import tune
from ray.air import FailureConfig, RunConfig
from ray.tune import TuneConfig

from torchDDP_config import RayBaseConfig

class RayDDPTuner:
    def __init__(self, config: RayBaseConfig):
        if not config.train_script:
            raise ValueError("RayBaseConfig.train_script 不能为空。")
        self.config = config

    def _build_search_alg(self):
        name = self.config.search_alg.lower()
        kwargs = dict(self.config.search_alg_kwargs)
        kwargs.setdefault("metric", self.config.metric)
        kwargs.setdefault("mode", self.config.mode)

        if name == "random":
            return None
        if name == "optuna":
            from ray.tune.search.optuna import OptunaSearch

            return OptunaSearch(**kwargs)
        if name == "hyperopt":
            from ray.tune.search.hyperopt import HyperOptSearch

            return HyperOptSearch(**kwargs)
        if name == "bayesopt":
            from ray.tune.search.bayesopt import BayesOptSearch

            return BayesOptSearch(**kwargs)
        if name == "bohb":
            from ray.tune.search.bohb import TuneBOHB

            return TuneBOHB(**kwargs)
        raise ValueError(f"Unsupported search_alg: {self.config.search_alg}")

    def _build_scheduler(self):
        name = self.config.scheduler.lower()
        kwargs = dict(self.config.scheduler_kwargs)

        if name == "fifo":
            return None
        if name == "asha":
            from ray.tune.schedulers import ASHAScheduler

            return ASHAScheduler(**kwargs)
        if name == "median":
            from ray.tune.schedulers import MedianStoppingRule

            return MedianStoppingRule(**kwargs)
        if name == "hyperband":
            from ray.tune.schedulers import HyperBandScheduler

            return HyperBandScheduler(**kwargs)
        if name == "pbt":
            from ray.tune.schedulers import PopulationBasedTraining

            return PopulationBasedTraining(**kwargs)
        if name == "bohb":
            from ray.tune.schedulers import HyperBandForBOHB

            return HyperBandForBOHB(**kwargs)
        raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")

    def _build_cmd(self) -> list[str]:
        script = str(Path(self.config.train_script).expanduser().resolve())
        if self.config.use_torchrun:
            return [
                self.config.python_executable,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                str(self.config.nproc_per_node),
                script,
            ]
        return [self.config.python_executable, script]

    @staticmethod
    def _parse_eval_dict_from_log(log_text: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        pattern = re.compile(r"eval metrics:\s*(\{.*?\})", re.IGNORECASE | re.DOTALL)
        for m in pattern.finditer(log_text):
            raw = m.group(1)
            try:
                obj = ast.literal_eval(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                for k, v in obj.items():
                    try:
                        metrics[str(k)] = float(v)
                    except Exception:
                        continue
        return metrics

    def _read_metric_from_file(self, trial_dir: str) -> Dict[str, float]:
        result_path = os.path.join(trial_dir, self.config.result_filename)
        if not os.path.isfile(result_path):
            return {}
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            out: Dict[str, float] = {}
            for k, v in data.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    continue
            return out
        return {}

    def _extract_metrics(self, stdout: str, trial_dir: str) -> Dict[str, float]:
        by_file = self._read_metric_from_file(trial_dir)
        if self.config.metric in by_file:
            return by_file
        by_log = self._parse_eval_dict_from_log(stdout)
        merged = {**by_log, **by_file}
        return merged

    def _trainable(self, trial_params: Dict[str, Any]) -> None:
        merged_params = {**self.config.fixed_params, **trial_params}
        merged_params.setdefault("seed", self.config.seed)
        trial_dir = tune.get_context().get_trial_dir()

        env = os.environ.copy()
        env.update(self.config.env_vars)
        env["PYTHONHASHSEED"] = str(self.config.seed)
        env["TRIAL_PARAMS"] = json.dumps(merged_params, ensure_ascii=False)
        env["RAY_TRIAL_DIR"] = trial_dir
        env["RAY_RESULT_FILE"] = os.path.join(trial_dir, self.config.result_filename)

        cmd = self._build_cmd()
        run_cwd = (
            str(Path(self.config.train_cwd).expanduser().resolve())
            if self.config.train_cwd
            else str(Path(self.config.train_script).expanduser().resolve().parent)
        )

        proc = subprocess.run(
            cmd,
            cwd=run_cwd,
            env=env,
            capture_output=True,
            text=True,
        )

        if self.config.echo_subprocess_log:
            if proc.stdout:
                print(proc.stdout)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)

        if proc.returncode != 0:
            raise RuntimeError(
                f"Trial failed with exit={proc.returncode}. "
                f"stderr tail: {proc.stderr[-1000:] if proc.stderr else 'N/A'}"
            )

        metrics = self._extract_metrics(proc.stdout or "", trial_dir)
        if self.config.metric not in metrics:
            raise RuntimeError(
                f"Metric '{self.config.metric}' not found. "
                f"Please write {self.config.result_filename} in trial dir or print eval metrics dict."
            )
        tune.report(**metrics)

    def run(self):
        ray.init(ignore_reinit_error=True, **self.config.ray_init_kwargs)
        try:
            trainable = tune.with_resources(
                self._trainable,
                resources=self.config.resources_per_trial,
            )
            tuner = tune.Tuner(
                trainable,
                param_space=self.config.param_space,
                tune_config=TuneConfig(
                    metric=self.config.metric,
                    mode=self.config.mode,
                    num_samples=self.config.num_samples,
                    max_concurrent_trials=self.config.max_concurrent_trials,
                    search_alg=self._build_search_alg(),
                    scheduler=self._build_scheduler(),
                ),
                run_config=RunConfig(
                    name=self.config.experiment_name,
                    storage_path=self.config.local_dir or self.config.storage_path,
                    verbose=self.config.verbose,
                    failure_config=FailureConfig(fail_fast=self.config.fail_fast),
                ),
            )
            return tuner.fit()
        finally:
            ray.shutdown()


def build_default_config() -> RayBaseConfig:
    """默认示例配置，可直接复制后按任务覆写。"""
    return RayBaseConfig(
        train_script="./training/ddp_meralionser.py",
        experiment_name="meralion_ddp_tune",
        metric="Eval/uar",
        mode="max",
        nproc_per_node=2,
        resources_per_trial={"CPU": 8, "GPU": 2},
        num_samples=20,
        search_alg="optuna",
        scheduler="asha",
        param_space={
            "learning_rate": tune.loguniform(1e-6, 5e-4),
            "batch_size": tune.choice([16, 32, 64]),
            "gradient_accumulation_steps": tune.choice([1, 2, 4]),
            "lr_warmup_steps": tune.uniform(0.01, 0.2),
        },
    )


if __name__ == "__main__":
    cfg = build_default_config()
    results = RayDDPTuner(cfg).run()
    best = results.get_best_result(metric=cfg.metric, mode=cfg.mode)
    print("=" * 60)
    print(f"Best metric {cfg.metric}: {best.metrics.get(cfg.metric)}")
    print(f"Best config: {best.config}")
    print(f"Best path: {best.path}")
    print("=" * 60)
