import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray
from ray import tune
from ray.air import FailureConfig
from ray.tune import RunConfig, TuneConfig

from torchDDP_config import RayBaseConfig

class RayDDPTuner:
    def __init__(self, config: RayBaseConfig):
        if not config.train_script:
            raise ValueError("RayBaseConfig.train_script 不能为空。")
        self.config = config
        self._bayesopt_categorical_map: Dict[str, List[Any]] = {}

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
    def _normalize_storage_path(path: str) -> str:
        # Ray + pyarrow 在部分版本下不接受相对路径，统一转绝对路径
        return str(Path(path).expanduser().resolve())

    def _resolve_effective_max_concurrency(self) -> Optional[int]:
        """按 cluster 资源与 trial 资源计算实际并发上限。"""
        cluster = ray.cluster_resources()
        trial_cpu = float(self.config.resources_per_trial.get("CPU", 0) or 0)
        trial_gpu = float(self.config.resources_per_trial.get("GPU", 0) or 0)

        by_cpu: Optional[int] = None
        by_gpu: Optional[int] = None

        if trial_cpu > 0:
            by_cpu = int(cluster.get("CPU", 0) // trial_cpu)
        if trial_gpu > 0:
            by_gpu = int(cluster.get("GPU", 0) // trial_gpu)

        candidates = [v for v in [by_cpu, by_gpu] if v is not None]
        resource_limit = min(candidates) if candidates else None

        if resource_limit is not None:
            resource_limit = max(1, resource_limit)
        user_limit = self.config.max_concurrent_trials

        if user_limit is None:
            return resource_limit
        if resource_limit is None:
            return user_limit
        return max(1, min(user_limit, resource_limit))

    def _build_ray_init_kwargs(self) -> Dict[str, Any]:
        """构造 ray.init 参数，确保 worker 能 import 当前项目模块。"""
        kwargs = dict(self.config.ray_init_kwargs)
        runtime_env = dict(kwargs.get("runtime_env", {}))

        project_dir = str(Path(__file__).resolve().parent)
        runtime_env.setdefault("working_dir", project_dir)

        env_vars = dict(runtime_env.get("env_vars", {}))
        old_pythonpath = env_vars.get("PYTHONPATH", "")
        env_vars["PYTHONPATH"] = (
            f"{project_dir}:{old_pythonpath}" if old_pythonpath else project_dir
        )
        runtime_env["env_vars"] = env_vars

        existed_excludes = list(runtime_env.get("excludes", []))
        merged_excludes = list(dict.fromkeys(existed_excludes + self.config.runtime_env_excludes))
        runtime_env["excludes"] = merged_excludes

        kwargs["runtime_env"] = runtime_env
        return kwargs

    @staticmethod
    def _is_categorical_domain(obj: Any) -> bool:
        cls_name = obj.__class__.__name__.lower()
        return hasattr(obj, "categories") and "categorical" in cls_name

    def _prepare_param_space(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """为不同搜索算法预处理搜索空间。"""
        if self.config.search_alg.lower() != "bayesopt":
            return param_space

        self._bayesopt_categorical_map = {}

        def _convert(node: Any, path: list[str]) -> Any:
            if isinstance(node, dict):
                return {k: _convert(v, path + [k]) for k, v in node.items()}

            if self._is_categorical_domain(node):
                categories = list(getattr(node, "categories"))
                if len(categories) == 0:
                    raise ValueError(f"Empty categorical space at {'/'.join(path)}")
                if len(categories) == 1:
                    return categories[0]
                key = "/".join(path)
                self._bayesopt_categorical_map[key] = categories
                # BayesOpt 仅支持连续边界，先编码为连续变量，trial 前再反解
                return tune.uniform(0.0, float(len(categories) - 1))

            return node

        return _convert(param_space, [])

    @staticmethod
    def _set_nested_value(data: Dict[str, Any], path: List[str], value: Any) -> None:
        cur = data
        for p in path[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[path[-1]] = value

    @staticmethod
    def _get_nested_value(data: Dict[str, Any], path: List[str]) -> Any:
        cur = data
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return None
            cur = cur[p]
        return cur

    def _decode_trial_params_for_bayesopt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self._bayesopt_categorical_map:
            return params
        decoded = dict(params)
        for key, categories in self._bayesopt_categorical_map.items():
            path = key.split("/")
            raw_val = self._get_nested_value(decoded, path)
            if raw_val is None:
                continue
            try:
                idx = int(round(float(raw_val)))
            except Exception:
                idx = 0
            idx = max(0, min(len(categories) - 1, idx))
            self._set_nested_value(decoded, path, categories[idx])
        return decoded

    @staticmethod
    def _find_latest_checkpoint(root_dir: str) -> Optional[str]:
        root = Path(root_dir)
        if not root.exists():
            return None
        candidates: List[Path] = []
        for state_file in root.rglob("training_state.pt"):
            ckpt_dir = state_file.parent
            if ckpt_dir.name.startswith("checkpoint-"):
                candidates.append(ckpt_dir)
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0])

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
        if self.config.search_alg.lower() == "bayesopt":
            trial_params = self._decode_trial_params_for_bayesopt(trial_params)
        merged_params = {**self.config.fixed_params, **trial_params}
        merged_params.setdefault("seed", self.config.seed)
        trial_dir = tune.get_context().get_trial_dir()
        trial_store_dir = os.path.join(trial_dir, self.config.trial_store_subdir)
        os.makedirs(trial_store_dir, exist_ok=True)
        merged_params.setdefault("store_dir", trial_store_dir)

        if self.config.auto_resume_trial and not merged_params.get("resume_from_checkpoint"):
            latest_ckpt = self._find_latest_checkpoint(trial_store_dir)
            if latest_ckpt:
                merged_params["resume_from_checkpoint"] = latest_ckpt

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
        storage_path = self._normalize_storage_path(
            self.config.local_dir or self.config.storage_path
        )
        # 让 Ray 主进程也能读取配置里的环境变量（用于关闭 warning 等）
        for k, v in self.config.env_vars.items():
            os.environ.setdefault(k, str(v))

        ray_init_kwargs = self._build_ray_init_kwargs()
        ray.init(ignore_reinit_error=True, **ray_init_kwargs)
        try:
            effective_concurrency = self._resolve_effective_max_concurrency()
            if self.config.verbose >= 0:
                cluster = ray.cluster_resources()
                print(
                    "[RayDDPTuner] cluster CPU/GPU="
                    f"{cluster.get('CPU', 0)}/{cluster.get('GPU', 0)}, "
                    f"trial CPU/GPU={self.config.resources_per_trial.get('CPU', 0)}/"
                    f"{self.config.resources_per_trial.get('GPU', 0)}, "
                    f"max_concurrent_trials={self.config.max_concurrent_trials}, "
                    f"effective={effective_concurrency}"
                )
            param_space = self._prepare_param_space(self.config.param_space)
            trainable = tune.with_resources(
                self._trainable,
                resources=self.config.resources_per_trial,
            )
            tuner = tune.Tuner(
                trainable,
                param_space=param_space,
                tune_config=TuneConfig(
                    metric=self.config.metric,
                    mode=self.config.mode,
                    num_samples=self.config.num_samples,
                    max_concurrent_trials=effective_concurrency,
                    search_alg=self._build_search_alg(),
                    scheduler=self._build_scheduler(),
                ),
                run_config=RunConfig(
                    name=self.config.experiment_name,
                    storage_path=storage_path,
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
