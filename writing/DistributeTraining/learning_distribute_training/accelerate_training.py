import json
import logging
import math
import os
import gc
import random
import shutil
from tqdm import tqdm
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_scheduler

logger = logging.getLogger(__name__)


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
    resume_from_checkpoint: Optional[str] = None

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

    torch_compile: bool = True
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



def write_json(json_path: str, json_file: Dict[str, Any], special_name: Optional[str] = None) -> None:
    if special_name:
        dir_name = os.path.dirname(json_path)
        base_name = os.path.basename(json_path)
        new_name = f"{special_name}_{base_name}"
        json_path = os.path.join(dir_name, new_name)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_file, f, ensure_ascii=False, indent=2)


class BasicTrainer:
    def __init__(
        self,
        config: BasicConfig,
        model: Optional[nn.Module] = None,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        processor: Optional[AutoProcessor] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler=None,
        criterion: Optional[nn.Module] = None,
    ):
        self.config = config
        self.model = model
        if self.config.torch_compile:
            torch.compile(self.model, **self.config.compile_config)
            #TODO: model grad_checkpoint 支持

        self.processor = processor
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion

        self._set_random_seed()
        self._logger_init()
        self._accelerator_init()
        if self.config.accelerator_config["log_with"] == "tensorboard":
            self._tensorboard_init()

        self.optimizer = optimizer if optimizer is not None else self._build_optimizer()

        if lr_scheduler is None:
            self.lr_scheduler = self._build_lr_scheduler()
        else:
            self.lr_scheduler = lr_scheduler

        self._prepare_components()

        self.global_step = 0
        self.starting_epoch = 0
        self.steps_completed_in_current_epoch = 0

    def _logger_init(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        file_path = os.path.join(self.config.output_dir, f"{self.config.tracker_project_name}.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            handlers=[logging.FileHandler(file_path), logging.StreamHandler()],
            force=True,
        )

    def _set_random_seed(self) -> None:
        seed = self.config.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _tensorboard_init(self):
        def filter_hparams(config_dict):
            filtered = {}
            for k, v in config_dict.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    filtered[k] = v
                elif isinstance(v, torch.Tensor):
                    filtered[k] = v
            return filtered

        if self.accelerator.is_main_process:
            tracker_config = dict(vars(self.config))
            tracker_config = filter_hparams(tracker_config)
            self.accelerator.init_trackers(
                self.config.tracker_project_name, config=tracker_config)

    def _accelerator_init(self) -> None:
        # TODO: tensorboard 日志文件记录
        from accelerate.utils import GradientAccumulationPlugin, ProjectConfiguration

        plugins: Dict[str, Any] = {
            "gradient_accumulation_plugin": GradientAccumulationPlugin(**self.config.gradient_plugin)
        }

        if self.config.deepspeed_plugin:
            from accelerate.utils import DeepSpeedPlugin

            plugins["deepspeed_plugin"] = DeepSpeedPlugin(**self.config.deepspeed_plugin)
        elif self.config.fsdp2_plugin:
            from accelerate.utils import FullyShardedDataParallelPlugin

            plugins["fsdp_plugin"] = FullyShardedDataParallelPlugin(**self.config.fsdp2_plugin)

        accelerator_kwargs = dict(self.config.accelerator_config)
        if accelerator_kwargs.get("project_config", None) in ("", None):
            accelerator_kwargs.pop("project_config", None)

        accelerator_project_config = ProjectConfiguration(project_dir=self.config.store_dir)
        self.accelerator = Accelerator(project_config= accelerator_project_config, **plugins, **accelerator_kwargs)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if self.config.optim_name.lower() == "adamw_8bit":
            import bitsandbytes as bnb

            optimizer_class = bnb.optim.AdamW8bit
        elif self.config.optim_name.lower() == "adamw":
            optimizer_class = torch.optim.AdamW
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optim_name}")

        return optimizer_class(trainable_params, lr=self.config.learning_rate)

    def _compute_num_training_steps(self) -> int:
        if self.config.max_train_steps > 0:
            return self.config.max_train_steps

        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        return self.config.epoch * num_update_steps_per_epoch

    def _build_lr_scheduler(self):
        num_training_steps = self._compute_num_training_steps()

        if self.config.lr_warmup_steps <= 1:
            num_warmup_steps = int(self.config.lr_warmup_steps * num_training_steps)
        else:
            num_warmup_steps = int(self.config.lr_warmup_steps)

        return get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _prepare_components(self) -> None:
        prepared = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        )
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        ) = prepared

        if self.criterion is not None:
            self.criterion = self.accelerator.prepare(self.criterion)

    def _resolve_resume_path(self) -> Optional[str]:
        resume_path = None

        if self.config.resume_from_checkpoint:
            manual_path = self.config.resume_from_checkpoint
            if os.path.isabs(manual_path):
                resume_path = manual_path
            else:
                resume_path = os.path.join(self.config.output_dir, manual_path)

        interrupted_path = os.path.join(self.config.output_dir, "checkpoint-interrupted")
        if os.path.exists(interrupted_path):
            return interrupted_path

        if resume_path and os.path.exists(resume_path):
            return resume_path

        if not os.path.exists(self.config.output_dir):
            return None

        checkpoints = [
            d
            for d in os.listdir(self.config.output_dir)
            if d.startswith("checkpoint-") and d.split("checkpoint-")[-1].isdigit()
        ]
        if not checkpoints:
            return None

        checkpoints.sort(key=lambda x: int(x.split("checkpoint-")[-1]), reverse=True)
        return os.path.join(self.config.output_dir, checkpoints[0])

    def load_checkpoint(self) -> Tuple[int, int, int]:
        resume_path = self._resolve_resume_path()
        if not resume_path:
            return 0, 0, 0

        logger.info("Resuming training from %s", resume_path)
        self.accelerator.load_state(resume_path)

        trainer_state_path = os.path.join(resume_path, "trainer_state.json")
        state = {}
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

        global_step = int(state.get("global_step", 0))
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        starting_epoch = global_step // max(1, num_update_steps_per_epoch)
        steps_completed_in_current_epoch = global_step % max(1, num_update_steps_per_epoch)

        logger.info(
            "Resume state loaded: global_step=%s starting_epoch=%s completed_steps=%s",
            global_step,
            starting_epoch,
            steps_completed_in_current_epoch,
        )
        return starting_epoch, global_step, steps_completed_in_current_epoch

    def save_checkpoint(self, global_step: int, epoch: int, suffix: str = "") -> None:
        if not self.accelerator.is_main_process:
            return

        ckpt_name = f"checkpoint{suffix}-{global_step}" if suffix else f"checkpoint-{global_step}"
        save_path = os.path.join(self.config.output_dir, ckpt_name)
        os.makedirs(save_path, exist_ok=True)

        self.accelerator.save_state(save_path)
        write_json(
            os.path.join(save_path, "trainer_state.json"),
            {
                "global_step": global_step,
                "epoch": epoch,
                "saved_at": datetime.now().isoformat(),
            },
        )

        logger.info("Saved checkpoint at step %s -> %s", global_step, save_path)

        if self.config.checkpoints_total_limit is not None:
            checkpoints = [
                d
                for d in os.listdir(self.config.output_dir)
                if d.startswith("checkpoint-")
                and d.split("checkpoint-")[-1].isdigit()
                and "interrupted" not in d
            ]
            checkpoints.sort(key=lambda x: int(x.split("checkpoint-")[-1]))
            expired = checkpoints[:-self.config.checkpoints_total_limit]
            for old_ckpt in expired:
                old_path = os.path.join(self.config.output_dir, old_ckpt)
                shutil.rmtree(old_path, ignore_errors=True)
                logger.info("Removed old checkpoint: %s", old_ckpt)

    def _move_batch_to_device(self, batch):
        if isinstance(batch, dict):
            return {k: v.to(self.accelerator.device) if hasattr(v, "to") else v for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            moved = [v.to(self.accelerator.device) if hasattr(v, "to") else v for v in batch]
            return type(batch)(moved)
        return batch.to(self.accelerator.device) if hasattr(batch, "to") else batch

    def _clean_cuda_gc(self):
        gc.collect()
        torch.cuda.empty_cache()

    def _compute_grad_norm(self) -> float:
        grad_norms = [
            torch.norm(p.grad.detach(), p=2.0)
            for p in self.model.parameters()
            if p.grad is not None
        ]
        if not grad_norms:
            return 0.0
        return float(torch.norm(torch.stack(grad_norms), p=2.0).item())

    def compute_loss(self, batch) -> torch.Tensor:
        batch = self._move_batch_to_device(batch)

        if self.config.task_type == "llm":
            if isinstance(batch, dict):
                outputs = self.model(**batch)
            else:
                raise ValueError("LLM task expects batch to be dict for model(**batch)")

            if hasattr(outputs, "loss") and outputs.loss is not None:
                return outputs.loss
            raise ValueError("LLM model output does not contain .loss")

        if self.config.task_type == "classification":
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Classification task expects batch like (inputs, labels)")

            inputs, labels = batch[0], batch[1]
            logits = self.model(inputs)

            if self.criterion is not None:
                return self.criterion(logits, labels)
            return nn.functional.cross_entropy(logits, labels)

        raise ValueError(f"Unsupported task_type={self.config.task_type}")

    def training_step(self, batch) -> float:
        loss = self.compute_loss(batch)
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(),self.config.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return float(loss.detach().item())

    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            pbar = tqdm(
                total=len(self.train_dataloader),
                disable=not self.accelerator.is_main_process,
                desc=f"EVAL",
                dynamic_ncols=True,
            )
            for batch in self.eval_dataloader:
                loss = self.compute_loss(batch)
                total_loss += float(loss.detach().item())
                total_steps += 1
                pbar.update(1)

        self.model.train()
        if total_steps == 0:
            return {"eval_loss": 0.0}
        return {"eval_loss": total_loss / total_steps}

    def train(self) -> None:
        self.starting_epoch, self.global_step, self.steps_completed_in_current_epoch = self.load_checkpoint()

        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        max_train_steps = (
            self.config.max_train_steps
            if self.config.max_train_steps > 0
            else self.config.epoch * num_update_steps_per_epoch
        )

        self.model.train()
        try:
            for epoch in range(self.starting_epoch, self.config.epoch):
                pbar = tqdm(
                    total=len(self.train_dataloader),
                    disable=not self.accelerator.is_main_process,
                    desc=f"Epoch {epoch}",
                    dynamic_ncols=True,
                )
                if epoch == self.starting_epoch:
                    pbar.update(self.steps_completed_in_current_epoch)

                for step, batch in enumerate(self.train_dataloader):
                    if epoch == self.starting_epoch and step < self.steps_completed_in_current_epoch:
                        continue

                    try:
                        with self.accelerator.accumulate(self.model):
                            loss_value = self.training_step(batch)
                    except Exception as e:
                        logger.error(f"ERROR: {e}")
                        self.optimizer.zero_grad()
                        self._clean_cuda_gc()
                        continue

                    pbar.update(1)

                    if self.accelerator.sync_gradients:
                        self.global_step += 1
                        grad_norm = self._compute_grad_norm()
                        self.accelerator.log({
                                "Train/loss": loss_value,
                                "Train/grad_norm": grad_norm,
                                "Train/lr": self.lr_scheduler.get_last_lr()[0],
                            },step=self.global_step,)

                        pbar.set_postfix({
                                "loss": f"{loss_value:.4f}",
                                "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}",
                            })
                        #TODO: 测试断点加载/重训
                        if self.config.checkpointing_steps > 0 and self.global_step % self.config.checkpointing_steps == 0:
                            self.save_checkpoint(
                                global_step=self.global_step,
                                epoch=epoch,)

                    if self.global_step >= max_train_steps:
                        break

                pbar.close()
                if self.eval_dataloader is not None:
                    metrics = self.evaluate()
                    self.accelerator.log(metrics, step=self.global_step)
                    logger.info("Epoch %s eval metrics: %s", epoch, metrics)

                if self.global_step >= max_train_steps:
                    break
            self.save_checkpoint(
                global_step=self.global_step,
                epoch=self.config.epoch,
                suffix="-last",
            )

        except KeyboardInterrupt:
            logger.warning("Training interrupted, saving interrupted checkpoint...")
            self.save_checkpoint(
                global_step=self.global_step,
                epoch=epoch,
                suffix="-interrupted",
            )
            raise


if __name__ == "__main__":
    logger.info("BasicTrainer module loaded.")
