import gc
import json
import logging
import math
import os
import random
import shutil
from collections.abc import Mapping
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

from accelerate_config import BasicConfig
from utils import write_json, get_gpu_info

logger = logging.getLogger(__name__)

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
            self.model = torch.compile(self.model, **self.config.compile_config)

        self.processor = processor
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion

        self._set_random_seed()
        self._logger_init()
        self._accelerator_init()


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
        def flatten_dict(d, parent_key="", sep="."):
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k

                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key))
                else:
                    items[new_key] = v
            return items

        def sanitize(v):
            if isinstance(v, (int, float, str, bool)):
                return v
            elif v is None:
                return "None"
            elif isinstance(v, torch.Tensor):
                return v
            else:
                return str(v)

        if self.accelerator.is_main_process:
            tracker_config = self.config.to_dict()
            tracker_config = flatten_dict(tracker_config)
            tracker_config = {k: sanitize(v) for k, v in tracker_config.items()}

            self.accelerator.init_trackers(
                self.config.tracker_project_name,
                config=tracker_config
            )

    def _profile_init(self):
        self.profile = profile(
            activities= [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=2
            ),
            on_trace_ready=tensorboard_trace_handler(self.config.output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        )

    def _accelerator_init(self) -> None:
        from accelerate.utils import GradientAccumulationPlugin, ProjectConfiguration

        plugins = {
            "gradient_accumulation_plugin": GradientAccumulationPlugin(**self.config.gradient_plugin)
        }

        if self.config.deepspeed_plugin:
            from accelerate.utils import DeepSpeedPlugin
            plugins["deepspeed_plugin"] = DeepSpeedPlugin(**self.config.deepspeed_plugin)
        elif self.config.fsdp2_plugin:
            from accelerate.utils import FullyShardedDataParallelPlugin
            plugins["fsdp_plugin"] = FullyShardedDataParallelPlugin(**self.config.fsdp2_plugin)

        accelerator_kwargs = dict(self.config.accelerator_config)
        accelerator_kwargs.pop("project_config", None)

        accelerator_project_config = ProjectConfiguration(
            project_dir=self.config.store_dir
        )

        self.accelerator = Accelerator(
            project_config=accelerator_project_config,
            **plugins,
            **accelerator_kwargs
        )
        if self.config.torch_profile and self.accelerator.is_main_process:
            self._profile_init()
            self.profile.start()
        if self.config.accelerator_config["log_with"] == "tensorboard":
            self._tensorboard_init()

    def _build_vllm_server(self):
        """ llm eval 阶段可以考虑使用 vllm 进行生成
        TODO ֧支持vllm 进行生成 离线/在线
        """
        pass

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
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler
        )

        if self.criterion is not None:
            self.criterion = self.accelerator.prepare(self.criterion)

    def load_checkpoint(self) -> Tuple[int, int, int]:
        resume_path = self.config.resume_from_checkpoint
        if not os.path.exists(resume_path):
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

    def _clean_cuda_gc(self):
        gc.collect()
        torch.cuda.empty_cache()

    def _move_batch_to_device(self, batch):
        if isinstance(batch, dict):
            return {k: v.to(self.accelerator.device) if hasattr(v, "to") else v for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            moved = [v.to(self.accelerator.device) if hasattr(v, "to") else v for v in batch]
            return type(batch)(moved)
        if isinstance(batch, Mapping):
            return {k: v.to(self.accelerator.device) if hasattr(v, "to") else v for k, v in batch.items()}
        return batch.to(self.accelerator.device) if hasattr(batch, "to") else batch

    def compute_loss(self, batch) -> torch.Tensor:
        batch = self._move_batch_to_device(batch)

        if self.config.task_type == "llm":
            outputs = self.model(**batch)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                return outputs.loss
            if isinstance(outputs, dict) and "loss" in outputs:
                return outputs["loss"]
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

    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            pbar = tqdm(
                total=len(self.eval_dataloader),
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
            return {"Eval/eval_loss": 0.0}
        return {"Eval/eval_loss": total_loss / total_steps}

    def training_step(self, batch) -> Dict[str, float]:
        training_info = {}
        loss = self.compute_loss(batch)
        self.accelerator.backward(loss)
        grad_norm = 0.0
        if self.accelerator.sync_gradients:
            grad_norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            ).item()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        gpu_info = get_gpu_info(self.accelerator)
        training_info = {
            "Train/loss": float(loss.detach().item()),
            "Train/grad_norm": grad_norm,
            "Train/lr": self.lr_scheduler.get_last_lr()[0]}
        training_info.update(gpu_info)
        return training_info

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
        epoch = self.starting_epoch
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
                            train_metrics = self.training_step(batch)
                    except Exception as e:
                        logger.error(f"ERROR: {e}")
                        self.optimizer.zero_grad()
                        self._clean_cuda_gc()
                        continue

                    pbar.update(1)

                    if self.accelerator.sync_gradients:
                        self.global_step += 1
                        if self.global_step % self.config.logging_steps == 0:
                            self.accelerator.log(train_metrics,step=self.global_step,)
                        if self.config.torch_profile:
                            self.profile.step()

                        pbar.set_postfix({
                                "loss": f"{train_metrics['Train/loss']:.4f}",
                                "lr": f"{train_metrics['Train/lr']:.2e}",
                            })
                        if self.config.checkpointing_steps > 0 and self.global_step % self.config.checkpointing_steps == 0:
                            self.save_checkpoint(global_step=self.global_step,epoch=epoch,)

                    if self.global_step >= max_train_steps:
                        break

                pbar.close()
                if self.eval_dataloader is not None:
                    metrics = self.evaluate()
                    self.accelerator.log(metrics, step=self.global_step)
                    logger.info("Epoch %s eval metrics: %s", epoch, metrics)

                if self.global_step >= max_train_steps:
                    break

            self.save_checkpoint(global_step=self.global_step,epoch=self.config.epoch,suffix="-last",)
            if self.config.torch_profile:
                self.profile.stop()

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

