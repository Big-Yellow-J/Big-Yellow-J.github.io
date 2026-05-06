import logging
import math
import os
import random
import shutil
from collections.abc import Mapping
from datetime import datetime
from typing import Dict, Optional, Tuple

import datasets
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

from torchDDP_config import BasicConfig
from utils import write_json, get_gpu_info, clean_cuda_gc

logger = logging.getLogger(__name__)

class DDPTrainer:
    def __init__(
        self,
        config: BasicConfig,
        model: Optional[nn.Module] = None,
        train_dataset: "Dataset | IterableDataset | datasets.Dataset | None" = None,
        eval_dataset: "Dataset | IterableDataset | datasets.Dataset | None" = None,
        processor: Optional[AutoProcessor] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler=None,
        criterion: Optional[nn.Module] = None,
    ):
        self.config = config
        self.model = model
        if self.config.torch_compile:
            self.model = torch.compile(self.model, **self.config.compile_config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_distributed = self.world_size > 1
        self.is_main_process = self.rank == 0
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        self.tb_writer: Optional[SummaryWriter] = None
        self.wandb_run = None

        self._dist_init()
        self._set_seed()
        self._logger_init()
        self._build_dataloader()
        self._prepare_model_optimizer_scheduler()
        self._tracker_init()

        self.global_step = 0
        self.starting_epoch = 0
        self.steps_completed_in_current_epoch = 0

    def _dist_init(self) -> None:
        if not self.is_distributed:
            return
        backend = self.config.backend
        if backend == "nccl" and not torch.cuda.is_available():
            backend = "gloo"
            if self.is_main_process:
                logger.warning("CUDA unavailable; fallback backend from nccl to gloo.")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend=backend, init_method="env://")

    def _set_seed(self) -> None:
        seed = self.config.seed + self.rank
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _logger_init(self) -> None:
        if self.is_main_process:
            file_path = os.path.join(
                self.config.output_dir, f"{self.config.tracker_project_name}.log"
            )
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                handlers=[logging.FileHandler(file_path), logging.StreamHandler()],
                force=True,
            )
        else:
            logging.basicConfig(level=logging.ERROR, force=True)

    def _build_dataloader(self) -> None:
        self.train_dataloader = None
        self.eval_dataloader = None
        common_loader_kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "persistent_workers": self.config.persistent_workers and self.config.num_workers > 0,
            "prefetch_factor": self.config.prefetch_factor if self.config.num_workers > 0 else None,
        }

        if self.train_dataset is not None:
            train_sampler = None
            shuffle = self.config.train_dataset_shuffle
            if self.is_distributed:
                train_sampler = DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=self.config.train_dataset_shuffle,
                    drop_last=False,
                )
                shuffle = False
            self.train_dataloader = DataLoader(
                self.train_dataset,
                sampler=train_sampler,
                shuffle=shuffle,
                **common_loader_kwargs,
            )

        if self.eval_dataset is not None:
            eval_sampler = None
            shuffle = self.config.eval_dataset_shuffle
            if self.is_distributed:
                eval_sampler = DistributedSampler(
                    self.eval_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                    drop_last=False,
                )
                shuffle = False
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                sampler=eval_sampler,
                shuffle=shuffle,
                **common_loader_kwargs,
            )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.config.optim_name.lower() == "adamw":
            return torch.optim.AdamW(trainable_params, lr=self.config.learning_rate)
        if self.config.optim_name.lower() == "adamw_8bit":
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(trainable_params, lr=self.config.learning_rate)
        raise ValueError(f"Unsupported optimizer: {self.config.optim_name}")

    def _compute_training_steps(self) -> int:
        if self.train_dataloader is None:
            return 0
        if self.config.max_train_steps > 0:
            return self.config.max_train_steps
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        return self.config.epoch * num_update_steps_per_epoch

    def _build_lr_scheduler(self):
        num_training_steps = self._compute_training_steps()
        if self.config.lr_warmup_steps <= 1:
            num_warmup_steps = int(self.config.lr_warmup_steps * max(1, num_training_steps))
        else:
            num_warmup_steps = int(self.config.lr_warmup_steps)
        return get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max(1, num_training_steps),
        )

    def _prepare_model_optimizer_scheduler(self) -> None:
        if self.model is None:
            raise ValueError("model is required for DDPTrainer.")
        self.model = self.model.to(self.device)
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                output_device=self.local_rank if self.device.type == "cuda" else None,
            )

        if self.optimizer is None:
            self.optimizer = self._build_optimizer()
        if self.lr_scheduler is None:
            self.lr_scheduler = self._build_lr_scheduler()

    def _tracker_init(self) -> None:
        if not self.is_main_process:
            return
        log_with = self.config.log_with.lower()
        if "tensorboard" in log_with:
            tb_dir = os.path.join(self.config.output_dir, "tb")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
        if "wandb" in log_with:
            try:
                import wandb

                project = self.config.wandb_project or self.config.project_name
                self.wandb_run = wandb.init(
                    project=project,
                    entity=self.config.wandb_entity,
                    name=self.config.tracker_project_name,
                    dir=self.config.output_dir,
                    mode=self.config.wandb_mode,
                    config=self.config.to_dict(),
                    reinit=True,
                )
            except Exception as e:
                logger.exception("WandB init failed, fallback to non-wandb logging: %s", e)
                self.wandb_run = None

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if not self.is_main_process:
            return
        if self.tb_writer is not None:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)
            self.tb_writer.flush()
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def _close_trackers(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def _reduce_scalar(self, value: torch.Tensor) -> torch.Tensor:
        if not self.is_distributed:
            return value
        reduced = value.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced

    def _barrier(self) -> None:
        """同步"""
        if self.is_distributed:
            dist.barrier()

    def _cleanup(self) -> None:
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()

    def _unwrap_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _move_batch_to_device(self, batch):
        if isinstance(batch, dict):
            return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            moved = [v.to(self.device) if hasattr(v, "to") else v for v in batch]
            return type(batch)(moved)
        if isinstance(batch, Mapping):
            return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
        return batch.to(self.device) if hasattr(batch, "to") else batch

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
                raise ValueError("Classification task expects (inputs, labels)")
            inputs, labels = batch[0], batch[1]
            logits = self.model(inputs)
            if self.criterion is not None:
                return self.criterion(logits, labels)
            return nn.functional.cross_entropy(logits, labels)
        raise ValueError(f"Unsupported task_type={self.config.task_type}")

    def training_step(self, batch, should_sync: bool) -> Dict[str, float]:
        loss = self.compute_loss(batch)
        scaled_loss = loss / self.config.gradient_accumulation_steps
        if isinstance(self.model, DDP) and not should_sync:
            with self.model.no_sync():
                scaled_loss.backward()
        else:
            scaled_loss.backward()

        grad_norm = 0.0
        if should_sync:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            ).item()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {
            "Train/loss": float(loss.detach().item()),
            "Train/grad_norm": float(grad_norm),
            "Train/lr": float(self.lr_scheduler.get_last_lr()[0]),
        }

    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataloader is None:
            return {}
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        total_steps = torch.tensor(0, device=self.device, dtype=torch.long)
        with torch.no_grad():
            for batch in self.eval_dataloader:
                loss = self.compute_loss(batch)
                total_loss += loss.detach().float()
                total_steps += 1
        total_loss = self._reduce_scalar(total_loss)
        total_steps = self._reduce_scalar(total_steps)
        self.model.train()
        denom = max(1, int(total_steps.item()))
        return {"Eval/eval_loss": float(total_loss.item() / denom)}

    def save_checkpoint(self, global_step: int, epoch: int, suffix: str = "") -> None:
        if not self.is_main_process:
            return
        ckpt_name = f"checkpoint{suffix}-{global_step}" if suffix else f"checkpoint-{global_step}"
        save_path = os.path.join(self.config.output_dir, ckpt_name)
        os.makedirs(save_path, exist_ok=True)
        state = {
            "model": self._unwrap_model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "saved_at": datetime.now().isoformat(),
            "config": self.config.to_dict(),
        }
        torch.save(state, os.path.join(save_path, "training_state.pt"))
        write_json(
            os.path.join(save_path, "trainer_state.json"),
            {"global_step": global_step, "epoch": epoch, "saved_at": state["saved_at"]},
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
                shutil.rmtree(os.path.join(self.config.output_dir, old_ckpt), ignore_errors=True)
                logger.info("Removed old checkpoint: %s", old_ckpt)

    def load_checkpoint(self) -> Tuple[int, int, int]:
        resume_path = self.config.resume_from_checkpoint
        if not resume_path or not os.path.exists(resume_path):
            return 0, 0, 0
        state_file = os.path.join(resume_path, "training_state.pt")
        if not os.path.exists(state_file):
            return 0, 0, 0
        map_location = self.device if self.device.type == "cuda" else "cpu"
        state = torch.load(state_file, map_location=map_location)
        self._unwrap_model().load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])

        global_step = int(state.get("global_step", 0))
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        starting_epoch = global_step // max(1, num_update_steps_per_epoch)
        steps_completed_in_current_epoch = global_step % max(1, num_update_steps_per_epoch)
        if self.is_main_process:
            logger.info(
                "Resume state loaded: global_step=%s starting_epoch=%s completed_steps=%s",
                global_step,
                starting_epoch,
                steps_completed_in_current_epoch,
            )
        return starting_epoch, global_step, steps_completed_in_current_epoch

    def train(self) -> None:
        if self.train_dataloader is None:
            raise ValueError("train_dataset is required to start training.")
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
                if isinstance(self.train_dataloader.sampler, DistributedSampler):
                    self.train_dataloader.sampler.set_epoch(epoch)
                pbar = tqdm(
                    total=len(self.train_dataloader),
                    disable=not self.is_main_process,
                    desc=f"Epoch {epoch}",
                    dynamic_ncols=True,
                )
                if epoch == self.starting_epoch:
                    pbar.update(self.steps_completed_in_current_epoch)

                for step, batch in enumerate(self.train_dataloader):
                    if epoch == self.starting_epoch and step < self.steps_completed_in_current_epoch:
                        continue
                    is_accum_boundary = (
                        (step + 1) % self.config.gradient_accumulation_steps == 0
                        or (step + 1) == len(self.train_dataloader)
                    )
                    try:
                        train_metrics = self.training_step(batch=batch, should_sync=is_accum_boundary)
                    except Exception as e:
                        logger.exception("Step failed, skip batch. error=%s", e)
                        self.optimizer.zero_grad(set_to_none=True)
                        clean_cuda_gc()
                        continue

                    pbar.update(1)
                    if is_accum_boundary:
                        self.global_step += 1
                        gpu_info = get_gpu_info()
                        train_metrics.update(gpu_info)
                        self._log_metrics(train_metrics, step=self.global_step)
                        if self.is_main_process:
                            pbar.set_postfix(
                                {
                                    "loss": f"{train_metrics['Train/loss']:.4f}",
                                    "lr": f"{train_metrics['Train/lr']:.2e}",
                                }
                            )
                        if (
                            self.config.checkpointing_steps > 0
                            and self.global_step % self.config.checkpointing_steps == 0
                        ):
                            self.save_checkpoint(global_step=self.global_step, epoch=epoch)
                            self._barrier()
                        elif (
                            self.config.checkpointing_steps <= 0
                            and self.global_step % len(self.train_dataloader) == 0
                        ):
                            self.save_checkpoint(global_step=self.global_step, epoch=epoch)
                            self._barrier()
                    if self.global_step >= max_train_steps:
                        break

                pbar.close()
                if self.eval_dataloader is not None:
                    metrics = self.evaluate()
                    self._log_metrics(metrics, step=self.global_step)
                    if self.is_main_process:
                        logger.info("Epoch %s eval metrics: %s", epoch, metrics)
                if self.global_step >= max_train_steps:
                    break

            self.save_checkpoint(global_step=self.global_step, epoch=self.config.epoch, suffix="-last")
            self._barrier()
        except KeyboardInterrupt:
            if self.is_main_process:
                logger.warning("Training interrupted, saving interrupted checkpoint...")
            self.save_checkpoint(global_step=self.global_step, epoch=epoch, suffix="-interrupted")
            self._barrier()
            raise
        finally:
            self._close_trackers()
            self._cleanup()

