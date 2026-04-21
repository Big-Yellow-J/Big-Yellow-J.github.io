import os
import math
import json
import torch
import random
import shutil
import random
import logging
import numpy as np
import torch.nn as nn
from datetime import datetime
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field

log = get_logger(__name__)


@dataclass
class BasicConfig:
    # 基础参数配置
    cache_dir: str = '/root/autodl-tmp/Model/'
    project_name: str = 'Training-BasicConfig'
    current_date = datetime.now().strftime("%Y%m%d")
    special_num = random.randint(0, 9999)
    tracker_project_name: str = f'{current_date}-{project_name}-{special_num:04d}'
    output_dir: str = f'{cache_dir}/result-QwenVL-Docparse/{tracker_project_name}/'

    # 模型保存参数
    small_dataset: float = 0  # 小批量输出做测试
    checkpointing_steps: int = 0  # 存储一次参数步数
    checkpoints_total_limit: int = 10  # 只存储2组参数
    resume_from_checkpoint: str = None  # 恢复训练路径

    # 模型训练参数
    seed: int = 10086
    epoch: int = 10
    batch_size: int = 1
    max_train_steps: int = 0
    learning_rate: float = 2e-5
    lr_warmup_steps: float = 0.1
    lr_scheduler: str = 'cosine'
    gradient_accumulation_steps: int = 1  # 梯度累计步数

    # 优化器配置
    optim_name: str= "adamW" # adamw/ adamw_8bit


def write_json(json_path, json_file, special_name=None):
    if special_name:
        dir_name = os.path.dirname(json_path)
        base_name = os.path.basename(json_path)
        new_name = f"{special_name}_{base_name}"
        json_path = os.path.join(dir_name, new_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_file, f, ensure_ascii=False, indent=2)

def logger_init(output_dir, tracker_project_name):
    '''日志初始化'''
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{tracker_project_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(file_path),
            logging.StreamHandler()
        ],
    )

def random_state(seed):
    '''控制随机种子'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

def save_checkpoint(accelerator: Accelerator, config: BasicConfig,
                    global_step: int, epoch: int, suffix="",
                    train_loader=None, logger: log = None):
    """保存模型参数，并对数据集状态进行保存方便断点继续训练"""
    if not accelerator.is_main_process:
        return

    save_path = os.path.join(config.output_dir, f"checkpoint{suffix}-{global_step}")
    os.makedirs(save_path, exist_ok=True)

    dataloader_states = None
    if train_loader is not None:
        if train_loader is not None:
            try:
                from accelerate.data_loader import DataLoaderState
                dataloader_state = DataLoaderState(train_loader)
                dataloader_states = [dataloader_state]
                logger.info(f"DataLoader state captured (step {global_step})")
            except Exception as e:
                logger.warning(f"Failed to capture DataLoader state: {e}")

    accelerator.save_state(save_path, dataloader_states=dataloader_states)
    write_json(os.path.join(save_path, "trainer_state.json"),
               {"global_step": global_step, "epoch": epoch})
    logger.info(f"Saved checkpoint (global_step {global_step}) → {save_path}")

    # 保存数据集状态
    if train_loader is not None:
        try:
            dataloader_info = {
                "global_step": global_step,
                "epoch": epoch,
                "dataloader_len": len(train_loader),
                "timestamp": datetime.now().isoformat()
            }

            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'state_dict'):
                sampler_state = train_loader.sampler.state_dict()
                dataloader_info["sampler_state"] = sampler_state

            if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'state_dict'):
                batch_sampler_state = train_loader.batch_sampler.state_dict()
                dataloader_info["batch_sampler_state"] = batch_sampler_state

            write_json(os.path.join(save_path, "dataloader_state.json"), dataloader_info)
        except Exception as e:
            logger.warning(f"Failed to save DataLoader info: {e}")
    logger.info(f"Saved checkpoint (global_step {global_step}) → {save_path}")

    if config.checkpoints_total_limit is not None:
        checkpoints = [
            d for d in os.listdir(config.output_dir)
            if d.startswith("checkpoint-") and "interrupted" not in d
        ]
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        for old_ckpt in checkpoints[:-config.checkpoints_total_limit]:
            old_path = os.path.join(config.output_dir, old_ckpt)
            shutil.rmtree(old_path)
            logger.info(f"Removed old checkpoint: {old_ckpt}")


def load_checkpoint(accelerator: Accelerator, config: BasicConfig,
                    train_loader: DataLoader = None,
                    logger: log = None):
    """加载模型参数，并恢复数据集状态"""
    resume_from_checkpoint = None
    starting_epoch, global_step = 0, 0
    steps_completed_in_current_epoch = 0  # 记录当前epoch已完成步数
    num_update_steps_per_epoch = 0

    if config.resume_from_checkpoint:
        resume_from_checkpoint = config.resume_from_checkpoint
        logger.info(f"Manual resume specified: {resume_from_checkpoint}")
    else:
        # 自动检测最新 checkpoint
        checkpoints = [d for d in os.listdir(config.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else -1,
                                 reverse=True)
            resume_from_checkpoint = os.path.join(config.output_dir, checkpoints[0])
            logger.info(f"Auto-detected latest checkpoint: {resume_from_checkpoint}")

        # 优先使用 interrupted checkpoint
        interrupted_path = os.path.join(config.output_dir, "checkpoint-interrupted")
        if os.path.exists(interrupted_path):
            resume_from_checkpoint = interrupted_path
            logger.info(f"Found interrupted checkpoint, resuming from: {resume_from_checkpoint}")

    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        logger.info(f"Resuming training from {resume_from_checkpoint}")
        num_update_steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)

        try:
            accelerator.load_state(resume_from_checkpoint, strict_load=False)
            logger.info("Loaded optimizer/scheduler/rng states successfully (skipped model weights)")
        except Exception as e:
            logger.warning(f"Failed to load non-model states: {e}. Will reinitialize optimizer/scheduler.")

        trainer_state_path = os.path.join(resume_from_checkpoint, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r") as f:
                state = json.load(f)
                global_step = state.get("global_step", 0)
                epoch_from_state = state.get("epoch", 0)
        else:
            if "checkpoint-" in resume_from_checkpoint:
                try:
                    global_step = int(resume_from_checkpoint.split("checkpoint-")[-1])
                except:
                    global_step = 0
            epoch_from_state = 0

        starting_epoch = global_step // num_update_steps_per_epoch
        steps_completed_in_current_epoch = global_step % num_update_steps_per_epoch

        logger.info(
            f"Calculated: epoch {starting_epoch}, step {global_step}, completed {steps_completed_in_current_epoch} steps in current epoch")

        dataloader_state_path = os.path.join(resume_from_checkpoint, "dataloader_state.json")
        if os.path.exists(dataloader_state_path):
            try:
                with open(dataloader_state_path, "r") as f:
                    dataloader_info = json.load(f)
                logger.info(f"Found DataLoader info from {dataloader_info.get('timestamp', 'unknown')}")

                if train_loader is not None:
                    if "sampler_state" in dataloader_info and hasattr(train_loader, 'sampler') and hasattr(
                            train_loader.sampler, 'load_state_dict'):
                        train_loader.sampler.load_state_dict(dataloader_info["sampler_state"])
                        logger.info("Restored sampler state")

                    if "batch_sampler_state" in dataloader_info and hasattr(train_loader, 'batch_sampler') and hasattr(
                            train_loader.batch_sampler, 'load_state_dict'):
                        train_loader.batch_sampler.load_state_dict(dataloader_info["batch_sampler_state"])
                        logger.info("Restored batch sampler state")
            except Exception as e:
                logger.warning(f"Failed to restore DataLoader info: {e}")

        logger.info(
            f"Resume complete: global_step={global_step}, epoch≈{starting_epoch}, completed_steps={steps_completed_in_current_epoch}")

    is_single_epoch = config.epoch == 1
    if is_single_epoch:
        logger.info("Single epoch training detected")
        starting_epoch = 0

        total_steps_in_epoch = num_update_steps_per_epoch
        steps_completed_in_current_epoch = min(global_step, total_steps_in_epoch)

        if steps_completed_in_current_epoch >= total_steps_in_epoch:
            steps_completed_in_current_epoch = 0
            logger.warning("Previous training completed the epoch, restarting from beginning")

    return starting_epoch, global_step, steps_completed_in_current_epoch

class BasicTrainer:
    def __init__(
            self, config: BasicConfig,
            model: nn.Module=None,
            train_dataset: Dataset=None,
            eval_dataset: Dataset=None,
            optimizers: torch.optim.Optimizer=None,
    ):
        self.config = config
        self.model = model
        self.optimizers = optimizers if optimizers else self._load_optimizer()

    def _load_optimizer(self):
        #TODO: 训练参数还可以优化一下特别是针对 lora 情况
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.config.optim_name == "adamw_8bit":
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        elif self.config.optim_name == "adamw":
            optimizer_class = torch.optim.AdamW
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optim_name}")
        param_groups = [
            {
                'params': trainable_params,
                'lr': self.config.learning_rate,
            }
        ]
        optimizer = optimizer_class(param_groups)
        return optimizer