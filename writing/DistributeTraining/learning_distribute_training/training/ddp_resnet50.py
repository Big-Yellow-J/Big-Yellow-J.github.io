from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import load_dataset
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from learning_distribute_training.torchDDP_training import BasicConfig, DDPTrainer
    from learning_distribute_training.utils import *
except ModuleNotFoundError:
    from torchDDP_training import BasicConfig, DDPTrainer
    from utils import *

@dataclass
class ResNet50DDPConfig(BasicConfig):
    store_dir: str = "/root/autodl-tmp/.cache/HuangJieCode/outputs"
    cache_dir: str = "/root/autodl-fs/huggingface"
    task_type: str = "classification"
    project_name: str = "Training-ResNet50-Torch-DDP"
    checkpointing_steps: int = -1
    checkpoints_total_limit: int = 1
    seed: int = 42
    epoch: int = 1
    batch_size: int = 512
    num_workers: int = 8
    max_train_steps: int = 0
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: float = 0.03
    log_with: str = "tensorboard"
    resume_from_checkpoint: str = "/root/autodl-tmp/.cache/HuangJieCode/outputs/20260506-Training-ResNet50-Torch-DDP-7983/checkpoint-interrupted-169"

class ResNet50DDPTrainer(DDPTrainer):
    def __init__(self, config: ResNet50DDPConfig):
        model = self._load_resnet50()
        train_dataset, eval_dataset = self._load_dataset(config)
        super().__init__(
            config=config,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    def _load_resnet50(self) -> nn.Module:
        from torchvision.models import ResNet50_Weights, resnet50

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model

    def _load_dataset(self, config: ResNet50DDPConfig):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        raw_datasets = load_dataset("uoft-cs/cifar10", cache_dir=config.cache_dir)

        def train_transforms(examples):
            examples["pixel_values"] = [
                transform_train(image.convert("RGB")) for image in examples["img"]
            ]
            return examples

        def val_transforms(examples):
            examples["pixel_values"] = [
                transform_test(image.convert("RGB")) for image in examples["img"]
            ]
            return examples

        train_dataset = raw_datasets["train"].with_transform(train_transforms)
        val_dataset = raw_datasets["test"].with_transform(val_transforms)
        return train_dataset, val_dataset

    def _build_dataloader(self) -> None:
        super()._build_dataloader()

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
            return pixel_values, labels

        if self.train_dataloader is not None:
            self.train_dataloader = DataLoader(
                self.train_dataloader.dataset,
                batch_size=self.train_dataloader.batch_size,
                sampler=self.train_dataloader.sampler,
                shuffle=False,
                num_workers=self.train_dataloader.num_workers,
                pin_memory=self.train_dataloader.pin_memory,
                collate_fn=collate_fn,
                drop_last=self.train_dataloader.drop_last,
                persistent_workers=getattr(self.train_dataloader, "persistent_workers", False),
                prefetch_factor=getattr(self.train_dataloader, "prefetch_factor", None),
            )
        if self.eval_dataloader is not None:
            self.eval_dataloader = DataLoader(
                self.eval_dataloader.dataset,
                batch_size=self.eval_dataloader.batch_size,
                sampler=self.eval_dataloader.sampler,
                shuffle=False,
                num_workers=self.eval_dataloader.num_workers,
                pin_memory=self.eval_dataloader.pin_memory,
                collate_fn=collate_fn,
                drop_last=self.eval_dataloader.drop_last,
                persistent_workers=getattr(self.eval_dataloader, "persistent_workers", False),
                prefetch_factor=getattr(self.eval_dataloader, "prefetch_factor", None),
            )

    def evaluate(self) -> Dict[str, float]:
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        total_correct = torch.tensor(0, device=self.device, dtype=torch.long)
        total_samples = torch.tensor(0, device=self.device, dtype=torch.long)

        with torch.no_grad():
            for batch in self.eval_dataloader:
                images, labels = self._move_batch_to_device(batch)
                logits = self.model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                preds = torch.argmax(logits, dim=-1)

                batch_size = torch.tensor(labels.size(0), device=self.device, dtype=torch.long)
                total_loss += loss.detach().float() * batch_size.float()
                total_correct += (preds == labels).sum().detach()
                total_samples += batch_size

        total_loss = self._reduce_scalar(total_loss)
        total_correct = self._reduce_scalar(total_correct)
        total_samples = self._reduce_scalar(total_samples)
        self.model.train()

        num_samples = max(1, int(total_samples.item()))
        return {
            "Eval/loss": float(total_loss.item() / num_samples),
            "Eval/ACC": float(total_correct.item() / num_samples),
        }


if __name__ == "__main__":
    """
    torchrun --nproc_per_node=1 ddp_resnet50.py
    """
    config = ResNet50DDPConfig()
    trainer = ResNet50DDPTrainer(config)
    trainer.train()
