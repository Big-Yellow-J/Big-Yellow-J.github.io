from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dataclasses import asdict, dataclass, field
from torch.utils.data import DataLoader
from accelerate_training import BasicConfig, BasicTrainer


@dataclass
class ResNet50Config(BasicConfig):
    store_dir: str = "/root/autodl-tmp/.cache/HuangJieCode/outputs"
    cache_dir: str = "/root/autodl-fs/huggingface"
    task_type: str = "classification"
    project_name: str = 'Training-ResNet50-DDP'
    seed: int = 42
    batch_size: int = 32
    num_workers: int= 8
    max_train_steps: int = 0


class ResNet50Trainer(BasicTrainer):
    def __init__(self, config: ResNet50Config):
        model = self._load_resnet50()
        train_dataloader, eval_dataloader = self._load_dataset(config)
        super().__init__(
            config=config,
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
        )

    def _load_resnet50(self):
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        return model

    def _load_dataset(self, config: ResNet50Config):
        from datasets import load_dataset
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        raw_datasets = load_dataset("uoft-cs/cifar10", cache_dir= config.cache_dir)

        def train_transforms(examples):
            examples["pixel_values"] = [transform_train(image.convert("RGB")) for image in examples["img"]]
            return examples

        def val_transforms(examples):
            examples["pixel_values"] = [transform_test(image.convert("RGB")) for image in examples["img"]]
            return examples
        train_dataset = raw_datasets["train"].with_transform(train_transforms)
        val_dataset = raw_datasets["test"].with_transform(val_transforms)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            labels = torch.tensor([example["label"] for example in examples])
            return pixel_values, labels

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn
        )

        return train_loader, val_loader

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        total_correct = torch.tensor(0, device=self.accelerator.device, dtype=torch.long)
        total_samples = torch.tensor(0, device=self.accelerator.device, dtype=torch.long)

        with torch.no_grad():
            for batch in self.eval_dataloader:
                images, labels = self._move_batch_to_device(batch)
                outputs = self.model(images)
                loss = nn.functional.cross_entropy(outputs, labels)

                preds = torch.argmax(outputs, dim=-1)
                correct = (preds == labels).sum()
                batch_size = torch.tensor(labels.size(0), device=self.accelerator.device, dtype=torch.long)

                total_loss += loss.detach().float() * batch_size.float()
                total_correct += correct.detach()
                total_samples += batch_size

        total_loss = self.accelerator.gather(total_loss).sum().item()
        total_correct = self.accelerator.gather(total_correct).sum().item()
        total_samples = self.accelerator.gather(total_samples).sum().item()
        self.model.train()

        if total_samples == 0:
            return {"Eval/loss": 0.0, "Eval/ACC": 0.0}
        return {"Eval/loss": total_loss / total_samples, "Eval/ACC": total_correct / total_samples}

if __name__ == "__main__":
    """
    DDP: export HF_ENDPOINT=https://hf-mirror.com && accelerate launch resnet50_training.py
    """
    config = ResNet50Config()
    trainer = ResNet50Trainer(config)
    trainer.train()
