import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from learning_distribute_training.accelerate_training import BasicTrainer
    from learning_distribute_training.accelerate_config import BasicConfig
except ModuleNotFoundError:
    from accelerate_training import BasicTrainer
    from accelerate_config import BasicConfig

@dataclass
class ResNet50Config(BasicConfig):
    store_dir: str = "/root/autodl-tmp/.cache/HuangJieCode/outputs"
    cache_dir: str = "/root/autodl-fs/huggingface"
    task_type: str = "classification"
    project_name: str = 'Training-ResNet50-Accelerate-DDP'
    checkpointing_steps: int = -1
    checkpoints_total_limit: int = 1
    seed: int = 42
    epoch: int = 1
    batch_size: int = 512
    num_workers: int= 8
    max_train_steps: int = 0
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: float = 0.03

@dataclass
class ResNet50ConfigDeepSpeed(BasicConfig):
    store_dir: str = "/root/autodl-tmp/.cache/HuangJieCode/outputs"
    cache_dir: str = "/root/autodl-fs/huggingface"
    task_type: str = "classification"
    project_name: str = 'Training-ResNet50-DeepSpeed'
    checkpointing_steps: int = -1
    checkpoints_total_limit: int = 1
    seed: int = 42
    epoch: int = 1
    batch_size: int = 512
    num_workers: int= 8
    max_train_steps: int = 0

    deepspeed_config: Dict[str, Any] = field(default_factory=lambda: {
        "stage": 2,
        "offload_optimizer_device": "cpu",
        "offload_param_device": "none",
        "gradient_clipping": 1.0,
    })

@dataclass
class ResNet50ConfigFSDP2(BasicConfig):
    store_dir: str = "/root/autodl-tmp/.cache/HuangJieCode/outputs"
    cache_dir: str = "/root/autodl-fs/huggingface"
    task_type: str = "classification"
    project_name: str = 'Training-ResNet50-FSDP'
    checkpointing_steps: int = -1
    checkpoints_total_limit: int = 1
    seed: int = 42
    epoch: int = 1
    batch_size: int = 512
    num_workers: int= 8
    max_train_steps: int = 0
    pass

class ResNet50Trainer(BasicTrainer):
    def __init__(self, config: ResNet50Config):
        model = self._load_resnet50()
        train_dataset, eval_dataset = self._load_dataset(config)
        super().__init__(
            config=config,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    def _build_dataloader(self) -> None:
        from torch.utils.data import DataLoader

        num_workers = int(getattr(self.config, "num_workers", 0))

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            labels = torch.tensor([example["label"] for example in examples])
            return pixel_values, labels

        common_kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
        }

        if self.train_dataset is not None:
            self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, **common_kwargs)
        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(self.eval_dataset, shuffle=False, **common_kwargs)

    def _load_resnet50(self):
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
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

        return train_dataset, val_dataset

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
    DDP: export HF_ENDPOINT=https://hf-mirror.com && accelerate launch accelerate_resnet50.py
    """
    config = ResNet50Config()
    # config = ResNet50ConfigDeepSpeed()
    trainer = ResNet50Trainer(config)
    trainer.train()
