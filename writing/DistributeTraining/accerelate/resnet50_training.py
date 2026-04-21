import os
import math
import json
import shutil
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from transformers import get_scheduler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, DeepSpeedPlugin
from torch.utils.data import DataLoader
from dataclasses import dataclass


from basic_training import BasicConfig, save_checkpoint, load_checkpoint, random_state, write_json, logger_init

@dataclass
class ResNet50Config(BasicConfig):
    cache_dir: str = ""
    project_name: str = 'Training-ResNet50'
    seed: int = 42

    distribute_name: str = None
    log_with: str= "tensroboard"
    mixed_precision: str= "bf16"
    gradient_accumulation_steps: int= 1
    
def distribute_config(config: ResNet50Config):
    if  config.distribute_name == "deepspeed":
        pass
    elif config.distribute_name == "fsdp":
        pass
    pass

def load_resnet50():
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    return model

def load_dataset(config: ResNet50Config):
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

    train_dataset = torchvision.datasets.CIFAR10(
        root=config.cache_dir, train=True, download=True, transform=transform_train
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=config.cache_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size= config.batch_size,
        shuffle=True, 
        num_workers=16
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=16
    )

    return train_loader, val_loader

def evaluate(model, val_loader, accelerator, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            preds = torch.argmax(outputs, dim=-1)
            correct = (preds == labels).sum()

            total_loss += loss.detach().float()
            total_correct += correct.detach()
            total_samples += labels.size(0)

    total_loss = accelerator.gather(total_loss).sum().item()
    total_correct = accelerator.gather(total_correct).sum().item()
    total_samples = accelerator.gather(torch.tensor(total_samples, device=accelerator.device)).sum().item()

    return total_loss / total_samples, total_correct / total_samples

def main():
    config = ResNet50Config()
    random_state(config.seed)

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir)
    if config.distribute_name:
        pass
    else:
        accelerator = Accelerator(
            log_with= config.log_with,
            project_config=  accelerator_project_config,
            gradient_accumulation_steps= config.gradient_accumulation_steps,
            mixed_precision= config.mixed_precision
        )

    # 日志设置
    if accelerator.is_main_process:
        logger_init(config.output_dir, config.tracker_project_name)
        logger.info(accelerator.state, main_process_only=False)
        os.makedirs(config.output_dir, exist_ok=True)

    # 加载模型数据
    model = load_resnet50()
    loss_fn = nn.CrossEntropyLoss()
    train_loader, val_loader = load_dataset(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_scheduler("cosine", optimizer,
                                 num_warmup_steps= int(config.max_train_steps* 0.1),
                                 num_training_steps= config.max_train_steps)
    model, loss_fn, train_loader, val_loader, lr_scheduler = accelerator.prepare(model, loss_fn, train_loader, val_loader, lr_scheduler)

    global_step = 0

    for epoch in range(config.num_train_epochs):
        model.train()

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                images, labels = batch

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and global_step % config.logging_steps == 0:
                logger.info(f"epoch={epoch} step={global_step} loss={loss.item():.4f}")

            global_step += 1

            if global_step >= config.max_train_steps:
                break

        # ======== Validation ========
        val_loss, val_acc = evaluate(model, val_loader, accelerator, loss_fn)

        if accelerator.is_main_process:
            logger.info(f"[VAL] epoch={epoch} loss={val_loss:.4f} acc={val_acc:.4f}")

            # 保存 checkpoint（只在主进程）
            accelerator.save_state(os.path.join(config.output_dir, f"ckpt_epoch_{epoch}"))

        if global_step >= config.max_train_steps:
            break

