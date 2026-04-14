import os
import math
import timm
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from dataclasses import dataclass
from datasets import load_dataset
from accelerate import Accelerator
from torchvision import transforms
from transformers import get_scheduler
from torch.utils.data import DataLoader
from accelerate.utils import set_seed, ProjectConfiguration

import bitsandbytes as bnb
from torchao.quantization import quantize_, Int8WeightOnlyConfig

@dataclass
class Config:
    model_name: str="vit_base_patch16_224"
    data_name: str="uoft-cs/cifar100"
    mixed_precision: str="fp16"
    log_with: str="tensorboard"
    output_dir: str="./Output"
    lr_scheduler: str="cosine"
    cache_dir: str="./data"
    method: str="torchao" # torchao base bnb

    epochs: int=100
    batch_size: int=64
    save_interval: int=5
    data_ratio: float=0.1
    random_seed: int=10086
    max_grad_norm: float=1.0
    max_train_steps: int=None
    learning_rate: float=1e-4
    lr_warmup_steps: float=0.1
    gradient_accumulation_steps: int=1

class VitQuantTrainer:
    def __init__(self, config: Config, method: str="torchao"):
        self.config = config
        self.config.method = method
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],std=[0.2675, 0.2565, 0.2761])])
        self._accelerate()
        self._load_datasets()
        self._load_model_optimizer()
        self._load_lr_scheduler()
        self.model, self.train_loader, self.val_loader, self.lr_scheduler, self.optimizer = self.accelerator.prepare(
            self.model, self.train_loader, self.val_loader, self.lr_scheduler, self.optimizer
        )

    def _load_datasets(self):
        def transform_fn(examples):
            examples["img"] = [self.transform(img) for img in examples["img"]]
            return examples

        def load_datasets(split_, transform_=True):
            dataset_ = load_dataset(self.config.data_name, cache_dir=self.config.cache_dir, split=split_)
            if self.config.data_ratio < 1:
                nums_datasets = int(len(dataset_)* self.config.data_ratio)
                dataset_ = dataset_.select(range(nums_datasets))
            if transform_:
                dataset_ = dataset_.with_transform(transform_fn)
            return dataset_

        train_ds = load_datasets(split_='train')
        val_ds = load_datasets(split_='test')
        self.train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=4)

    def _load_model_optimizer(self):
        self.model = timm.create_model(self.config.model_name, pretrained=False,
                                       cache_dir=self.config.cache_dir,)
        if self.config.method== 'bnb':
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    pass
            self.optimizer = bnb.optim.Adam8bit(self.model.parameters(),
                                                lr=self.config.learning_rate)
        elif self.config.method == "torchao":
            quantize_(self.model, Int8WeightOnlyConfig())
            self.model = torch.compile(self.model)
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=self.config.learning_rate)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=self.config.learning_rate)

    def _load_lr_scheduler(self):
        if self.config.max_train_steps is None:
            len_train_dataloader_after_sharding = math.ceil(len(self.train_loader) / self.accelerator.num_processes)
            num_update_steps_per_epoch = math.ceil(
                len_train_dataloader_after_sharding / self.config.gradient_accumulation_steps)
            num_training_steps_for_scheduler = (
                    self.config.epochs * num_update_steps_per_epoch * self.accelerator.num_processes
            )
        else:
            num_training_steps_for_scheduler = self.config.max_train_steps * self.accelerator.num_processes

        self.lr_scheduler = get_scheduler(
            name= self.config.lr_scheduler,
            optimizer= self.optimizer,
            num_warmup_steps= self.config.lr_warmup_steps* len(self.train_loader),
            num_training_steps=num_training_steps_for_scheduler,
        )

    def _accelerate(self):
        project_name = f"VIT-CIFAR100-{self.config.method}"
        logging_dir = os.path.join(self.config.output_dir, project_name)
        accelerator_project = ProjectConfiguration(project_dir=self.config.output_dir,
                                                   logging_dir=logging_dir)
        self.accelerator = Accelerator(mixed_precision=self.config.mixed_precision,
                                       log_with=self.config.log_with,
                                       project_config=accelerator_project,
                                       gradient_accumulation_steps=self.config.gradient_accumulation_steps)

    def _one_epoch(self):
        self.model.train()
        total_loss, num_batches = 0.0, 0

        tqdm_bar = tqdm(total= len(self.train_loader), desc="Train",
                        disable=not self.accelerator.is_local_main_process,
                        dynamic_ncols=True)
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch['img']
            labels = batch['fine_label'] if 'fine_label' in batch else batch['label']

            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            self.accelerator.backward(loss)
            if self.config.max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            tqdm_bar.update(1)
        avg_loss = total_loss / num_batches
        return avg_loss

    def trainer(self):
        best_val_acc = 0
        set_seed(self.config.random_seed)
        self.accelerator.init_trackers("vit_training")
        for epoch in range(self.config.epochs):
            train_loss = self._one_epoch()
            if self.accelerator.is_main_process:
                val_acc, val_loss = self.val()
                self.accelerator.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "epoch": epoch
                }, step=epoch)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_checkpoint(epoch, val_acc, is_best=True)
                if epoch % self.config.save_interval == 0:
                    self._save_checkpoint(epoch, val_acc)
        self.accelerator.end_training()
        print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")

    def val(self):
        total_loss, total, correct = 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['img']
                labels = batch['fine_label'] if 'fine_label' in batch else batch['label']

                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.val_loader)
        return accuracy, avg_loss

    def _save_checkpoint(self, epoch, val_acc, is_best=False):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_acc,
            'config': self.config
        }
        if is_best:
            save_path = os.path.join(self.config.output_dir, f"best_model_{self.config.method}.pth")
            self.accelerator.save(checkpoint, save_path)
            print(f"Best model saved to {save_path} (acc: {val_acc:.2f}%)")
        else:
            save_path = os.path.join(self.config.output_dir, f"checkpoint_epoch_{epoch}.pth")
            self.accelerator.save(checkpoint, save_path)
            print(f"Checkpoint saved to {save_path}")

if __name__== "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="torchao")
    args = parser.parse_args()

    train_config = Config()
    train = VitQuantTrainer(config= train_config, method=args.method)
    train.trainer()
