import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from transformers import ViTForImageClassification, ViTImageProcessor
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import numpy as np
import argparse
import os
import torch.cuda.nvtx as nvtx
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RandomImageDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=(3, 224, 224), num_classes=1000):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, *image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ViTTrainer:
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator(
            mixed_precision='fp16',
            log_with='tensorboard',
            project_dir=args.log_dir
        )
        self.device = self.accelerator.device

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("vit_training_run")

        from modelscope import snapshot_download
        model_dir = snapshot_download('AI-ModelScope/vit-base-patch16-224', cache_dir='/root/autodl-tmp/Model/vit/')

        self.model = ViTForImageClassification.from_pretrained(
            model_dir,
            num_labels=args.num_classes,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        self.processor = ViTImageProcessor.from_pretrained(args.model_name)

        train_ds = RandomImageDataset(num_samples=args.num_samples, num_classes=args.num_classes)
        val_ds = RandomImageDataset(num_samples=200, num_classes=args.num_classes)

        self.train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                       pin_memory=True)
        self.val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

    def train_epoch(self, epoch, profiler=None):
        self.model.train()
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            with self.accelerator.autocast():
                nvtx.range_push("forward")
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                nvtx.range_pop()

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

            nvtx.range_push("optimizer")
            self.optimizer.step()
            self.optimizer.zero_grad()
            nvtx.range_pop()

            total_loss += loss.detach().item()

            if profiler and self.accelerator.is_main_process:
                profiler.step()

        return total_loss / len(self.train_loader)

    def run(self):
        log_root = self.args.log_dir
        prof_log_dir = os.path.join(log_root, "plugins/profile")
        if self.accelerator.is_main_process:
            os.makedirs(prof_log_dir, exist_ok=True)
        if self.args.enable_profiling and self.accelerator.is_main_process:
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1),
                on_trace_ready=tensorboard_trace_handler(log_root),
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
                with_stack=True
            )
            prof.start()
        else:
            prof = None

        try:
            for epoch in range(1, self.args.num_epochs + 1):
                loss = self.train_epoch(epoch, prof)
                if self.accelerator.is_main_process:
                    logger.info(f"Epoch {epoch} | Loss: {loss:.4f}")
                    self.accelerator.log({"train_loss": loss}, step=epoch)
        finally:
            if prof:
                prof.stop()
                if self.accelerator.is_main_process:
                    logger.info("Profiler finished and files exported.")

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--enable_profiling", action="store_true")
    parser.add_argument("--log_dir", type=str, default="./logs")
    args = parser.parse_args()

    torch.manual_seed(42)
    ViTTrainer(args).run()

# nsys profile \
#     --trace=cuda,nvtx,osrt,cudnn,cublas \
#     --capture-range=cudaProfilerApi \
#     --stop-on-range-end=true \
#     --cudabacktrace=true \
#     -o ./nsys_report \
#     python torch_profiler.py