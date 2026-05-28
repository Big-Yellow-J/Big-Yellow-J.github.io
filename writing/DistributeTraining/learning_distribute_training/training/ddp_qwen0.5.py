from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from learning_distribute_training.torchDDP_training import DDPTrainer
    from learning_distribute_training.torchDDP_config import BasicConfig
except ModuleNotFoundError:
    from torchDDP_training import DDPTrainer
    from torchDDP_config import BasicConfig
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*use_reentrant parameter.*"
)

@dataclass
class Qwen2DDPConfig(BasicConfig):
    task_type: str = "llm"
    project_name: str = "Qwen2.5-0.5B-Torch-DDP"
    model_name_or_path: str = "/home/huangjie/MdiriCode/ModelParameterCache/qwen2-0.5B" #"Qwen/Qwen2.5-0.5B"
    store_dir: str = "/home/huangjie/MdiriCode/ModelTrainingResult"
    cache_dir: str = "/home/huangjie/MdiriCode/ModelParameterCache"

    dataset_name: str = "HuggingFaceH4/MATH-500"
    system_text: str = "You are a mathematician, directly outputting the answers to mathematical problems."
    split: str = "test"
    eval_split_ratio: float = 0.1
    block_size: int = 128
    num_proc: int = 1

    epoch: int = 20
    batch_size: int = 32
    learning_rate: float = 2e-5
    checkpointing_steps: int = 100
    checkpoints_total_limit: int = 2
    seed: int = 42
    max_train_steps: int = 0

    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    lr_scheduler: str = "cosine"
    lr_warmup_steps: float = 0.03
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 1
    log_with: str = "tensorboard"
    optim_name: str = "adamw"
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Optional[dict] = None
    distributed_strategy: str = "fsdp2"  # "ddp" | "fsdp2"


class Qwen2DDPTrainer(DDPTrainer):
    def __init__(self, config: Qwen2DDPConfig):
        super().__init__(config=config, model=None, train_dataset=None, eval_dataset=None)
        model, tokenizer = self._load_model(config)
        train_dataset, eval_dataset = self._load_dataset(config, tokenizer)
        self.collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        self.tokenizer = tokenizer
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._build_dataloader()
        self._prepare_model_optimizer_scheduler()

    def _build_dataloader(self) -> None:
        super()._build_dataloader()
        if self.train_dataloader is not None:
            self.train_dataloader = DataLoader(
                self.train_dataloader.dataset,
                batch_size=self.train_dataloader.batch_size,
                sampler=self.train_dataloader.sampler,
                shuffle=False,
                num_workers=self.train_dataloader.num_workers,
                pin_memory=self.train_dataloader.pin_memory,
                collate_fn=self.collator,
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
                collate_fn=self.collator,
                drop_last=self.eval_dataloader.drop_last,
                persistent_workers=getattr(self.eval_dataloader, "persistent_workers", False),
                prefetch_factor=getattr(self.eval_dataloader, "prefetch_factor", None),
            )

    def _load_model(self, config: Qwen2DDPConfig):
        if self.is_distributed:
            if self.is_main_process:
                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_name_or_path,
                    cache_dir=config.cache_dir,
                    trust_remote_code=True,local_files_only=True
                )
            self._barrier()
            if not self.is_main_process:
                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_name_or_path,
                    cache_dir=config.cache_dir,
                    trust_remote_code=True,local_files_only=True
                )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name_or_path,
                cache_dir=config.cache_dir,
                trust_remote_code=True,local_files_only=True
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token

        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32

        # 仅 rank 0 下载模型权重
        if self.is_distributed:
            if self.is_main_process:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_name_or_path,
                    cache_dir=config.cache_dir,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    use_cache=False, local_files_only=True
                )
            self._barrier()
            if not self.is_main_process:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_name_or_path,
                    cache_dir=config.cache_dir,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    use_cache=False,local_files_only=True
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                cache_dir=config.cache_dir,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                use_cache=False,local_files_only=True
            )

        model.set_attn_implementation("sdpa")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False

        peft_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        model = get_peft_model(model, peft_config)
        
        model = model.to(torch_dtype)
        model.train()
        return model, tokenizer

    def _load_dataset(self, config: Qwen2DDPConfig, tokenizer):
        def format_function(example):
            return {
                "messages": [
                    {"role": "system", "content": config.system_text},
                    {"role": "user", "content": example["problem"]},
                    {"role": "assistant", "content": example["answer"]},
                ]
            }

        def tokenize_function(examples):
            texts = tokenizer.apply_chat_template(
                examples["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return tokenizer(
                texts,
                truncation=True,
                max_length=config.block_size,
                padding="max_length",
                return_overflowing_tokens=False,
            )

        dataset = load_dataset(config.dataset_name, split=config.split, cache_dir=config.cache_dir)
        split_dataset = dataset.train_test_split(test_size=config.eval_split_ratio, seed=config.seed)
        raw_datasets = DatasetDict(
            {
                "train": split_dataset["train"],
                "validation": split_dataset["test"],
            }
        )

        train_dataset = raw_datasets["train"].map(
            format_function,
            remove_columns=raw_datasets["train"].column_names,
        )
        eval_dataset = raw_datasets["validation"].map(
            format_function,
            remove_columns=raw_datasets["validation"].column_names,
        )

        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=config.num_proc,
            remove_columns=["messages"],
        )
        tokenized_eval = eval_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=config.num_proc,
            remove_columns=["messages"],
        )
        return tokenized_train, tokenized_eval


if __name__ == "__main__":
    """
    export HF_ENDPOINT=https://hf-mirror.com && CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 ddp_qwen0.5.py
    """
    config = Qwen2DDPConfig()
    # config.distributed_strategy = "fsdp2"
    trainer = Qwen2DDPTrainer(config)
    trainer.train()
