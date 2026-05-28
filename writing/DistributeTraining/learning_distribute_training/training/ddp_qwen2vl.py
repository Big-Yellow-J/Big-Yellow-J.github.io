"""
Qwen2-VL 多模态模型 FSDP 分布式训练脚本

支持的模型:
  - Qwen/Qwen2-VL-2B-Instruct  (推荐, 78G 显存友好)
  - Qwen/Qwen2-VL-7B-Instruct  (需多卡 + LoRA)

运行方式:
  export HF_ENDPOINT=https://hf-mirror.com && CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 ddp_qwen2vl.py
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    PreTrainedTokenizerBase,
    Qwen3VLForConditionalGeneration,
    Qwen2VLProcessor,
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

warnings.filterwarnings("ignore", message=".*use_reentrant parameter.*")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class QwenVLConfig(BasicConfig):
    task_type: str = "multimodal"
    project_name: str = "Qwen2-VL-2B-FSDP-LoRA"
    model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    store_dir: str = "/home/huangjie/MdiriCode/ModelTrainingResult"
    cache_dir: str = "/home/huangjie/MdiriCode/ModelParameterCache/Qwen3-VL-8B"

    dataset_name: str = "HuggingFaceM4/Docmatix"
    split: str = "train"
    eval_split_ratio: float = 0.05

    # 图像 / 文本处理参数
    max_pixels: int = 256 * 28 * 28       # 约 200704, 对应 256 个 vision tokens
    min_pixels: int = 64 * 28 * 28
    max_length: int = 512                  # 文本最大 token 数

    num_proc: int = 1

    epoch: int = 5
    batch_size: int = 4                    # 多模态模型显存占用大, batch 需调小
    learning_rate: float = 1e-4            # LoRA 学习率可以稍微大一些
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = 2
    seed: int = 42
    max_train_steps: int = 0

    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    lr_scheduler: str = "cosine"
    lr_warmup_steps: float = 0.03
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 4   # 通过梯度累积等效增大 batch
    log_with: str = "tensorboard"
    optim_name: str = "adamw"
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Optional[dict] = None
    distributed_strategy: str = "fsdp2"    # "ddp" | "fsdp2"

class QwenVLTrainer(DDPTrainer):
    def __init__(self, config: QwenVLConfig):
        super().__init__(config=config, model=None, train_dataset=None, eval_dataset=None)
        model, processor = self._load_model(config)
        train_dataset, eval_dataset = self._load_dataset(config, processor)
        self.processor = processor
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._build_dataloader()
        self._prepare_model_optimizer_scheduler()

    def _load_model(self, config: QwenVLConfig):
        from modelscope import snapshot_download
        if self.is_main_process or not self.is_distributed:
            local_path = snapshot_download(config.model_name_or_path, cache_dir=config.cache_dir)
        if self.is_distributed:
            self._barrier()
            if not self.is_main_process:
                local_path = snapshot_download(config.model_name_or_path, cache_dir=config.cache_dir)

        model_cls = Qwen3VLForConditionalGeneration
        load_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=self._get_torch_dtype(),
            attn_implementation="sdpa",
        )

        processor = AutoProcessor.from_pretrained(
            local_path,
            trust_remote_code=True,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )
        model = model_cls.from_pretrained(local_path, **load_kwargs)

        # Pad token
        tokenizer: PreTrainedTokenizerBase = processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

        # LoRA 注入: 对 vision/language 的线性层微调
        lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        peft_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            bias="none",
            target_modules=lora_targets,
        )
        model = get_peft_model(model, peft_config)

        # 统一 dtype (FSDP 要求)
        model = model.to(self._get_torch_dtype())
        model.train()
        if self.is_main_process:
            print(f"[Qwen2-VL] Model loaded, trainable params: {model.num_parameters(only_trainable=True)}")
        return model, processor

    @staticmethod
    def _get_torch_dtype():
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    # --- Dataset loading ---
    def _load_dataset(self, config: QwenVLConfig, processor: Qwen2VLProcessor):
        """加载多模态数据集并转换为模型输入格式"""
        dataset = load_dataset(config.dataset_name, split=config.split, cache_dir=config.cache_dir)
        split_dataset = dataset.train_test_split(test_size=config.eval_split_ratio, seed=config.seed)
        raw_datasets = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
        })

        def preprocess_fn(examples):
            """
            将多模态样本转为 Qwen2-VL 所需的 messages + images 格式,
            然后通过 processor 生成 input_ids / attention_mask / pixel_values 等。
            """
            batch_messages = []
            batch_images = []

            for question, answer, image in zip(
                examples.get("question", [""] * len(examples.get("image", []))),
                examples.get("answer", [""] * len(examples.get("image", []))),
                examples.get("image", []),
            ):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question or "Describe this image."},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": answer or ""}]},
                ]
                batch_messages.append(messages)
                batch_images.append([image] if image is not None else [])

            # Apply chat template & tokenize
            texts = [
                processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                for msgs in batch_messages
            ]
            model_inputs = processor(
                text=texts,
                images=[imgs[0] if imgs else None for imgs in batch_images],
                truncation=True,
                max_length=config.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            # labels = input_ids (causal LM 自回归)
            model_inputs["labels"] = model_inputs["input_ids"].clone()
            return {k: v.squeeze(0) if v.ndim == 3 else v for k, v in model_inputs.items()}

        # 处理训练集
        train_dataset = raw_datasets["train"].map(
            preprocess_fn,
            batched=True,
            batch_size=8,
            num_proc=config.num_proc,
            remove_columns=raw_datasets["train"].column_names,
        )
        # 处理验证集
        eval_dataset = raw_datasets["validation"].map(
            preprocess_fn,
            batched=True,
            batch_size=8,
            num_proc=config.num_proc,
            remove_columns=raw_datasets["validation"].column_names,
        )
        return train_dataset, eval_dataset

    # --- DataLoader (使用默认 collate, batch 本身就是 dict) ---
    def _build_dataloader(self) -> None:
        super()._build_dataloader()
        # 多模态 batch 已经是 padded dict, 不需要额外 collator
        for attr in ("train_dataloader", "eval_dataloader"):
            loader = getattr(self, attr, None)
            if loader is not None:
                setattr(
                    self,
                    attr,
                    DataLoader(
                        loader.dataset,
                        batch_size=loader.batch_size,
                        sampler=loader.sampler,
                        shuffle=False,
                        num_workers=loader.num_workers,
                        pin_memory=loader.pin_memory,
                        drop_last=loader.drop_last,
                        persistent_workers=getattr(loader, "persistent_workers", False),
                        prefetch_factor=getattr(loader, "prefetch_factor", None),
                    ),
                )

    # --- Loss (复用父类, 多模态模型 output 自带 .loss) ---
    # compute_loss() 从 DDPTrainer 继承, 自动处理 dict batch -> **batch

    # --- Evaluate ---
    def evaluate(self):
        from tqdm import tqdm as _tqdm
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            pbar = _tqdm(
                total=len(self.eval_dataloader),
                disable=not self.is_main_process,
                desc="EVAL",
                dynamic_ncols=True,
            )
            for batch in self.eval_dataloader:
                loss = self.compute_loss(batch)
                total_loss += float(loss.detach().item())
                total_steps += 1
                pbar.update(1)

        self.model.train()
        denom = max(1, total_steps)
        return {"Eval/eval_loss": total_loss / denom}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # export HF_ENDPOINT=https://hf-mirror.com && CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 ddp_qwen2vl.py
    config = QwenVLConfig()
    trainer = QwenVLTrainer(config)
    # trainer.train()
