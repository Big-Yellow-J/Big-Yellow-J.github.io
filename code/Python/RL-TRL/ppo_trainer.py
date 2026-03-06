import os
import re
import gc
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import random
import torch.nn as nn
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Any

from datasets import load_dataset
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# 关键：引入 accelerate 来处理分布式/设备状态
from accelerate import Accelerator

@dataclass
class CustomPPOConfig(PPOConfig):
    random_seed: int = 2026
    project_name: str = "Qwen-PPO-Math"
    cache_dir: str = "/root/autodl-fs/Model"
    current_date: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d"))
    special_num: int = field(default_factory=lambda: random.randint(0, 9999))
    tracker_project_name: str = field(init=False)
    output_dir: str = field(init=False)

    dataset_name: str = "trl-lib/DeepMath-103K"
    dataset_from: str = "hf"
    data_ratio: float = 0.1
    shuffle_dataset: bool = True

    model_name: str = (
        "/root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/"
        "Model/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
    )
    bf16: bool = True

    save_steps: int = 100
    logging_steps: int = 5

    ppo_epochs: int = 1               # 注意：新版常用 ppo_epochs 而非 num_ppo_epochs
    kl_coef: float = 0.05
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 4
    response_length: int = 1024       # 用于 generation 的最大长度
    per_device_train_batch_size: int = 2
    report_to: str = "tensorboard"
    learning_rate: float = 1e-6

    def __post_init__(self):
        self.tracker_project_name = f"{self.current_date}-{self.project_name}-{self.special_num:04d}"
        self.output_dir = f"/root/autodl-fs/Model/Outputs/{self.tracker_project_name}"


THINK_PROMPT_SUFFIX = (
    "请使用以下格式完整回答问题：\n"
    "<think>\n你的逐步推理过程（请详细写出思考步骤）\n</think>\n"
    "<answer>\n最终答案（只写答案本身，不要重复问题）\n</answer>\n"
)


def tokenize_dataset(examples, tokenizer, max_length=1024):
    model_inputs = tokenizer(
        examples["prompt"],
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
    }


def format_dataset_hf(examples, tokenizer):
    processed_prompts = []
    for messages in examples["prompt"]:
        messages = [msg.copy() for msg in messages]
        last_msg = messages[-1]
        if last_msg["role"] == "user":
            last_msg["content"] = last_msg["content"].rstrip() + "\n\n" + THINK_PROMPT_SUFFIX
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        processed_prompts.append(formatted_text)

    return {"prompt": processed_prompts, "solution": examples["solution"]}


def load_datasets(config: CustomPPOConfig, tokenizer=None, split: str = "train"):
    if config.dataset_from != "hf":
        raise NotImplementedError("目前仅支持 HuggingFace 数据集")

    dataset = load_dataset(
        config.dataset_name,
        split=split,
        cache_dir=config.cache_dir
    )
    if config.data_ratio < 1.0:
        dataset = dataset.shuffle(seed=config.random_seed)
        num_samples = int(len(dataset) * config.data_ratio)
        dataset = dataset.select(range(num_samples))

    if tokenizer is not None:
        dataset = dataset.map(
            lambda x: format_dataset_hf(x, tokenizer),
            batched=True,
            desc="Applying chat template"
        )
        print("=" * 100)
        if len(dataset) > 0:
            print(f"示例 Prompt:\n{dataset[0]['prompt']}")
            print(f"示例 Solution:\n{dataset[0]['solution']}")
        print("=" * 100)

        dataset = dataset.map(
            lambda x: tokenize_dataset(x, tokenizer),
            batched=True,
            desc="Tokenizing"
        )

        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print("=" * 100)
    print(f"数据集大小: {len(dataset)}")
    print("=" * 100)
    return dataset


def get_peft_config():
    return LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head"],
    )


def load_model_tokenizer(config: CustomPPOConfig):
    compute_dtype = torch.bfloat16 if config.bf16 else torch.float16

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        torch_dtype=compute_dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
        cache_dir=config.cache_dir,
        return_dict=True,  # 确保返回对象
    )

    # 重点：强制所有子模块同步 return_dict
    model.config.return_dict = True
    model.pretrained_model.config.return_dict = True
    model.config.use_cache = False
    model.pretrained_model.config.use_cache = False

    # 之前的“接线”代码
    base_prefix = model.pretrained_model.base_model_prefix
    model.base_model_prefix = base_prefix
    setattr(model, base_prefix, model.pretrained_model)

    if hasattr(model.pretrained_model, "generation_config"):
        model.generation_config = model.pretrained_model.generation_config
    model.is_gradient_checkpointing = getattr(model.pretrained_model, "is_gradient_checkpointing", False)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

class DummyRuleBasedRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.accuracy_weight = 1.0
        self.reasoning_weight = 0.6
        self.format_weight = 0.8
        self.length_weight = 0.3

    def forward(self, completions: List[str], solutions: List[Any], **kwargs) -> torch.Tensor:
        rewards = []
        for completion, gt in zip(completions, solutions):
            ans_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL | re.IGNORECASE)
            if ans_match:
                pred = ans_match.group(1).strip()
                gt_clean = str(gt).strip() if gt is not None else ""
                acc_score = 1.0 if pred == gt_clean else 0.0
            else:
                acc_score = 0.0

            think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL | re.IGNORECASE)
            has_think = bool(think_match and think_match.group(1).strip())
            has_answer = bool(ans_match)

            format_score = 1.0 if has_think and has_answer else (0.4 if has_think or has_answer else 0.0)

            length = len(completion.strip())
            if 100 <= length <= 800:
                len_score = 1.0
            elif length < 60:
                len_score = 0.1 + length / 600.0
            else:
                len_score = max(0.3, 1.2 - (length - 800) / 1200.0)

            total_reward = (
                self.accuracy_weight * acc_score +
                self.reasoning_weight * (acc_score + format_score) / 2 +
                self.format_weight * format_score +
                self.length_weight * len_score
            )
            rewards.append(total_reward)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.tensor(rewards, device=device, dtype=torch.float32)

class CustomPPOTrainer(PPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        try:
            result = super().compute_loss(model=model, inputs=inputs, return_outputs=return_outputs)
            return result
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"[OOM] Batch skipped: {str(e)[:200]}...")
                torch.cuda.empty_cache()
                gc.collect()
                device = next(model.parameters()).device
                dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
                if return_outputs:
                    return dummy_loss, {"oom_skipped": 1.0, "loss": 0.0}
                return dummy_loss
            raise

if __name__ == "__main__":
    config = CustomPPOConfig()
    accelerator = Accelerator()
    config.distributed_state = accelerator.state
    peft_config = get_peft_config()

    model, tokenizer = load_model_tokenizer(config)
    model.pretrained_model = get_peft_model(model.pretrained_model, peft_config)

    dataset = load_datasets(config, tokenizer)
    reward_model = DummyRuleBasedRewardModel()
    trainer = CustomPPOTrainer(
        args=config,
        model=model,
        ref_model=None,           # 自动创建参考模型
        reward_model=reward_model,
        value_model=model,
        train_dataset=dataset,
        processing_class=tokenizer
    )
    trainer.train()
    final_dir = os.path.join(config.output_dir, "final_lora")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("训练完成！输出目录：", final_dir)