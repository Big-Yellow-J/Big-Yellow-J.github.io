import os
import re
import gc
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import random
from datetime import datetime
from dataclasses import dataclass, field

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from typing import List, Any
@dataclass
class CustomGRPOConfig(GRPOConfig):
    # trl: 0.22.2
    # https://github.com/huggingface/trl/blob/v0.29.0/trl/trainer/grpo_config.py#L23
    # torch_compile = True

    random_seed: int = 2026
    project_name: str = "Qwen-GRPO-Math"
    cache_dir: str = "/root/autodl-fs/Model"
    current_date: str = datetime.now().strftime("%Y%m%d")
    special_num: int = random.randint(0, 9999)
    tracker_project_name: str = f'{current_date}-{project_name}-{special_num:04d}'
    output_dir: str = field(default_factory=lambda: f"/root/autodl-fs/Model/Outputs-Compile/{CustomGRPOConfig.tracker_project_name}")

    # 保存设置
    save_total_limit: int=2 # 只保存两个

    dataset_name: str = "trl-lib/DeepMath-103K"
    dataset_from: str = "hf"
    data_ratio: float = 0.01
    shuffle_dataset: bool = True

    num_generations: int = 8
    generation_batch_size: int = 8      # generation_batch_size % num_generations == 0 generation_batch_size % per_device_train_batch_size == 0
    max_prompt_length: int = 512
    max_completion_length: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    reward_weights: List[str] = field(default_factory=lambda: [2.0, 1.2, 1.0, 0.4]) # 不同奖励函数权重
    # model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    model_name: str="/root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/Model/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
    bf16: bool = True
    learning_rate: float = 1e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = False
    loss_type: str = "dapo"
    beta: float = 0.04

    logging_steps: int = 5
    save_steps: int = 100
    report_to: str = "tensorboard"
    disable_dropout: bool = True

    use_vllm: bool = True
    vllm_mode: str = "colocate"          # 推荐先试这个（训练和推理共用 GPU）
    # vllm_mode: str = "server"          # 如果 colocate OOM，再改成 server 模式（需单独起 vllm 服务）
    vllm_gpu_memory_utilization: float = 0.65

####################
# 格式化数据为 think-answer格式
####################
THINK_PROMPT_SUFFIX = (
    "请使用以下格式完整回答问题：\n"
    "<think>\n你的逐步推理过程（请详细写出思考步骤）\n</think>\n"
    "<answer>\n最终答案（只写答案本身，不要重复问题）\n</answer>\n"
)

def format_dataset_hf(examples, tokenizer):
    processed_prompts = []
    for messages in examples["prompt"]:
        last_msg = messages[-1]
        if last_msg["role"] == "user":
            last_msg["content"] = last_msg["content"].rstrip() + "\n\n" + THINK_PROMPT_SUFFIX
        
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        processed_prompts.append(formatted_text)
    
    return {
        "prompt": processed_prompts,
        "solution": examples["solution"]
    }

def load_datasets(config: CustomGRPOConfig, tokenizer=None, split: str='train'):
    '''
    GPPO训练数据基本格式：
    {'prompt': [{'content': ' xxx', 'role': 'user'}], 'solution': 'xxx'}
    '''
    if config.dataset_from == "hf":
        dataset = load_dataset(
            config.dataset_name,
            split= split,
            cache_dir=config.cache_dir
        )
        print(dataset[0])
        if tokenizer is not None:
            dataset = dataset.map(
                lambda x: format_dataset_hf(x, tokenizer),
                batched=True,
                remove_columns=dataset.column_names
            )
        if config.data_ratio < 1.0:
            dataset = dataset.shuffle(seed=config.random_seed)
            num_samples = int(len(dataset) * config.data_ratio)
            dataset = dataset.select(range(num_samples))
        # print("="*100)
        # print(f"数据集大小: {len(dataset)}")
        # print(f"示例 Prompt:\n{dataset[0]['prompt']}")
        # print(f"示例 Solution:\n{dataset[0]['solution']}")
        # print("="*100)
        return dataset
    else:
        raise NotImplementedError("目前仅支持 HuggingFace 数据集")

####################
# 模型加载
####################
def load_model_tokenizer(config: CustomGRPOConfig):
    compute_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        device_map="auto",
        torch_dtype=compute_dtype,
        # attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_peft_config():
    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head"],
    )
    return peft_config

####################
# 奖励函数设置
####################
def accuracy_reward(completions: list, solution: list,**kwargs: Any):
    '''提取answer中回答判断是否正确'''
    rewards = []
    for completion, gt in zip(completions, solution):
        # 提取 <answer> ... </answer> 內容
        match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL | re.IGNORECASE)
        if match:
            pred = match.group(1).strip()
            gt_clean = str(gt).strip() if gt is not None else ""
            reward = 1.0 if pred == gt_clean else 0.0
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards

def reasoning_accuracy_reward(completions: list,solution: list,**kwargs: Any):
    '''格式奖励函数'''
    rewards = []
    for completion, gt in zip(completions, solution):
        ans_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL | re.IGNORECASE)
        think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL | re.IGNORECASE)

        has_think = bool(think_match and think_match.group(1).strip())
        has_answer = bool(ans_match)

        if not has_answer:
            rewards.append(0.0)
            continue

        pred = ans_match.group(1).strip()
        gt_clean = str(gt).strip() if gt is not None else ""

        if pred == gt_clean:
            reward = 1.0
        elif has_think:
            reward = 0.4 
        else:
            reward = 0.0

        rewards.append(reward)
    return rewards

def format_reward(completions: List[str], **kwargs) -> List[float]:
    '''回答格式奖励函数'''
    rewards = []
    # print("="*100, f"\n {completions} \n", "="*100)
    for text in completions:
        has_think = bool(re.search(r'<think>.*?</think>', text, re.DOTALL))
        has_answer = bool(re.search(r'<answer>.*?</answer>', text, re.DOTALL))
        if has_think and has_answer:
            rewards.append(1.0)
        elif has_think or has_answer:
            rewards.append(0.4)
        else:
            rewards.append(0.0)
    return rewards

def length_reward(completions: List[str], **kwargs) -> List[float]:
    '''长度奖励函数'''
    rewards = []
    for text in completions:
        length = len(text.strip())
        if 100 <= length <= 800:
            rewards.append(1.0)
        elif length < 60:
            rewards.append(0.1 + length / 600)
        else:
            rewards.append(max(0.3, 1.2 - (length - 800)/1200))
    return rewards

####################
# 自定义Trainer 避免OOM 导致中断
####################
class CustomGRPOTrainer(GRPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        try:
            result = super().compute_loss(
                model=model,
                inputs=inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
            if return_outputs:
                loss, metrics = result
                return loss, metrics
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

    def training_step(self, model, inputs, num_items_in_batch):
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except Exception as e:
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"[training_step OOM/ERROR] {str(e)[:150]} → skip update, loss=0")
                torch.cuda.empty_cache()
                gc.collect()
                return torch.tensor(0.0, device=model.device if hasattr(model, 'device') else 'cuda')
            raise

if __name__ == "__main__":
    config = CustomGRPOConfig()
    model, tokenizer = load_model_tokenizer(config)
    dataset = load_datasets(config, tokenizer)
    print(dataset[0])
    peft_config = get_peft_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = CustomGRPOTrainer(
        model=model,
        reward_funcs=[
            accuracy_reward,
            reasoning_accuracy_reward,
            format_reward,
            length_reward,
        ],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    # trainer.save_model(os.path.join(config.output_dir, "final_lora"))
    # tokenizer.save_pretrained(os.path.join(config.output_dir, "final_lora"))
    # print("训练完成！输出目录：", config.output_dir)