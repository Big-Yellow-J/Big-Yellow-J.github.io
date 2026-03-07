import os
import re
import random
import warnings
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore")

import torch
from torch import nn
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
from tqdm import tqdm

from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    GenerationConfig
)
from datasets import load_dataset

set_seed(42)

@dataclass
class ScriptArguments:
    data_ratio: float = field(default=0.001)
    model_name: str = field(default="gpt2")
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: str = field(default="/root/autodl-tmp/Model")
    dataset_name: str = field(default="trl-lib/DeepMath-103K")

    max_prompt_length: int=field(default=1024)
    max_generate_length: int=field(default=1204)
    max_length: int = field(default=2048)
    min_length: int = field(default=10)

    learning_rate: float = field(default=1e-6)

    batch_size: int = field(default=16)
    mini_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    ppo_epochs: int = field(default=4)

    adap_kl_ctrl: bool = field(default=True)
    init_kl_coef: float = field(default=0.2)
    target: float = field(default=6.0)
    horizon: float = field(default=10000)
    cliprange_reward: float = field(default=0.2)
    output_dir: str = field(default="./ppo_output")
    seed: int = field(default=42)
    use_peft: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)

class RewardModel:
    def __init__(self):
        pass

    def _reward_format(self, text_completions: str):
        format_score = 0.0
        think_match = re.search(r'<think>(.*?)</think>', text_completions, re.DOTALL | re.IGNORECASE)
        ans_match = re.search(r'<answer>(.*?)</answer>', text_completions, re.DOTALL | re.IGNORECASE)
        has_think = bool(think_match and think_match.group(1).strip())
        has_answer = bool(ans_match and ans_match.group(1).strip())
        if has_think and has_answer:
            format_score = 1.0
        elif has_think or has_answer:
            format_score = 0.4
        return format_score, has_think, has_answer

    def _reward_length(self, text_completions: str):
        length = len(text_completions.strip())
        if 200 <= length <= 1000:
            return 1.0
        elif length < 200:
            return length / 200
        else:
            return max(0.4, 1.0 - (length - 1000) / 2000)

    def _reward_repetition(self, text_completions: str):
        words = text_completions.split()
        if len(words) < 10:
            return 0.0
        ngrams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
        ratio = len(set(ngrams)) / len(ngrams) if ngrams else 1.0
        return ratio - 1.0

    def compute_reward(self, completions: List[str], solutions: List[str]) -> List[float]:
        rewards = []
        for text, gt in zip(completions, solutions):
            try:
                format_score, accuracy_score = 0.0, 0.0
                think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
                ans_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
                has_think = bool(think_match and think_match.group(1).strip())
                has_answer = bool(ans_match and ans_match.group(1).strip())

                if has_think and has_answer:
                    format_score = 1.0
                elif has_think or has_answer:
                    format_score = 0.4
                if has_answer:
                    pred = ans_match.group(1).strip()
                    gt_clean = str(gt).strip()
                    if pred == gt_clean:
                        accuracy_score = 1.0
                    else:
                        accuracy_score = 0.1 if has_think else 0.0

                repeation_score = self._reward_repetition(text)
                length_score = self._reward_length(text)
                final_reward = format_score * 0.2 + accuracy_score * 0.6 + repeation_score * 0.1 + length_score * 0.1
                rewards.append(float(final_reward))
            except Exception as e:
                print(f"獎勵計算錯誤: {e}")
                rewards.append(0.0)
        return rewards

class PPOTrainerWrapper:
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        output_dir: str,
        use_peft: bool = False,
        **kwargs
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.use_peft = use_peft

        self.script_args = ScriptArguments(
            model_name=model_name,
            dataset_name=dataset_name,
            output_dir=output_dir,
            use_peft=use_peft,
            **kwargs
        )

        self._setup_tokenizer()
        self._setup_model()
        self._setup_datasets()
        self._setup_ppo_config()
        self._setup_reward_model()

    def _setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.script_args.tokenizer_name or self.script_args.model_name,
            cache_dir= self.script_args.cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _setup_model(self):
        base_model = AutoModelForCausalLM.from_pretrained(
            self.script_args.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir=self.script_args.cache_dir,
        )

        if self.use_peft:
            from peft import LoraConfig, get_peft_model
            peft_config = LoraConfig(
                r=self.script_args.lora_r,
                lora_alpha=self.script_args.lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(base_model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = base_model

        self.ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir=self.script_args.cache_dir,
        )

        gen_config = GenerationConfig.from_pretrained(
            self.script_args.model_name,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id,
            cache_dir=self.script_args.cache_dir,
        )
        self.ppo_model.pretrained_model.generation_config = gen_config
        if self.use_peft:
            try:
                self.ppo_model.pretrained_model.base_model.model.generation_config = gen_config
            except AttributeError:
                pass

        self.ref_model = create_reference_model(self.ppo_model)

    def _format_dataset(self, examples):
        THINK_PROMPT_SUFFIX = (
            "\n\n請使用以下格式完整回答問題：\n"
            "<think>\n你的逐步推理過程（請詳細寫出思考步驟）\n</think>\n"
            "<answer>\n最終答案\n</answer>\n"
        )
        if "instruction" in examples:
            raw_texts = examples["instruction"]
        elif "problem" in examples:
            raw_texts = examples["problem"]
        else:
            first_key = list(examples.keys())[0]
            raw_texts = examples[first_key]

        formatted_prompts = []
        for text in raw_texts:
            content = text[0]['content'] if isinstance(text, list) and len(text) > 0 else text
            content_with_think = THINK_PROMPT_SUFFIX + str(content).rstrip()
            messages = [{"role": "user", "content": content_with_think}]
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                max_length=self.script_args.max_prompt_length,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            formatted_prompts.append(formatted_text)

        tokenized = self.tokenizer(
            formatted_prompts,
            truncation=True,
            max_length=self.script_args.max_prompt_length,
            padding=False,
        )
        tokenized["query"] = formatted_prompts
        return tokenized

    def _setup_datasets(self, **kwargs):
        cache_dir = kwargs.get('cache_dir', self.script_args.cache_dir)
        if self.dataset_name == "trl-lib/DeepMath-103K":
            dataset = load_dataset(
                self.dataset_name,
                split="train",
                cache_dir=cache_dir
            )
            data_nums = int(len(dataset) * self.script_args.data_ratio)
            if data_nums > 0 and data_nums < len(dataset):
                dataset = dataset.select(range(data_nums))
            self.train_dataset = dataset.map(
                self._format_dataset,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Injecting think prompt and tokenizing"
            )
            self.query_dataset = self.train_dataset
            print("="*100, f"\nDataset info: {self.query_dataset}\n", "="*100)

    def _setup_ppo_config(self):
        project_name= "Qwen-PPO-Math"
        current_date= datetime.now().strftime("%Y%m%d")
        special_num= random.randint(0, 9999)
        self.tracker_project_name= f'{current_date}-{project_name}-{special_num:04d}'
        self.ppo_config = PPOConfig(
            batch_size=self.script_args.batch_size,
            mini_batch_size=self.script_args.mini_batch_size,
            gradient_accumulation_steps=self.script_args.gradient_accumulation_steps,
            learning_rate=self.script_args.learning_rate,
            ppo_epochs=self.script_args.ppo_epochs,
            adap_kl_ctrl=self.script_args.adap_kl_ctrl,
            init_kl_coef=self.script_args.init_kl_coef,
            target=self.script_args.target,
            horizon=self.script_args.horizon,
            cliprange=self.script_args.cliprange_reward,
            cliprange_value=self.script_args.cliprange_reward,
            gamma=1.0,
            lam=0.95,
            seed=self.script_args.seed,
            log_with="tensorboard",
            tracker_project_name=self.tracker_project_name,
            project_kwargs={
                "logging_dir": f"{self.script_args.output_dir}/logs",
            },
        )

    def _setup_reward_model(self):
        self.reward_model = RewardModel()

    def train(self):
        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.ppo_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 0.9,
            "temperature": 0.7,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        device = ppo_trainer.accelerator.device
        batch_size = self.script_args.batch_size

        progress_bar = tqdm(range(num_iterations), desc="PPO Training")
        for iteration in progress_bar:
            start = iteration * batch_size
            end = start + batch_size
            batch_indices = range(start, min(end, len(self.query_dataset)))
            batch = self.query_dataset.select(batch_indices)

            if len(batch) < batch_size:
                continue
            query_tensors = [
                torch.as_tensor(q["input_ids"], dtype=torch.long, device=device)
                for q in batch
            ]

            response_tensors = []
            for i, query_tensor in enumerate(query_tensors):
                input_length = query_tensor.shape[0]
                with torch.no_grad():
                    output = ppo_trainer.generate(
                        query_tensor,
                        max_new_tokens=self.script_args.max_generate_length,
                        **generation_kwargs
                    )
                resp = output[0, input_length:]
                if resp.numel() == 0:
                    resp = torch.tensor(
                        [self.tokenizer.eos_token_id],
                        dtype=torch.long,
                        device=device
                    )
                response_tensors.append(resp)
            responses = [
                self.tokenizer.decode(gen, skip_special_tokens=True)
                for gen in response_tensors
            ]
            query_texts = [
                self.tokenizer.decode(q, skip_special_tokens=True)
                for q in query_tensors
            ]
            batch = batch.add_column("response", responses)
            batch = batch.add_column("query_text", query_texts)

            full_texts = [q + r for q, r in zip(query_texts, responses)]
            raw_rewards = self.reward_model.compute_reward(full_texts, batch["query_text"])

            rewards = [
                torch.tensor(float(r), dtype=torch.float32, device=device)
                for r in raw_rewards
            ]

            n_queries = len(query_tensors)
            n_responses = len(response_tensors)
            n_rewards = len(rewards)
            if not (n_queries == n_responses == n_rewards == batch_size):
                continue

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            batch_dict = {
                "query": query_texts,
                "response": responses
            }
            ppo_trainer.log_stats(stats, batch_dict, rewards)

            mean_reward = torch.mean(torch.stack(rewards)).item()  # 用 torch 計算更安全
            progress_bar.set_postfix({
                "reward": f"{mean_reward:.4f}",
                "kl": f"{stats['objective/kl']:.4f}",
            })
            if (iteration + 1) % 20 == 0:
                self.save_model(iteration + 1)
        self.save_model("final")
        return ppo_trainer

    def save_model(self, step: str):
        save_path = f"{self.script_args.output_dir}/checkpoint-{step}"
        self.ppo_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"模型已保存到: {save_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="TRL PPO 训练 (trl 0.11.0)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="trl-lib/DeepMath-103K")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/Model/Outputs/PPOTrainer/")
    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--cache_dir", type=str, default="/root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/Model")
    args = parser.parse_args()

    trainer = PPOTrainerWrapper(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        use_peft=args.use_peft,
        cache_dir=args.cache_dir,
    )
    trainer.train(num_iterations=args.num_iterations)

if __name__ == "__main__":
    main()