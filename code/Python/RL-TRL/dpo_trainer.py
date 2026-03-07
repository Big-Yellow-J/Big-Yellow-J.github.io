import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import glob
import shutil
import torch
import random
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass, field

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class CustomDPOConfig:
    random_seed: int = 2026
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    project_name: str = "Qwen-DPO-Training"
    cache_dir: str = "/root/autodl-fs/Model"
    current_date: str = datetime.now().strftime("%Y%m%d")
    special_num: int = random.randint(0, 9999)
    tracker_project_name: str = f'{current_date}-{project_name}-{special_num:04d}'
    output_dir: str = field(default_factory=lambda: f"/root/autodl-fs/Model/Outputs/{CustomDPOConfig.tracker_project_name}")

    # 保存/日志设置
    save_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 2
    report_to: str = "tensorboard"

    # 数据集配置
    # dataset_name: str = "trl-lib/ultrafeedback_binarized"
    dataset_name: str = "vicgalle/OpenHermesPreferences-roleplay"
    split: str = "train"
    data_ratio: float = 1

    # 训练配置
    max_steps: int = 5000
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = False

    # DPO 特定
    max_prompt_length: int = 1024
    max_completion_length: int = 1024
    loss_type: str = "sigmoid"
    label_smoothing: float = 0.0

    # PEFT 设置
    use_peft: bool = True
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    task_type: str = "CAUSAL_LM"
    target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

def format_dataset_trl_lib(example):
    return {
        "prompt": [{"role": "user", "content": example["chosen"][0]["content"]}],
        "chosen": [{"role": "assistant", "content": example["chosen"][-1]["content"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"][-1]["content"]}],
    }
def format_dataset_vicgalle(example):
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "chosen": [{"role": "assistant", "content": example["chosen"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]}],
    }

class CustomDPOTrainer:
    def __init__(self, config: CustomDPOConfig, format_function=None):
        self.raw_config = config  # 保存原始配置
        self.format_function = format_function

        # 顺序：先加载基础组件，再加载模型
        self._load_tokenizer()
        self._load_dataset()
        self._load_model()
        self._set_dpoconfig()  # 将 raw_config 转换为 trl 的 DPOConfig

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.raw_config.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            cache_dir=self.raw_config.cache_dir,
            attn_implementation="flash_attention_2",
        )

        if self.raw_config.use_peft:
            peft_config = LoraConfig(
                r=self.raw_config.r,
                lora_alpha=self.raw_config.lora_alpha,
                lora_dropout=self.raw_config.lora_dropout,
                task_type=self.raw_config.task_type,
                bias="none",
                target_modules=self.raw_config.target_modules
            )
            self.ref_model = None
            self.model = get_peft_model(model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = model
            self.ref_model = None

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.raw_config.model_name,
            cache_dir=self.raw_config.cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # DPO 通常建议左填充

    def _load_dataset(self):
        dataset = load_dataset(
            self.raw_config.dataset_name,
            split=self.raw_config.split,
            cache_dir=self.raw_config.cache_dir,
        )
        data_nums = int(len(dataset) * self.raw_config.data_ratio)
        dataset = dataset.shuffle(self.raw_config.random_seed).select(range(data_nums))

        if self.format_function:
            self.dataset = dataset.map(
                self.format_function,
                batched=False,
                remove_columns=dataset.column_names,
                desc="Formatting DPO Dataset"
            )
        else:
            self.dataset = dataset

    def _set_dpoconfig(self):
        # 将自定义配置映射到 trl.DPOConfig
        self.trl_dpo_args = DPOConfig(
            save_steps=self.raw_config.save_steps,
            logging_steps=self.raw_config.logging_steps,
            save_total_limit=self.raw_config.save_total_limit,
            run_name=self.raw_config.tracker_project_name,
            output_dir=self.raw_config.output_dir,

            learning_rate=self.raw_config.learning_rate,
            max_steps=self.raw_config.max_steps,
            per_device_train_batch_size=self.raw_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.raw_config.gradient_accumulation_steps,

            max_prompt_length=self.raw_config.max_prompt_length,
            max_completion_length=self.raw_config.max_completion_length,

            loss_type=self.raw_config.loss_type,
            label_smoothing=self.raw_config.label_smoothing,
            report_to=self.raw_config.report_to,
            gradient_checkpointing=self.raw_config.gradient_checkpointing,
            remove_unused_columns=False,  # DPO 必须为 False
        )

    def train(self):
        dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=self.trl_dpo_args,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
        )
        # dpo_trainer.train()
        accelerator = dpo_trainer.accelerator

        # 日志记录
        tb_log_dir = os.path.join(self.trl_dpo_args.output_dir, "runs")
        writer = None
        if accelerator.is_main_process:
            os.makedirs(tb_log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_log_dir)

        dpo_trainer.create_optimizer_and_scheduler(num_training_steps=self.raw_config.max_steps)
        model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
            self.model,
            dpo_trainer.optimizer,
            dpo_trainer.lr_scheduler,
            dpo_trainer.get_train_dataloader()
        )

        global_step = 0
        save_steps = self.raw_config.save_steps
        max_steps = self.raw_config.max_steps
        progress_bar = tqdm(total=max_steps, desc="DPO Training",disable=not accelerator.is_local_main_process)
        model.train()
        while global_step < max_steps:
            for batch in train_dataloader:
                if global_step >= max_steps:
                    break
                with accelerator.accumulate(model):
                    try:
                        loss, outputs = dpo_trainer.compute_loss(model, batch, return_outputs=True)
                        if torch.isnan(loss) or torch.isinf(loss):
                            if accelerator.is_local_main_process:
                                print(f"[WARNING] NaN/Inf loss at step {global_step}, skipping")
                            optimizer.zero_grad(set_to_none=True)
                            torch.cuda.empty_cache()
                            continue

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    except Exception as e:
                        if accelerator.is_local_main_process:
                            print(f"Error at step {global_step}: {e}")
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        continue

                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)
                    if accelerator.is_main_process:
                        outputs['loss'] = loss.detach().float().item()
                        progress_bar.set_postfix({"loss": f"{outputs['loss']:.4f}"})
                        writer.add_scalar("train/loss", outputs['loss'], global_step)
                        writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], global_step)
                        for key, value in outputs.items():
                            writer.add_scalar(key, value, global_step)
                    if global_step % save_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_local_main_process:
                            unwrapped = accelerator.unwrap_model(model)
                            ckpt_dir = os.path.join(self.raw_config.output_dir, f"checkpoint-{global_step}")
                            unwrapped.save_pretrained(
                                ckpt_dir,
                                is_main_process=accelerator.is_main_process,
                                save_function=accelerator.save,
                                safe_serialization=True
                            )
                            print(f"Checkpoint saved: {ckpt_dir}")
                            if hasattr(self.raw_config, 'save_total_limit') and self.raw_config.save_total_limit > 0:
                                checkpoint_dirs = sorted(
                                    glob.glob(os.path.join(self.raw_config.output_dir, "checkpoint-*")),
                                    key=lambda x: int(x.split('-')[-1])  # 按 step 数字排序
                                )
                                num_to_delete = len(checkpoint_dirs) - self.raw_config.save_total_limit
                                if num_to_delete > 0:
                                    for old_ckpt in checkpoint_dirs[:num_to_delete]:
                                        try:
                                            shutil.rmtree(old_ckpt)
                                            print(f"Deleted old checkpoint: {old_ckpt}")
                                        except Exception as e:
                                            print(f"Failed to delete {old_ckpt}: {e}")
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            unwrapped = accelerator.unwrap_model(model)
            final_dir = os.path.join(self.raw_config.output_dir, "final_model")
            unwrapped.save_pretrained(
                final_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                safe_serialization=True
            )
            if writer is not None:
                writer.flush()
                writer.close()
        accelerator.end_training()
        progress_bar.close()

if __name__ == "__main__":
    dpo_config = CustomDPOConfig()
    # dpo_trainer = CustomDPOTrainer(dpo_config, format_dataset_trl_lib)
    dpo_trainer = CustomDPOTrainer(dpo_config, format_dataset_vicgalle)
    dpo_trainer.train()