import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import glob
import shutil
import torch
import random
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

@dataclass
class CustomDPOConfig:
  # 基础设置
  compile: str=True
  random_seed: int = 2026
  model_name: str = "Qwen/Qwen2-0.5B-Instruct"
  project_name: str = "Qwen-DPO-Training"
  cache_dir: str = "/root/autodl-fs/Model"

  # 保存/日志设置
  save_steps: int = 100
  logging_steps: int = 10
  save_total_limit: int = 2
  report_to: str = "tensorboard"

  # 数据集配置
  dataset_name: str = "vicgalle/OpenHermesPreferences-roleplay"
  split: str = "train"
  data_ratio: float = 0.95
  eval_data_ratio: float = 0.05
  num_workers: int=16

  # 训练配置
  max_steps: int = 1000
  eval_steps: float= 0.2
  eval_strategy: str= "steps" # no/steps/epoch
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
  target_modules: List[str] = field(
    default_factory=lambda: [
      "q_proj", "k_proj", "v_proj", "o_proj",
      "gate_proj", "up_proj", "down_proj",
    ]
  )

  def __post_init__(self):
    """后初始化处理动态字段"""
    current_date = datetime.now().strftime("%Y%m%d")
    special_num = random.randint(0, 9999)
    self.tracker_project_name = f'{current_date}-{self.project_name}-{special_num:04d}'
    self.output_dir = f"/root/autodl-fs/Model/Outputs/{self.tracker_project_name}"

def format_dataset_trl_lib(example):
  """格式化 trl-lib 数据集"""
  return {
    "prompt": [{"role": "user", "content": example["chosen"][0]["content"]}],
    "chosen": [{"role": "assistant", "content": example["chosen"][-1]["content"]}],
    "rejected": [{"role": "assistant", "content": example["rejected"][-1]["content"]}],
  }

def format_dataset_vicgalle(example):
  """格式化 vicgalle 数据集"""
  return {
    "prompt": [{"role": "user", "content": example["prompt"]}],
    "chosen": [{"role": "assistant", "content": example["chosen"]}],
    "rejected": [{"role": "assistant", "content": example["rejected"]}],
  }

class CustomDPOTrainer:
  def __init__(self, config: CustomDPOConfig, format_function=None):
    self.config = config
    self.format_function = format_function
    self.writer: Optional[SummaryWriter] = None
    # 顺序加载：tokenizer -> dataset -> model -> dpo config
    self._load_tokenizer()
    self._load_dataset()
    self._load_model()
    self._build_dpo_config()

  def _load_model(self):
    """加载并配置模型"""
    model = AutoModelForCausalLM.from_pretrained(
      self.config.model_name,
      torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
      device_map="auto",
      cache_dir=self.config.cache_dir,
      # attn_implementation="flash_attention_2",
    )

    if self.config.use_peft:
      peft_config = LoraConfig(
        r=self.config.r,
        lora_alpha=self.config.lora_alpha,
        lora_dropout=self.config.lora_dropout,
        task_type=self.config.task_type,
        bias="none",
        target_modules=self.config.target_modules,
      )
      self.model = get_peft_model(model, peft_config)
      self.model.print_trainable_parameters()
    else:
      self.model = model
    self.ref_model = None

  def _load_tokenizer(self):
    """加载分词器"""
    self.tokenizer = AutoTokenizer.from_pretrained(
      self.config.model_name,
      cache_dir=self.config.cache_dir,
    )
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
    self.tokenizer.padding_side = "left"

  def _load_dataset(self):
    """加载并处理数据集"""
    self.dataset = load_dataset(self.config.dataset_name, split=self.config.split, cache_dir=self.config.cache_dir, )
    dataset = self.dataset

    # 按比例抽取数据
    num_train = int(len(dataset) * self.config.data_ratio)
    num_eval = max(100, int(len(dataset) * self.config.eval_data_ratio))

    shuffled = dataset.shuffle(self.config.random_seed)
    train_dataset = shuffled.select(range(num_train))
    eval_start = min(num_train, len(dataset) - num_eval)
    eval_dataset = shuffled.select(range(eval_start, eval_start + num_eval))

    if self.format_function:
      self.train_dataset = dataset.map(
        self.format_function, batched=False, remove_columns=train_dataset.column_names,
        desc="Format Train Dataset"
      )
      self.eval_dataset = dataset.map(
        self.format_function, batched=False, remove_columns=eval_dataset.column_names,
        desc="Format eval Dataset"
      )
    else:
      self.train_dataset = train_dataset
      self.eval_dataset = eval_dataset

  def _build_dpo_config(self):
    """构建 DPOConfig"""
    self.dpo_config = DPOConfig(
      output_dir=self.config.output_dir,
      run_name=self.config.tracker_project_name,
      dataloader_num_workers=self.config.num_workers,

      # 训练参数
      max_steps=self.config.max_steps,
      learning_rate=self.config.learning_rate,
      per_device_train_batch_size=self.config.per_device_train_batch_size,
      gradient_accumulation_steps=self.config.gradient_accumulation_steps,

      # DPO 参数
      max_prompt_length=self.config.max_prompt_length,
      max_completion_length=self.config.max_completion_length,
      loss_type=self.config.loss_type,
      label_smoothing=self.config.label_smoothing,
      eval_steps=int(self.config.max_steps* self.config.eval_steps),
      eval_strategy= self.config.eval_strategy,

      # torch compile
      torch_compile= self.config.compile,

      # 保存/日志
      save_steps=self.config.save_steps,
      logging_steps=self.config.logging_steps,
      save_total_limit=self.config.save_total_limit,
      report_to=self.config.report_to,

      # 其他
      gradient_checkpointing=self.config.gradient_checkpointing,
      remove_unused_columns=False,
    )

  def _setup_logging(self) -> Optional[SummaryWriter]:
    """初始化 TensorBoard 日志"""
    special_name = "RUNS-Complie" if self.config.compile else "RUNS"
    self.tb_log_dir = os.path.join(self.config.output_dir, special_name)
    os.makedirs(self.tb_log_dir, exist_ok=True)
    return SummaryWriter(log_dir=self.tb_log_dir)

  def _cleanup_old_checkpoints(self):
    """清理旧检查点"""
    if self.config.save_total_limit <= 0:
      return

    checkpoint_dirs = sorted(
      glob.glob(os.path.join(self.config.output_dir, "checkpoint-*")),
      key=lambda x: int(x.split("-")[-1]),
    )

    num_to_delete = len(checkpoint_dirs) - self.config.save_total_limit
    for old_ckpt in checkpoint_dirs[:num_to_delete]:
      try:
        shutil.rmtree(old_ckpt)
        print(f"Deleted old checkpoint: {old_ckpt}")
      except Exception as e:
        print(f"Failed to delete {old_ckpt}: {e}")

  def _save_checkpoint(self, model, accelerator, global_step: int):
    """保存检查点"""
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
      unwrapped_model = accelerator.unwrap_model(model)
      ckpt_dir = os.path.join(self.config.output_dir, f"checkpoint-{global_step}")
      unwrapped_model.save_pretrained(
        ckpt_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=True,
      )
      print(f"Checkpoint saved: {ckpt_dir}")
      self._cleanup_old_checkpoints()

  def _save_final_model(self, model, accelerator):
    """保存最终模型"""
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
      unwrapped_model = accelerator.unwrap_model(model)
      final_dir = os.path.join(self.config.output_dir, "final_model")
      unwrapped_model.save_pretrained(
        final_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=True,
      )
      print(f"Final model saved: {final_dir}")

  def train(self):
    """执行 DPO 训练"""
    dpo_trainer = DPOTrainer(
      model=self.model,
      ref_model=self.ref_model,
      args=self.dpo_config,
      train_dataset=self.train_dataset,
      eval_dataset=self.eval_dataset,
      processing_class=self.tokenizer,
    )
    accelerator = dpo_trainer.accelerator

    if accelerator.is_main_process:
      self.writer = self._setup_logging()
      profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
          wait=1,  # 等待步数
          warmup=1,  # 预热步数
          active=3,  # 活跃步数
          repeat=2  # 重复次数
        ),
        on_trace_ready=tensorboard_trace_handler(self.tb_log_dir),
        record_shapes=True,  # 记录张量形状
        profile_memory=True,  # 记录内存使用
        with_stack=True,  # 记录调用栈
        with_flops=True  # 计算 FLOPs
      )
      profiler.start()

    dpo_trainer.create_optimizer_and_scheduler(num_training_steps=self.config.max_steps)
    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
      self.model,
      dpo_trainer.optimizer,
      dpo_trainer.lr_scheduler,
      dpo_trainer.get_train_dataloader(),
    )

    global_step = 0
    progress_bar = tqdm(
      total=self.config.max_steps,
      desc="DPO Training",
      disable=not accelerator.is_local_main_process,
    )
    model.train()

    for batch in train_dataloader:
      if global_step >= self.config.max_steps:
        break

      with accelerator.accumulate(model):
        loss, outputs = dpo_trainer.compute_loss(model, batch, return_outputs=True)
        accelerator.backward(loss)
        if accelerator.sync_gradients:
          accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
      if profiler and accelerator.is_main_process:
        profiler.step()

      if accelerator.sync_gradients:
        global_step += 1
        progress_bar.update(1)

        if accelerator.is_main_process:
          loss_value = loss.detach().item()
          progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})

          if self.writer:
            self.writer.add_scalar("train/loss", loss_value, global_step)
            for key, value in outputs.items():
              if key != "loss" and isinstance(value, (int, float)):
                self.writer.add_scalar(f"train/{key}", value, global_step)
          if global_step % 20 == 0 or global_step == self.config.max_steps:
            if accelerator.is_main_process:
              print(f"\n=== Evaluation at step {global_step} ===")
            eval_metrics = dpo_trainer.evaluate()
            if accelerator.is_main_process and self.writer:
              for k, v in eval_metrics.items():
                if isinstance(v, (int, float)):
                  self.writer.add_scalar(k, v, global_step)
              print("Eval metrics:", {k: f"{v:.4f}" for k, v in eval_metrics.items() if isinstance(v, (int, float))})
        if global_step % self.config.save_steps == 0:
          self._save_checkpoint(model, accelerator, global_step)
    # self._save_final_model(model, accelerator)

    if profiler and accelerator.is_main_process:
      # profiler.export_chrome_trace("trace.json")
      profiler.stop()
    if self.writer:
      self.writer.flush()
      self.writer.close()
    accelerator.end_training()
    progress_bar.close()

if __name__ == "__main__":
  dpo_config = CustomDPOConfig()
  dpo_trainer = CustomDPOTrainer(dpo_config, format_dataset_vicgalle)
  dpo_trainer.train()