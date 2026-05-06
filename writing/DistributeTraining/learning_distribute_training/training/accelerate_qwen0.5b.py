import torch
from dataclasses import dataclass, field

from datasets import DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from accelerate_training import BasicConfig, BasicTrainer


@dataclass
class Qwen2Config(BasicConfig):
    task_type: str = "llm"
    project_name: str = "Qwen2.5-FineTune"
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B"
    cache_dir: str = "/root/autodl-fs/huggingface"
    store_dir: str = "/root/autodl-tmp/.cache/HuangJieCode/outputs"

    dataset_name: str = "HuggingFaceH4/MATH-500"
    system_text: str = "You are a mathematician, directly outputting the answers to mathematical problems."
    split: str = "test"
    eval_split_ratio: float = 0.1
    block_size: int = 128
    num_proc: int = 1
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    batch_size: int = 8
    epoch: int = 5
    learning_rate: float = 2e-5
    checkpointing_steps: int = 100
    checkpoints_total_limit: int = 2
    seed: int = 42
    max_train_steps: int = 0
    gradient_plugin: dict = field(default_factory=lambda: {"num_steps": 1})


class Qwen2Trainer(BasicTrainer):
    def __init__(self, config: Qwen2Config):
        model, tokenizer = self._load_model(config)
        train_dataset, eval_dataset = self._load_dataset(config, tokenizer)
        self.collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        super().__init__(
            config=config,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        self.tokenizer = tokenizer

    def _build_dataloader(self) -> None:
        from torch.utils.data import DataLoader

        num_workers = int(getattr(self.config, "dataloader_num_workers", 0))
        common_kwargs = {
            "batch_size": self.config.batch_size,
            "collate_fn": self.collator,
            "num_workers": num_workers,
            "pin_memory": self.config.pin_memory,
            "persistent_workers": self.config.persistent_workers and num_workers > 0,
            "prefetch_factor": self.config.prefetch_factor if num_workers > 0 else None,
        }
        if self.train_dataset is not None:
            self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, **common_kwargs)
        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(self.eval_dataset, shuffle=False, **common_kwargs)

    def _load_model(self, config: Qwen2Config):
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token

        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            use_cache=True,
        )
        model.set_attn_implementation("sdpa")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        peft_config = LoraConfig(
            r= 64,
            lora_alpha= 128,
            lora_dropout= 0.05,
            task_type= "CAUSAL_LM",
            bias="none",
            target_modules= ["q_proj", "k_proj", "v_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.train()
        print(f"Model loaded: {model.config.model_type}, total params: {model.num_parameters()}")
        return model, tokenizer

    def _load_dataset(self, config: Qwen2Config, tokenizer):
        def format_function(example):
            messages = [
                {"role": "system", "content": config.system_text},
                {"role": "user", "content": example['problem']},
                {"role": "assistant", "content": example['answer']}
            ]
            return {"messages": messages}

        def tokenize_function(examples):
            texts = tokenizer.apply_chat_template(
                examples["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            model_inputs = tokenizer(
                texts,
                truncation=True,
                max_length=config.block_size,
                padding="max_length",
                return_overflowing_tokens=False,
            )
            return model_inputs

        dataset = load_dataset(config.dataset_name, split=config.split, cache_dir=config.cache_dir)
        split_dataset = dataset.train_test_split(test_size=config.eval_split_ratio, seed=config.seed)
        raw_datasets = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
        })

        train_dataset = raw_datasets["train"].map(
            format_function, remove_columns=raw_datasets["train"].column_names
        )
        eval_dataset = raw_datasets["validation"].map(
            format_function, remove_columns=raw_datasets["validation"].column_names
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

    def evaluate(self):
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            pbar = tqdm(
                total=len(self.eval_dataloader),
                disable=not self.accelerator.is_main_process,
                desc=f"EVAL",
                dynamic_ncols=True,
            )
            for batch in self.eval_dataloader:
                loss = self.compute_loss(batch)
                total_loss += float(loss.detach().item())
                total_steps += 1
                pbar.update(1)

        self.model.train()
        if total_steps == 0:
            return {"Eval/eval_loss": 0.0}
        return {"Eval/eval_loss": total_loss / total_steps}


if __name__ == "__main__":
    """
    DDP: export HF_ENDPOINT=https://hf-mirror.com && accelerate launch qwen0.5b_training.py 
    """
    config = Qwen2Config()
    trainer = Qwen2Trainer(config)
    trainer.train()
