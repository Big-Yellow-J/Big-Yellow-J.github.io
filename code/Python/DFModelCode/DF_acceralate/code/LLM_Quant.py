import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier

max_length = 2048
model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
data_apth  = '../test_datasets.jsonl'
store_dir  = '../tmp/Qwen2.5-1.5B-GPTQ-W8A8/'
cache_dir  = '/root/autodl-tmp/Model'
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True,
)

def preprocess(example):
    user_content = example.get("prefix", example.get("instruction", "Who are you?"))
    assistant_content = "I am Qwen Froom Alibaba Cloud."

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,                  
        add_generation_prompt=False,
    )

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
        add_special_tokens=False,
    )

    return {
        # "text": text,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }

dataset = load_dataset("json", data_files=data_apth, split="train")
dataset = dataset.shuffle(seed=42)
dataset = dataset.map(preprocess, desc="Preprocess",remove_columns=dataset.column_names)
print(dataset[0])

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
]

oneshot(
    model=model_name,
    cache_dir= cache_dir,
    dataset=dataset,
    recipe=recipe,
    # text_column="prompt", 这里不需要去指定名称因为我的数据已经被tokenizer处理好了
    max_seq_length=max_length,

    save_compressed=True,
    trust_remote_code_model=True,

    output_dir= store_dir,
    log_dir= f"{store_dir}/logs/"
)