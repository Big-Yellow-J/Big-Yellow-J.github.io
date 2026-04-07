import os
import re
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from openai import OpenAI
from transformers import AutoTokenizer
from grpo_trainer import load_datasets, format_dataset_hf, CustomGRPOConfig, load_model_tokenizer

# 先去启动vllm


def extract_user_question(text):
    pattern = r"<\|im_start\|>user\s*(.*?)\s*请使用以下格式"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

'''
mkdir -p /root/autodl-tmp/tmp-sore
export TMPDIR=/root/autodl-tmp/tmp-sore # 指定模型下载缓存路径
export TEMP=/root/autodl-tmp/tmp-sore
export TMP=/root/autodl-tmp/tmp-sore

vllm serve \
    /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/Model/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/ \
    --enable-lora \
    --lora-modules ref=/root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/Model/Outputs/20260305-Qwen-GRPO-Math-6612/checkpoint-18500/ref/ \
    --host 0.0.0.0 \
    --port 8001 \
    --gpu-memory-utilization 0.93 \
    --max-model-len 16384 \
    --max-num-seqs 256 \
    --max_lora_rank 128 \
    --trust-remote-code \
    --served-model-name qwen-grpo
    
vllm serve \
    /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/Model/MergeModel/ \
    --host 0.0.0.0 \
    --port 8001 \
    --gpu-memory-utilization 0.93 \
    --max-model-len 16384 \
    --max-num-seqs 256 \
    --max_lora_rank 128 \
    --trust-remote-code \
    --served-model-name qwen-grpo
'''
if __name__ == '__main__':
    SERVED_MODEL_NAME = "qwen-grpo"
    config = CustomGRPOConfig()
    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="vllm",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_datasets(config, tokenizer, split='test')
    for i, sample in enumerate(dataset):
        if i == 0:
            prompt_text = sample.get('question', sample.get('prompt', ""))
            solution_text = sample.get('answer', sample.get('solution', ""))
            if isinstance(prompt_text, list):
                prompt_text = prompt_text[-1].get('content', "")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ]
            # print(extract_user_question(prompt_text))
            try:
                chat_completion = client.chat.completions.create(
                    model=SERVED_MODEL_NAME,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048,
                    extra_body={
                        "repetition_penalty": 1.05,
                    }
                )
                output = chat_completion.choices[0].message.content
                # print(f"\n{'=' * 20} Prompt {'=' * 20}")
                # print(prompt_text)
                # print(f"\n{'=' * 20} Response {'=' * 20}")
                # print(output)
            except Exception as e:
                print(f"Error during inference: {e}")