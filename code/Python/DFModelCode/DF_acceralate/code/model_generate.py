import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_quant_path = "/root/autodl-tmp/Code/Big-Yellow-J.github.io/code/Python/DFModelCode/DF_acceralate/tmp/Qwen2.5-1.5B-GPTQ-W8A8"
model_quant = AutoModelForCausalLM.from_pretrained(model_quant_path,
                                                   device_map="auto",
                                                #    attn_implementation="flash_attention_2",
                                                   )
tokenizer_quant = AutoTokenizer.from_pretrained(model_quant_path)

# model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
# model = AutoModelForCausalLM.from_pretrained(model_name,
#                                              device_map="auto",
#                                              cache_dir="/root/autodl-tmp/Model",
#                                              torch_dtype=torch.bfloat16,
#                                              attn_implementation="flash_attention_2",)
# tokenizer = AutoTokenizer.from_pretrained(model_name,
#                                           cache_dir="/root/autodl-tmp/Model",)

def model_generate(model, tokenizer, prompt):
    gen_config = GenerationConfig(
        max_new_tokens=1000,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.05,
        do_sample=True,
    )
    s_time = time.time()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, generation_config=gen_config)
    response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"回复({time.time()- s_time}): {response.strip()}\n{'-'*60}")

SAMPLE_PROMPTS = ["晚上睡不着怎么办？","写一段关于秋天的诗","解释一下量子纠缠是什么",]
for prompt in SAMPLE_PROMPTS:
    model_generate(model_quant, tokenizer_quant, prompt)
    # model_generate(model, tokenizer, prompt)
