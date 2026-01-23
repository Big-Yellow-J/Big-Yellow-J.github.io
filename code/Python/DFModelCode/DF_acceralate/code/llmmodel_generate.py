import time
import torch
import multiprocessing
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
multiprocessing.set_start_method('spawn', force=True)

# model_quant_path = "/root/autodl-tmp/Code/Big-Yellow-J.github.io/code/Python/DFModelCode/DF_acceralate/tmp/Qwen2.5-1.5B-GPTQ-W8A8"
# model_quant_path = "/root/autodl-tmp/Code/Big-Yellow-J.github.io/code/Python/DFModelCode/DF_acceralate/tmp/Qwen2.5-1.5B-GPTQ-W4A16"
# model_quant = AutoModelForCausalLM.from_pretrained(model_quant_path,
#                                                    device_map="auto",
#                                                    attn_implementation="flash_attention_2",
#                                                    )
# tokenizer_quant = AutoTokenizer.from_pretrained(model_quant_path)

# model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
# model = AutoModelForCausalLM.from_pretrained(model_name,
#                                              device_map="auto",
#                                              cache_dir="/root/autodl-tmp/Model",
#                                              torch_dtype=torch.bfloat16,
#                                              attn_implementation="flash_attention_2",)
# tokenizer = AutoTokenizer.from_pretrained(model_name,
#                                           cache_dir="/root/autodl-tmp/Model",)

s_init_model_time = time.time()
model_quant = LLM("/root/autodl-tmp/Code/Big-Yellow-J.github.io/code/Python/DFModelCode/DF_acceralate/tmp/Qwen2.5-1.5B-GPTQ-W8A8", max_model_len=2048)
# model = LLM("Qwen/Qwen2.5-1.5B-Instruct", 
#             download_dir="/root/autodl-tmp/Model",
#             max_model_len=2048,)
print(f"初始化模型耗时: {time.time()-s_init_model_time:.2f}秒")

def generate(model, prompt):
    s_time = time.time()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    
    sampling_params = SamplingParams(
        temperature=0.8,          # 温度：值越高输出越随机
        top_p=0.9,                # 核采样：控制词汇多样性
        top_k=50,                 # Top-k采样
        max_tokens=1024,          # 最大生成长度
        frequency_penalty=0.1,    # 频率惩罚：减少重复内容
        presence_penalty=0.1,     # 存在惩罚：鼓励新话题
        repetition_penalty=1.1,   # 重复惩罚（某些模型支持）
    )
    output = model.chat(messages, sampling_params)
    elapsed_time = time.time() - s_time
    print(f"生成耗时: {elapsed_time:.2f}秒")
    print(f"输出结果: {output}")

# def model_generate(model, tokenizer, prompt):
#     gen_config = GenerationConfig(
#         max_new_tokens=1000,
#         temperature=0.7,
#         top_p=0.95,
#         repetition_penalty=1.05,
#         do_sample=True,
#     )
#     s_time = time.time()
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user",   "content": prompt},
#     ]
#     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer(text, return_tensors="pt").to(model.device)
#     output_ids = model.generate(**inputs, generation_config=gen_config)
#     response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
#     print(f"\nPrompt: {prompt}")
#     print(f"回复({time.time()- s_time}): {response.strip()}\n{'-'*60}")

SAMPLE_PROMPTS = ["晚上睡不着怎么办？","写一段关于秋天的诗","解释一下量子纠缠是什么",]
for prompt in SAMPLE_PROMPTS:
    # model_generate(model_quant, tokenizer_quant, prompt) #4.8G
    # model_generate(model, tokenizer, prompt) # 5.6G
    # generate(model, prompt) # 88.85s 初始化 生成 10.34 0.16 1.25s
    generate(model_quant, prompt) # 78.27s 初始化 生成 1.72 0.56 0.71s

