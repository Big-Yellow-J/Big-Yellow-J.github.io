import os
import time
import pprint
from vllm import LLM, SamplingParams

def load_model(model_name, cache_dir=None):
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        download_dir=cache_dir,       # 指定缓存路径
        dtype="bfloat16",             # 或 "float16"
        tensor_parallel_size=1,       # 多GPU时可>1
        gpu_memory_utilization=0.5,
        max_num_seqs=16,              # 降低预热 dummy requests
        max_model_len=20480,          # 限制最大序列长度
        logprobs_mode= 'processed_logprobs'
    )
    return llm

def model_generate(llm, prompt):
    s_time = time.time()
    sampling_params = SamplingParams(
        temperature=0.8,          # 控制随机性
        top_p=0.9,                # nucleus sampling
        top_k=40,                 # 限制采样候选
        repetition_penalty=1.05,  # 防止复读
        max_tokens=20,            # 最大生成长度
        logprobs=5,               # 输出 top-5 token 概率
    )
    outputs = llm.generate([prompt], sampling_params)

    for i, output in enumerate(outputs):
        print(f"Prompt {i}: {output.prompt}")
        for j, gen in enumerate(output.outputs):
            print(f"  Sample {j}: {gen.text}")
            print(f"  Tokens: {gen.token_ids}")
            print(f"  Logprobs: {gen.logprobs}")
            print(f"  Finish reason: {gen.finish_reason}")
    print(f"✅ 推理完成，用时：{time.time()- s_time:.2f} 秒 {type(outputs)}")

if __name__ == '__main__':
    cache_dir = '/hy-tmp'
    model_name = 'Qwen/Qwen3-0.6B'
    prompt = 'Please tell me how to acceralate the llm generate!'
    llm_model = load_model(model_name= model_name,
                           cache_dir= cache_dir)
    model_generate(llm_model, prompt)