import re
import time
import torch
from tqdm import tqdm
from typing import Dict, List
from datasets import load_dataset
from vllm import LLM, SamplingParams

MODEL_PATH = "/root/autodl-tmp/Code/Big-Yellow-J.github.io/code/Python/DFModelCode/DF_acceralate/tmp/Qwen2.5-1.5B-GPTQ-W8A8" 
DATASET_NAME = "ceval/ceval-exam"
SPLIT = "validation" 
MAX_SAMPLES = 100 
MAX_TOKENS = 1024 
CACHE_DIR = "/root/autodl-tmp/Model"

def load_model(model_path: str):
    print("加载模型...")
    start = time.time()
    model = LLM(
        model=model_path,
        max_model_len=MAX_TOKENS * 2,
        dtype="auto",
        gpu_memory_utilization=0.9,
        enforce_eager=False,
    )
    load_time = time.time() - start
    print(f"模型加载耗时: {load_time:.2f} 秒")
    return model

def load_and_prepare_data(dataset_name: str, split: str, max_samples: int = -1) -> List[Dict]:
    print(f"加载数据集 {dataset_name} ({split} split)...")
    dataset = load_dataset(dataset_name, split=split, cache_dir= CACHE_DIR)
    print(f"数据集大小: {len(dataset)}")
    print("样本示例:", dataset[0])
    
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    processed = []
    for sample in dataset:
        question = sample['question']
        options = {
            'A': sample['A'],
            'B': sample['B'],
            'C': sample['C'],
            'D': sample['D'],
        }
        answer = sample['answer'] 
        
        prompt = f"""以下是一个选择题，请仔细阅读并选择正确的选项。
        问题：{question}
        A: {options['A']}
        B: {options['B']}
        C: {options['C']}
        D: {options['D']}
        请直接输出正确选项的字母（如 A、B、C 或 D），不要添加额外解释。"""
        
        processed.append({
            "prompt": prompt,
            "correct_answer": answer,
            "options": options,
        })
    print(f"预处理样本数: {len(processed)}")
    return processed

def generate_responses(model, data: List[Dict]) -> List[Dict]:
    results = []
    sampling_params = SamplingParams(
        temperature=0.0,         
        top_p=1.0,
        top_k=-1,
        max_tokens=10,           
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )
    
    for item in tqdm(data, desc="生成响应"):
        start = time.time()
        outputs = model.generate(item["prompt"], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        elapsed = time.time() - start
        
        predicted = re.search(r'[A-D]', response.upper())
        predicted = predicted.group(0) if predicted else "未知"
        
        results.append({
            "prompt": item["prompt"],
            "predicted": predicted,
            "correct": item["correct_answer"],
            "response_full": response,
            "time_seconds": elapsed,
        })
    
    return results

def evaluate_results(results: List[Dict]):
    total = len(results)
    correct = sum(1 for r in results if r["predicted"] == r["correct"])
    accuracy = correct / total * 100 if total > 0 else 0
    
    print("\n评估结果:")
    print(f"总样本数: {total}")
    print(f"正确数: {correct}")
    print(f"准确率: {accuracy:.2f}%")
    
    # 平均生成时间
    avg_time = sum(r["time_seconds"] for r in results) / total
    print(f"平均生成耗时: {avg_time:.2f} 秒")
    
    # 打印前 5 个错误示例（调试用）
    print("\n错误示例（前5个）：")
    errors = [r for r in results if r["predicted"] != r["correct"]]
    for i, err in enumerate(errors[:5], 1):
        print(f"错误 {i}:")
        print(f"Prompt: {err['prompt'][:100]}...")
        print(f"预测: {err['predicted']} | 正确: {err['correct']}")
        print(f"完整响应: {err['response_full']}")
        print("-" * 50)

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    data = load_and_prepare_data(DATASET_NAME, SPLIT, MAX_SAMPLES)
    results = generate_responses(model, data)
    evaluate_results(results)
    del model
    torch.cuda.empty_cache()