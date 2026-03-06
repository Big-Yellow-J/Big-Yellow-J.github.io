import re
import os
import torch
import subprocess
import logging
from datetime import datetime
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置日志输出到文件和控制台
def setup_logging(output_dir):
    """设置日志记录，同时输出到文件和控制台"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # 配置logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return log_file

# 自定义文件输出类
class TrainingLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_dir = os.path.join(output_dir, 'training_outputs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.txt_file = os.path.join(self.log_dir, f'training_output_{timestamp}.txt')
        self.reward_file = os.path.join(self.log_dir, f'reward_details_{timestamp}.txt')
        
    def log(self, message):
        """记录一般信息"""
        with open(self.txt_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        # print(message)
    
    def log_reward_details(self, question, answer, response, extracted_response, reward):
        """记录详细的奖励信息"""
        with open(self.reward_file, 'a', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"问题: {question}\n")
            f.write(f"正确答案: {answer}\n")
            f.write(f"模型响应: {response}\n")
            f.write(f"提取的答案: {extracted_response}\n")
            f.write(f"奖励: {reward}\n")
            f.write("=" * 80 + "\n\n")

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(cahce_dir, split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main', cache_dir=cahce_dir)[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

# 初始化日志记录器
output_dir = "/root/autodl-tmp/Code/Big-Yellow-J.github.io/code/Python/DFModelCode/DF_acceralate/tmp/GRPOTraining"
logger = TrainingLogger(output_dir)
log_file = setup_logging(output_dir)

logger.log(f"训练开始于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.log(f"日志文件保存在: {log_file}")
logger.log(f"训练输出文件: {logger.txt_file}")
logger.log(f"奖励详情文件: {logger.reward_file}")

# Reward functions - 修改为使用logger记录
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # 记录到文件
    logger.log_reward_details(
        question=q,
        answer=answer[0],
        response=responses[0],
        extracted_response=extracted_responses[0],
        reward="正确" if extracted_responses[0] == answer[0] else "错误"
    )
    
    # 同时输出到控制台
    print('-'*50, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = [0.5 if r.isdigit() else 0.0 for r in extracted_responses]
    
    # 记录奖励信息
    if rewards[0] > 0:
        logger.log(f"整数格式奖励: {rewards[0]}, 提取的答案: {extracted_responses[0]}")
    
    return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    rewards = [0.5 if match else 0.0 for match in matches]
    
    # 记录格式奖励信息
    if rewards[0] > 0:
        logger.log(f"严格格式奖励: {rewards[0]}")
    
    return rewards

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    rewards = [0.5 if match else 0.0 for match in matches]
    
    # 记录格式奖励信息
    if rewards[0] > 0:
        logger.log(f"宽松格式奖励: {rewards[0]}")
    
    return rewards

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    rewards = [count_xml(c) for c in contents]
    
    # 记录XML计数奖励
    if rewards[0] > 0:
        logger.log(f"XML计数奖励: {rewards[0]:.3f}")
    
    return rewards

run_name = "Qwen-1.5B-GRPO-gsm8k"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

dataset = get_gsm8k_questions(cahce_dir=output_dir)

logger.log(f"数据集加载完成，样本数量: {len(dataset)}")
logger.log(f"使用模型: {model_name}")
logger.log(f"运行名称: {run_name}")

training_args = GRPOConfig(
    # 如果要使用vllm需要本地 启动 vLLM 服务器
    # use_vllm= True,
    # vllm_mode="colocate",  # 使用离线模式（如果支持）
    # vllm_model_impl="flash_attention_2",

    output_dir=os.path.join(output_dir, 'GRPO-Qwen'),
    run_name=run_name,

    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',

    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,
    num_generations=4, # 一次只生成 4 次
    generation_batch_size=4,

    max_prompt_length=256,
    max_completion_length=786,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="tensorboard",
    log_on_each_node=False,
    save_only_model=True,
)

logger.log(f"训练参数配置完成")
logger.log(f"学习率: {training_args.learning_rate}")
logger.log(f"训练轮数: {training_args.num_train_epochs}")
logger.log(f"每步保存: {training_args.save_steps}")

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)

logger.log(f"LoRA配置完成: r={peft_config.r}, alpha={peft_config.lora_alpha}")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
        cache_dir='/root/autodl-tmp/Model/'
    ).to("cuda")
    logger.log(f"模型加载成功，设备: cuda")
except Exception as e:
    logger.log(f"模型加载失败: {str(e)}")
    raise

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/root/autodl-tmp/Model/')
tokenizer.pad_token = tokenizer.eos_token
logger.log(f"分词器加载完成")

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func],
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config
)

logger.log("开始训练...")
logger.log(f"奖励函数数量: {len(trainer.reward_funcs)}")

try:
    trainer.train()
    logger.log("训练完成！")
    
    # 保存模型
    lora_save_path = os.path.join(output_dir, 'lora_weights')
    trainer.save_model(output_dir=lora_save_path)
    logger.log(f"LoRA权重已保存到: {lora_save_path}")
    
    # 记录训练总结
    logger.log("=" * 80)
    logger.log("训练总结:")
    logger.log(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"模型保存路径: {lora_save_path}")
    logger.log(f"总训练步数: {trainer.state.global_step}")
    logger.log("=" * 80)
    
except Exception as e:
    logger.log(f"训练过程中出现错误: {str(e)}")
    import traceback
    logger.log(f"错误详情:\n{traceback.format_exc()}")
    raise

logger.log(f"所有输出已保存到: {logger.txt_file}")
logger.log(f"奖励详情已保存到: {logger.reward_file}")
logger.log("程序执行完毕")