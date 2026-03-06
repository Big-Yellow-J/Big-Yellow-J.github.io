from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_path = "/root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/Model/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/"
lora_path = "/root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/Model/Outputs/20260305-Qwen-GRPO-Math-6612/checkpoint-18500/ref/"
save_path = "//root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/Model/MergeModel"

base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype="auto", device_map="cpu")
model = PeftModel.from_pretrained(base, lora_path)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_path)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)