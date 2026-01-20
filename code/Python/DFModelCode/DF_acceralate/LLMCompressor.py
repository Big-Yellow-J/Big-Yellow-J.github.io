from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
]
oneshot(
    model='/root/autodl-tmp/Model/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306',#"Qwen/Qwen2.5-1.5B-Instruct",
    cache_dir= '/root/autodl-tmp/Model',
    dataset="open_platypus",
    recipe=recipe,
    output_dir="./tmp/LLMCompressor-GPTQ-W8A8-0.81",
    log_dir= './tmp/LLMCompressor-GPTQ-W8A8-0.81/log',
    
    max_seq_length=2048,
    num_calibration_samples=512,
)