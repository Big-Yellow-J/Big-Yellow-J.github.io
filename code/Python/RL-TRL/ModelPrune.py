import gc
import time
import torch
import torch.nn.utils.prune as prune
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum(torch.count_nonzero(p).item() for p in model.parameters())
    return total_params, nonzero_params

def load_model(model_name, cache_dir):
    local_path = snapshot_download(model_id= model_name,
                                   cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        local_path, cache_dir= cache_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(
        local_path, cache_dir= cache_dir
    )
    return model, processor

def prune_model(model, prune_ratio, way= 'unstructured'):
    s_time = time.time()
    total_params, nonzero_params = count_parameters(model)
    print(f"Model before pruning：{total_params} || {nonzero_params}")

    parameters = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            if 'lm_head' in name or 'embed_tokens' in name:
                continue
            parameters.append((module, 'weight'))
    if way== 'unstructured':
        prune.global_unstructured(
            parameters,
            pruning_method=prune.L1Unstructured,
            amount=prune_ratio,
            )
        for module, name in parameters:
            prune.remove(module, name)
    elif way== 'structured':
        prune_ratio = prune_ratio if 0<prune_ratio<1 else prune_ratio/100
        for module, name in parameters:
            prune.ln_structured(
                module,
                name=name,
                amount=prune_ratio,
                n=2,
                dim=0,
            )
            prune.remove(module, name)

    gc.collect()
    torch.cuda.empty_cache()
    total_params, nonzero_params = count_parameters(model)
    print(f"Model after pruning- {ratio}({time.time()- s_time})：{total_params} || {nonzero_params}")
    return model

def model_generate(model, processor, messages):
    s_time = time.time()
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    generate_inputs = {k: v for k, v in inputs.items() if k != 'mm_token_type_ids'}
    outputs = model.generate(**generate_inputs,
                             max_new_tokens=128,
                             eos_token_id=[
                                 processor.tokenizer.eos_token_id,
                             ],
                             pad_token_id=processor.tokenizer.eos_token_id,
                             repetition_penalty=1.1,
                             no_repeat_ngram_size=3,
                             )
    return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]), time.time()- s_time

if __name__ == "__main__":
    model_outputs = {}
    model_name, cache_dir = "Qwen/Qwen3.5-0.8B", "/root/autodl-fs/Model/"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "你是谁"}
            ]
        },
    ]
    model, processor = load_model(model_name= model_name,
                                  cache_dir= cache_dir)
    model_outputs['raw'] = model_generate(model, processor, messages)
    for ratio in [10, 30, 50, 70]:
        pruned_model = prune_model(model, ratio, way='structured')
        model_outputs[ratio] = model_generate(pruned_model, processor, messages)
        del pruned_model
        gc.collect()
        torch.cuda.empty_cache()
    for name in model_outputs:
        model_output = model_outputs[name]
        print(f"{name} （{model_output[1]}）:\n {model_output[0]}")
        print("="*100)