import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from modelscope import snapshot_download
from transformers import AutoProcessor, AutoModelForImageTextToText

def load_model_tokenizer(model_name, cache_dir):
    local_path = snapshot_download(model_id= model_name,
                                   cache_dir=cache_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        local_path,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",

    )
    processor = AutoProcessor.from_pretrained(
        local_path, cache_dir=cache_dir,
    )
    return model, processor

if __name__=="__main__":
    model, processor = load_model_tokenizer(
        model_name= "Qwen/Qwen3.5-0.8B",
        cache_dir= "/root/autodl-fs/Model/"
    )
    print("="*100, f"\n{model}\n", "="*100)
    messages = [
        {
            "role": "user",
            "content": [
                # {"type": "image",
                #  "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
                {"type": "text", "text": "写一个快速排序的python代码，我需要最后python代码"}
            ]
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs,
                             max_new_tokens=1024,
                             eos_token_id=[
                                 processor.tokenizer.eos_token_id,
                             ],
                             pad_token_id=processor.tokenizer.eos_token_id,
                             repetition_penalty=1.1,
                             no_repeat_ngram_size=3,
                             )
    print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))