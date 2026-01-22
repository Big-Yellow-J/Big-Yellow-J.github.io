import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier

max_length = 8192
model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'
data_apth  = '../test_datasets.jsonl'
store_dir  = '../tmp/Qwen3-VL-3B-GPTQ-W8A8/'
cache_dir  = '/root/autodl-tmp/Model'

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name,cache_dir= cache_dir,)
processor = AutoProcessor.from_pretrained(model_name, cache_dir= cache_dir)

def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}

def preprocess(example):
    user_text = (
        """You convert a document image into structured HTML.
            Use ONLY the following elements:
            <h2 data-bbox="x1 y1 x2 y2">Title</h2>
            <p data-bbox="x1 y1 x2 y2">Paragraph</p>
            <div class="image" data-bbox="x1 y1 x2 y2"></div>
            <div class="chart" data-bbox="x1 y1 x2 y2"></div>
            <div class="table" data-bbox="x1 y1 x2 y2"><table>...</table></div>
            <div class="formula" data-bbox="x1 y1 x2 y2">$$LaTeX$$</div>
            <div class="header" data-bbox="x1 y1 x2 y2">Header</div>
            <div class="footer" data-bbox="x1 y1 x2 y2">Footer</div>
            Rules:
            - Output ONLY HTML
            - Wrap everything in ONE <body>...</body>
            - No text outside <body>
            - Keep reading order
            - data-bbox is required
            Convert the image to HTML."""
        )
    # image = example["image"]
    image = "/root/autodl-tmp/Code/Big-Yellow-J.github.io/code/Python/DFModelCode/DF_acceralate/tmp/image/CacheDit-87-5.48.png"
    assistant_content = example["suffix"]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   
         "content": [
             {"type": "image", "image": image},
             {"type": "text", "text": user_text},
         ]},
        {"role": "assistant", "content": assistant_content},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            max_length=max_length,
            truncation=True,
        )

dataset = load_dataset("json", data_files=data_apth, split="train")
dataset = dataset.shuffle(seed=42)
dataset = dataset.map(preprocess, desc="Preprocess",remove_columns=dataset.column_names)
# print(dataset[0])

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear",scheme="W4A16",ignore=["lm_head", "re:visual.*", "re:model.visual.*"],),
    ]

oneshot(
    model=model,
    tokenizer=model_name,
    cache_dir= cache_dir,
    output_dir= store_dir,
    log_dir= f"{store_dir}/logs/",

    dataset=dataset,
    recipe=recipe,

    max_seq_length=max_length,
    trust_remote_code_model=True,
    data_collator=data_collator,
    sequential_targets=["Qwen2_5_VLDecoderLayer"],
)

# Save to disk in compressed-tensors format.
# SAVE_DIR = model_name.rstrip("/").split("/")[-1] + "-W4A16-G128"
# model.save_pretrained(f"{store_dir}/{SAVE_DIR}", save_compressed=True)
# processor.save_pretrained(f"{store_dir}/{SAVE_DIR}")