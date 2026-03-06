import os
os.environ["TRANSFORMERS_CACHE"] = "/data/huangjie/"
import json
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from PIL import Image
model= None
processor= None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def open_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_model(model_name='google/pix2struct-base'):
    '''直接加载 image-text-to-text模型'''
    global model, processor
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,).to(device)
    processor = AutoProcessor.from_pretrained(model_name)

if __name__ == '__main__':
    #  CUDA_VISIBLE_DEVICES=2 python image_text2text.py
    load_model()
    image = Image.open('../data/image/sa_324501.jpg')
    prompt = 'Describe this image'
    tmp_inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {}
    for k, v in tmp_inputs.items():
        v = v.to(model.device)
        if torch.is_floating_point(v):
            v = v.to(model.dtype)
        inputs[k] = v

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("Caption:", output)