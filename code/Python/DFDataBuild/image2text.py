import os
import random
import json
import requests
import torch
from PIL import Image
import multiprocessing as mp
from tqdm import tqdm
from multiprocessing import Process, Manager
from transformers import BlipProcessor, BlipForConditionalGeneration

os.environ["HF_DATASETS_CACHE"] = "/data/huangjie/"
os.environ["HF_HOME"] = "/data/huangjie/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/huangjie/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def open_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def open_image(image_path, local=True):
    if local:
        return Image.open(image_path).convert("RGB")
    else:
        return Image.open(requests.get(image_path, stream=True).raw).convert("RGB")

def model_server(model_name, request_queue, result_queue):
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    
    with torch.no_grad():
        while True:
            try:
                image_path, text = request_queue.get()
                if image_path is None:
                    break
                try:
                    image = open_image(image_path)
                    inputs = processor(image, text=text, return_tensors="pt") if text else processor(image, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    generated_ids = model.generate(**inputs, max_new_tokens=50)
                    caption = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
                    result_queue.put((os.path.basename(image_path), caption))
                except Exception as e:
                    print(f"[Error] {image_path}: {e}")
                    result_queue.put((os.path.basename(image_path), None))
            except Exception as e:
                print(f"[Server Error]: {e}")
                result_queue.put((None, None))

def worker(image_paths, prompt_list, request_queue, result_queue):
    results = []
    with tqdm(total= len(image_paths)) as pbar:
        for path in image_paths:
            text= random.choice(prompt_list) if prompt_list is not None else None
            request_queue.put((path, text))
            name, caption = result_queue.get()
            if caption:
                results.append({"image": name, "caption": caption, "prompt": text})
            pbar.update(1)
    return results

def main(
    image_dir,
    json_path= None,
    model_name="Salesforce/blip-image-captioning-large",
    output_file="result_image2text.json",
    num_processes=2,
):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    prompt_list = open_json(json_path) if json_path else None

    manager = Manager()
    request_queue = manager.Queue()
    result_queue = manager.Queue()

    server_process = Process(target=model_server, args=(model_name, request_queue, result_queue))
    server_process.start()

    chunks = chunkify(image_paths, num_processes)
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(worker, 
                               [(chunk, prompt_list, request_queue, result_queue) for chunk in chunks])

    request_queue.put((None, None))
    server_process.join()

    merged = [item for sublist in results for item in sublist]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

'''
"Salesforce/blip-image-captioning-large" 模型比较简单生成的描述也很简单
TODO: 如何只用小参数模型生成详细的图片描述！YOLO+SAM识别得到实体；CLIP实体判断；得到图片中所有的包含的实体+实体关系判断得到不同实体关系+LLM生成描述
'''

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main(image_dir="./image/", 
         json_path='./prompt_image2text.json',
         output_file="result_image2text.json", num_processes=2)