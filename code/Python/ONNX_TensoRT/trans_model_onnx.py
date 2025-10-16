import time
import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
 
class ImgModelWrapper(nn.Module):
    def __init__(self, model):
        super(ImgModelWrapper, self).__init__()
        self.model = model

    def forward(self, pixel_values):
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        return image_features

class TxtModelWrapper(nn.Module):
    def __init__(self, model):
        super(TxtModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        text_features = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return text_features

def trans_clip_onnx(clip_model_name, image_path, text= ["a photo of a cat"]):
    # 加载模型
    model = CLIPModel.from_pretrained(clip_model_name,
                                    use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(clip_model_name,
                                            use_safetensors=True)
    
    # 处理输入
    image = Image.open(image_path) 
    inputs = processor(text= text, images=image, return_tensors="pt", padding='max_length')
    
    # 转换ONNX
    img_model = ImgModelWrapper(model)
    txt_model = TxtModelWrapper(model)
    
    torch.onnx.export(img_model,
                    (inputs.pixel_values),
                    "clip_img.onnx",
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=['pixel_values'],
                    )
    torch.onnx.export(txt_model,
                    (inputs.input_ids, inputs.attention_mask),
                    "clip_txt.onnx",
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=['input_ids', 'attention_mask'],
                    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}, 
                                    'attention_mask': {0: 'batch', 1: 'seq'}},
                    )

def test_model_pt(clip_model_name, image_path):
    model = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(clip_model_name, use_safetensors=True)
    model.eval()

    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    s_pt_time = time.time()
    with torch.no_grad():
        pt_features = model.get_image_features(pixel_values=inputs.pixel_values)
    pt_features = pt_features.cpu().numpy()
    print(f"原始推理使用时间：{time.time()- s_pt_time:.2f} 秒")

    s_onnx_time = time.time()
    ort_session = ort.InferenceSession("clip_img.onnx", providers=["CPUExecutionProvider"])
    ort_inputs = {"pixel_values": inputs.pixel_values.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"ONNX 推理使用时间：{time.time()- s_onnx_time:.2f} 秒")

def classify_image_and_compare(clip_model_name, image_path, candidate_labels):
    model = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(clip_model_name, use_safetensors=True)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)

    # ---------------- PyTorch 推理 ----------------
    t0 = time.time()
    with torch.no_grad():
        pt_image_features = model.get_image_features(pixel_values=inputs.pixel_values)   # shape (1, D)
        pt_text_features  = model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)  # shape (N, D)
    pt_time = time.time() - t0

    # 转 numpy 并做 L2 归一化
    pt_image = pt_image_features.cpu().numpy()
    pt_text  = pt_text_features.cpu().numpy()
    pt_image = pt_image / np.linalg.norm(pt_image, axis=-1, keepdims=True)
    pt_text  = pt_text  / np.linalg.norm(pt_text, axis=-1, keepdims=True)

    # 计算相似度（image @ text.T），得到每个候选标签的分数
    pt_sim = (pt_image @ pt_text.T).squeeze(0)
    pt_best_idx = int(np.argmax(pt_sim))
    pt_best_label = candidate_labels[pt_best_idx]
    pt_best_score = float(pt_sim[pt_best_idx])

    # ---------------- ONNX 推理 ----------------
    ort_img_sess = ort.InferenceSession("clip_img.onnx", providers=["CPUExecutionProvider"])
    ort_txt_sess = ort.InferenceSession("clip_txt.onnx", providers=["CPUExecutionProvider"])

    ort_inputs_img = {"pixel_values": inputs.pixel_values.cpu().numpy().astype(np.float32)}
    ort_inputs_txt = {
        "input_ids": inputs.input_ids.cpu().numpy().astype(np.int64),
        "attention_mask": inputs.attention_mask.cpu().numpy().astype(np.int64)
    }

    t1 = time.time()
    ort_img_out = ort_img_sess.run(None, ort_inputs_img)
    ort_txt_out = ort_txt_sess.run(None, ort_inputs_txt)
    onnx_time = time.time() - t1

    onnx_image = ort_img_out[0]
    onnx_text  = ort_txt_out[0] 
    onnx_image = onnx_image / np.linalg.norm(onnx_image, axis=-1, keepdims=True)
    onnx_text  = onnx_text  / np.linalg.norm(onnx_text, axis=-1, keepdims=True)

    onnx_sim = (onnx_image @ onnx_text.T).squeeze(0)
    onnx_best_idx = int(np.argmax(onnx_sim))
    onnx_best_label = candidate_labels[onnx_best_idx]
    onnx_best_score = float(onnx_sim[onnx_best_idx])

    # ---------------- 输出对比 ----------------
    print("=== PyTorch 结果 ===")
    print(f"预测标签: {pt_best_label}")
    print(f"相似度(score): {pt_best_score:.6f}")
    print(f"推理时间: {pt_time:.6f} 秒 (包含 image & text 推理)")

    print("\n=== ONNX 结果 ===")
    print(f"预测标签: {onnx_best_label}")
    print(f"相似度(score): {onnx_best_score:.6f}")
    print(f"推理时间: {onnx_time:.6f} 秒 (包含 image & text 推理)")

if __name__ == '__main__':
    clip_model = "openai/clip-vit-base-patch32"
    image_path = "./test_2.jpg"

    trans_clip_onnx(clip_model, image_path)
    candidate_labels = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a car",
        "a photo of a building",
        "a photo of a person"
    ]
    classify_image_and_compare(clip_model, image_path, candidate_labels)