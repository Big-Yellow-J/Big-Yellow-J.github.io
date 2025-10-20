import time
import onnx
import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")


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

def trans_clip_onnx(
    clip_model_name,
    image_path,
    text=["a photo of a cat"],
    cache_dir='./',
    img_onnx_name=None,
    text_onnx_name=None,
    opset_version=17,
    precision='fp32',          # 支持 'fp32', 'fp16'
    dynamic_shapes=True,
):
    """
    导出 CLIP 模型的图像分支和文本分支为 ONNX，并可指定精度和动态 shape

    参数:
    - clip_model_name: CLIP 模型名称或路径
    - image_path: 输入图片路径
    - text: 文本列表
    - cache_dir: 模型缓存路径
    - img_onnx_name/text_onnx_name: 输出 ONNX 文件名
    - opset_version: ONNX opset 版本
    - precision: 导出精度 'fp32' 或 'fp16'
    - dynamic_shapes: 是否导出动态 shape
    """

    # 加载模型
    model = CLIPModel.from_pretrained(
        clip_model_name,
        use_safetensors=True,
        cache_dir=cache_dir
    )
    processor = CLIPProcessor.from_pretrained(
        clip_model_name,
        use_safetensors=True,
        cache_dir=cache_dir
    )

    # 处理输入
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        padding='max_length'
    )

    # 包装模型
    img_model = ImgModelWrapper(model)
    txt_model = TxtModelWrapper(model)

    # 输出文件名
    img_onnx_name = img_onnx_name or "clip_img.onnx"
    text_onnx_name = text_onnx_name or "clip_txt.onnx"

    # 设置导出 dtype
    export_dtype = torch.float16 if precision == 'fp16' else torch.float32
    if precision == 'fp16':
        img_model = img_model.half()
        txt_model = txt_model.half()
    for k, v in inputs.items():
        if v.dtype.is_floating_point:
            inputs[k] = v.to(export_dtype)

    # 导出图像分支
    torch.onnx.export(
        img_model,
        (inputs.pixel_values,),
        img_onnx_name,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['pixel_values'],
        dynamic_axes={'pixel_values': {0: 'batch', 2: 'height', 3: 'width'}} if dynamic_shapes else None,
    )

    # 导出文本分支
    torch.onnx.export(
        txt_model,
        (inputs.input_ids, inputs.attention_mask),
        text_onnx_name,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq'},
            'attention_mask': {0: 'batch', 1: 'seq'}
        } if dynamic_shapes else None,
    )

    # 检查 ONNX 文件
    for name in [img_onnx_name, text_onnx_name]:
        onnx_model = onnx.load(name)
        onnx.checker.check_model(onnx_model)
        print(f"{name} 检查通过，精度: {precision}, opset: {opset_version}")

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

def classify_image_and_compare(
    clip_model_name,
    image_path,
    candidate_labels,
    cache_dir='./',
    precision='fp32'  # 支持 'fp32' 或 'fp16'
):
    # ---------------- 加载模型 ----------------
    model = CLIPModel.from_pretrained(clip_model_name,
                                      use_safetensors=True,
                                      cache_dir=cache_dir)
    processor = CLIPProcessor.from_pretrained(clip_model_name,
                                              use_safetensors=True,
                                              cache_dir=cache_dir)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)

    # ---------------- PyTorch 推理 ----------------
    # 如果是 FP16，只转换浮点张量
    if precision == 'fp16':
        model = model.half()
        inputs['pixel_values'] = inputs['pixel_values'].half()
        # input_ids / attention_mask 保持 long 类型

    t0 = time.time()
    with torch.no_grad():
        pt_image_features = model.get_image_features(pixel_values=inputs.pixel_values)   # shape (1, D)
        pt_text_features  = model.get_text_features(input_ids=inputs.input_ids,
                                                    attention_mask=inputs.attention_mask)  # shape (N, D)
    pt_time = time.time() - t0

    # 转 numpy 并做 L2 归一化
    pt_image = pt_image_features.cpu().numpy()
    pt_text  = pt_text_features.cpu().numpy()
    pt_image = pt_image / np.linalg.norm(pt_image, axis=-1, keepdims=True)
    pt_text  = pt_text  / np.linalg.norm(pt_text, axis=-1, keepdims=True)

    # 计算相似度
    pt_sim = (pt_image @ pt_text.T).squeeze(0)
    pt_best_idx = int(np.argmax(pt_sim))
    pt_best_label = candidate_labels[pt_best_idx]
    pt_best_score = float(pt_sim[pt_best_idx])

    print("=== PyTorch 结果 ===")
    print(f"预测标签: {pt_best_label}")
    print(f"相似度(score): {pt_best_score:.6f}")
    print(f"推理时间: {pt_time:.6f} 秒 (包含 image & text 推理)")

    # ---------------- ONNX 推理 ----------------
    providers_list = [
        # ("TensorrtExecutionProvider", {"trt_max_workspace_size": 2147483648}),
        ["CUDAExecutionProvider"],
        ["CPUExecutionProvider"]
    ]
    for providers in providers_list:
        try:
            ort_img_sess = ort.InferenceSession("clip_img.onnx", providers=providers)
            ort_txt_sess = ort.InferenceSession("clip_txt.onnx", providers=providers)

            ort_inputs_img = {
                "pixel_values": inputs.pixel_values.cpu().numpy().astype(
                    np.float16 if precision == 'fp16' else np.float32
                )
            }
            ort_inputs_txt = {
                "input_ids": inputs.input_ids.cpu().numpy().astype(np.int64),
                "attention_mask": inputs.attention_mask.cpu().numpy().astype(np.int64)
            }

            t1 = time.time()
            ort_img_out = ort_img_sess.run(None, ort_inputs_img)
            ort_txt_out = ort_txt_sess.run(None, ort_inputs_txt)
            onnx_time = time.time() - t1

            # L2 归一化
            onnx_image = ort_img_out[0] / np.linalg.norm(ort_img_out[0], axis=-1, keepdims=True)
            onnx_text  = ort_txt_out[0] / np.linalg.norm(ort_txt_out[0], axis=-1, keepdims=True)

            # 相似度
            onnx_sim = (onnx_image @ onnx_text.T).squeeze(0)
            onnx_best_idx = int(np.argmax(onnx_sim))
            onnx_best_label = candidate_labels[onnx_best_idx]
            onnx_best_score = float(onnx_sim[onnx_best_idx])

            # ---------------- 输出对比 ----------------
            print(f"\n=== ONNX（使用后端：{providers}） 结果 ===")
            print(f"预测标签: {onnx_best_label}")
            print(f"相似度(score): {onnx_best_score:.6f}")
            print(f"推理时间: {onnx_time:.6f} 秒 (包含 image & text 推理)")

        except Exception as e:
            print(f"Failed with providers={providers}: {e}")


if __name__ == '__main__':
    clip_model = "openai/clip-vit-base-patch32"
    image_path = "./test_1.jpg"

    # trans_clip_onnx(clip_model, image_path, precision= 'fp16')
    candidate_labels = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a car",
        "a photo of a building",
        "a photo of a person"
    ]
    classify_image_and_compare(clip_model, image_path, 
                               candidate_labels, precision= 'fp16')