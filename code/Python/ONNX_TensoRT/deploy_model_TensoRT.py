import os
import numpy as np
from PIL import Image
import tensorrt as trt
from trt_infer import CLIPDualEncoder
from transformers import CLIPTokenizer

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
def preprocess_image(img_path, image_size=224):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((image_size, image_size))
    img = np.array(img).astype(np.float32) / 255.0
    img = (img - 0.48145466) / 0.26862954  # CLIP mean/std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_text(text, tokenizer, max_len=77):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=max_len, return_tensors="np")
    return tokens["input_ids"], tokens["attention_mask"]

def build_engine_from_onnx(onnx_file_path, engine_file_path, 
                           precision= 'fp32', 
                           max_batch_size=8, dynamic_shape=None):
    """
    生成 TensorRT engine
    
    onnx_file_path: ONNX 文件路径
    engine_file_path: 输出 engine 文件路径
    fp16: 是否使用 FP16 精度
    max_batch_size: 最大 batch size
    dynamic_shape: dict, 例如 {"input_ids": (1,77,32,77)}
    """
    print(f"Building engine from ONNX: {onnx_file_path}")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = trt.BuilderConfig()
    parser = trt.OnnxParser(network, TRT_LOGGER) 

    builder.max_batch_size = max_batch_size
    config.max_workspace_size = 4 * 1024 * 1024 * 1024  # 4GB，可根据显存调整

    if precision == 'fp32':
        pass
    elif precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)

    # 解析 ONNX
    with open(onnx_file_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 动态 shape 支持
    if dynamic_shape:
        profile = builder.create_optimization_profile()
        for input_name, (min_shape, opt_shape, max_shape) in dynamic_shape.items():
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        print("Dynamic shape profile added:", dynamic_shape)

    engine = builder.build_engine(network, config)
    if engine:
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print(f"Engine saved to: {engine_file_path}")
    else:
        print("ERROR: Engine build failed.")

    return engine

if __name__ == "__main__":
    build_engine_from_onnx(
        onnx_file_path="clip_img.onnx",
        engine_file_path="clip_img.engine",
        fp16=True
    )
    build_engine_from_onnx(
        onnx_file_path="clip_txt.onnx",
        engine_file_path="clip_txt.engine",
        fp16=True,
        dynamic_shape={
            "input_ids": ((1,77),(8,77),(32,77)),
            "attention_mask": ((1,77),(8,77),(32,77))
        }
    )

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_engine = CLIPDualEncoder("clip_visual.engine", "clip_text.engine")

    # ======== 图像编码 ========
    img = preprocess_image("test.jpg")
    image_feat = clip_engine.encode_image(img)
    print("Image embedding shape:", image_feat.shape)

    # ======== 文本编码 ========
    text = "a cute cat sitting on the sofa"
    input_ids, attention_mask = preprocess_text(text, tokenizer)
    text_feat = clip_engine.encode_text(input_ids, attention_mask)
    print("Text embedding shape:", text_feat.shape)

    # ======== 相似度计算 ========
    image_feat_norm = image_feat / np.linalg.norm(image_feat, axis=-1, keepdims=True)
    text_feat_norm = text_feat / np.linalg.norm(text_feat, axis=-1, keepdims=True)
    similarity = np.dot(image_feat_norm, text_feat_norm.T)
    print("Similarity:", similarity)