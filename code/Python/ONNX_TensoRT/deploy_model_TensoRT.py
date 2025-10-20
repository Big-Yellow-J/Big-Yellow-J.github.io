import os
import numpy as np
from PIL import Image
import tensorrt as trt

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
                           precision='fp32', 
                           max_batch_size=8, dynamic_shape=None):
    """
    安全的 TensorRT engine 构建函数
    - 在 WSL2 下避免 Builder 因 GPU 内存查询失败返回 nullptr
    """
    try:
        if not os.path.exists(onnx_file_path):
            print(f"ERROR: ONNX file not found: {onnx_file_path}")
            return None

        # 初始化 TensorRT
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        
        # 避免在 WSL2 下立刻查询 GPU memory
        if builder is None:
            print("WARNING: Builder creation failed, returning None")
            return None

        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        config = builder.create_builder_config()

        # 配置构建参数
        config.max_workspace_size = 2 * 1024 * 1024 * 1024  # 2GB
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        # 精度设置
        if precision.lower() == 'fp16':
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                print("WARNING: Platform doesn't support FP16, using FP32")
        elif precision.lower() == 'int8':
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
            else:
                print("WARNING: Platform doesn't support INT8, using FP32")

        # 解析 ONNX
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_file_path, "rb") as f:
            if not parser.parse(f.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return None

        # 动态 shape
        if dynamic_shape:
            profile = builder.create_optimization_profile()
            for input_name, shape_info in dynamic_shape.items():
                if isinstance(shape_info, tuple) and len(shape_info) == 3:
                    min_shape, opt_shape, max_shape = shape_info
                else:
                    min_shape = opt_shape = max_shape = shape_info
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            print("Dynamic shape profile added")

        # 构建 engine（serialized_network 避免 GPU 内存查询失败）
        print("Building engine (serialized)...")
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            print("ERROR: Engine build returned None. Possibly WSL2 GPU memory query issue.")
            return None

        # 保存 engine 文件
        with open(engine_file_path, "wb") as f:
            f.write(engine_bytes)
        print(f"Engine saved to: {engine_file_path}")

        # 反序列化返回可用 engine
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            print("ERROR: Failed to deserialize engine")
        return engine

    except Exception as e:
        print(f"ERROR in build_engine_from_onnx: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    build_engine_from_onnx(
        onnx_file_path="clip_img.onnx",
        engine_file_path="clip_img.engine",
    )
    build_engine_from_onnx(
        onnx_file_path="clip_txt.onnx",
        engine_file_path="clip_txt.engine",
        dynamic_shape={
            "input_ids": ((1,77),(8,77),(32,77)),
            "attention_mask": ((1,77),(8,77),(32,77))
        }
    )

    # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    # clip_engine = CLIPDualEncoder("clip_visual.engine", "clip_text.engine")

    # # ======== 图像编码 ========
    # img = preprocess_image("test.jpg")
    # image_feat = clip_engine.encode_image(img)
    # print("Image embedding shape:", image_feat.shape)

    # # ======== 文本编码 ========
    # text = "a cute cat sitting on the sofa"
    # input_ids, attention_mask = preprocess_text(text, tokenizer)
    # text_feat = clip_engine.encode_text(input_ids, attention_mask)
    # print("Text embedding shape:", text_feat.shape)

    # # ======== 相似度计算 ========
    # image_feat_norm = image_feat / np.linalg.norm(image_feat, axis=-1, keepdims=True)
    # text_feat_norm = text_feat / np.linalg.norm(text_feat, axis=-1, keepdims=True)
    # similarity = np.dot(image_feat_norm, text_feat_norm.T)
    # print("Similarity:", similarity)