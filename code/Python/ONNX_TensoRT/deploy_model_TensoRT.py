import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine_from_onnx(onnx_file_path, engine_file_path, fp16=True, max_batch_size=8, dynamic_shape=None):
    """
    生成 TensorRT engine
    
    onnx_file_path: ONNX 文件路径
    engine_file_path: 输出 engine 文件路径
    fp16: 是否使用 FP16 精度
    max_batch_size: 最大 batch size
    dynamic_shape: dict, 例如 {"input_ids": (1,77,32,77)}
    """
    print(f"Building engine from ONNX: {onnx_file_path}")
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         trt.BuilderConfig() as config:

        builder.max_batch_size = max_batch_size
        config.max_workspace_size = 4 * 1024 * 1024 * 1024  # 4GB，可根据显存调整

        if fp16:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("FP16 enabled")
            else:
                print("FP16 not supported on this GPU, using FP32")

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
    # ================== 示例 ==================
    # 图片 encoder
    build_engine_from_onnx(
        onnx_file_path="clip_img.onnx",
        engine_file_path="clip_img.engine",
        fp16=True
    )

    # 文本 encoder，CLIP 文本输入固定长度 77
    build_engine_from_onnx(
        onnx_file_path="clip_txt.onnx",
        engine_file_path="clip_txt.engine",
        fp16=True,
        dynamic_shape={
            "input_ids": ((1,77),(8,77),(32,77)),
            "attention_mask": ((1,77),(8,77),(32,77))
        }
    )