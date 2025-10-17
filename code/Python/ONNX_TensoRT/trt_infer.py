import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import threading
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTInferenceEngine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self._allocate_io_buffers()

        # 线程锁，保证高并发下显存安全
        self.lock = threading.Lock()

    def _load_engine(self, path):
        with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        print(f"[TRT] Engine loaded: {path}")
        return engine

    def _allocate_io_buffers(self):
        """初始化 IO buffer (如果是动态 shape，稍后可重新分配)"""
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            shape = self.engine.get_binding_shape(binding)

            # 动态 shape 会返回 (-1,...)
            volume = max(1, trt.volume(shape))
            host_mem = cuda.pagelocked_empty(volume, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({"name": binding, "host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"name": binding, "host": host_mem, "device": device_mem})

    def _set_input_shape(self, input_data_dict):
        """动态 shape 时设置 context 的输入 shape"""
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                shape = input_data_dict[binding].shape
                idx = self.engine.get_binding_index(binding)
                self.context.set_binding_shape(idx, shape)

    def infer(self, input_data_dict):
        with self.lock:
            # step 1: 动态 shape 支持
            self._set_input_shape(input_data_dict)

            # step 2: 重新分配 buffer（如果输入 shape 变了）
            self._reallocate_if_needed()

            # step 3: 拷贝输入
            for inp in self.inputs:
                np.copyto(inp["host"], input_data_dict[inp["name"]].ravel())
                cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

            # step 4: 推理
            t0 = time.time()
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            t1 = time.time()

            # step 5: 拷贝输出
            results = {}
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
            self.stream.synchronize()

            for out in self.outputs:
                out_shape = self.context.get_binding_shape(self.engine.get_binding_index(out["name"]))
                results[out["name"]] = np.array(out["host"]).reshape(out_shape)

            print(f"[TRT] Inference time: {(t1 - t0)*1000:.2f} ms")
            return results

    def _reallocate_if_needed(self):
        """检测输入 shape 是否变化，如果变化则重新分配显存"""
        for binding in self.engine:
            idx = self.engine.get_binding_index(binding)
            new_shape = self.context.get_binding_shape(idx)
            old_shape = self.engine.get_binding_shape(idx)
            if -1 in old_shape or np.prod(new_shape) != np.prod(old_shape):
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                volume = trt.volume(new_shape)
                for io_list in [self.inputs, self.outputs]:
                    for io in io_list:
                        if io["name"] == binding:
                            io["host"] = cuda.pagelocked_empty(volume, dtype)
                            io["device"] = cuda.mem_alloc(io["host"].nbytes)
                            self.bindings[idx] = int(io["device"])


# ================================
# 两个模型（视觉 + 文本）封装
# ================================
class CLIPDualEncoder:
    def __init__(self, visual_engine_path, text_engine_path):
        self.visual_engine = TRTInferenceEngine(visual_engine_path)
        self.text_engine = TRTInferenceEngine(text_engine_path)

    def encode_image(self, pixel_values: np.ndarray):
        """
        pixel_values: shape = (B, 3, H, W)
        """
        results = self.visual_engine.infer({"pixel_values": pixel_values})
        return results[list(results.keys())[0]]

    def encode_text(self, input_ids: np.ndarray, attention_mask: np.ndarray = None):
        """
        input_ids: shape = (B, seq_len)
        """
        input_dict = {"input_ids": input_ids}
        if attention_mask is not None:
            input_dict["attention_mask"] = attention_mask
        results = self.text_engine.infer(input_dict)
        return results[list(results.keys())[0]]
