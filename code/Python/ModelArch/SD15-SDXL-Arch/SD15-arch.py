import torch
from diffusers import StableDiffusionPipeline
import time

log_path = "./sd15_layer_shapes_detailed.log"
log_file = open(log_path, "w")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    cache_dir='/data/'
).to("cuda")

def make_hook(full_name):
    def hook(module, input, output):
        module_type = module.__class__.__name__
        
        input_shapes = []
        input_dtypes = []
        for x in input:
            if isinstance(x, torch.Tensor):
                input_shapes.append(tuple(x.shape))
                input_dtypes.append(str(x.dtype))
            else:
                input_shapes.append("Non-Tensor")
                input_dtypes.append("N/A")
        
        if isinstance(output, (tuple, list)):
            output_shapes = [tuple(o.shape) if isinstance(o, torch.Tensor) else "Non-Tensor" for o in output]
            output_dtypes = [str(o.dtype) if isinstance(o, torch.Tensor) else "N/A" for o in output]
        else:
            output_shapes = [tuple(output.shape) if isinstance(output, torch.Tensor) else "Non-Tensor"]
            output_dtypes = [str(output.dtype) if isinstance(output, torch.Tensor) else "N/A"]
        
        msg = f"\n[HOOK] {full_name}\n"
        msg += f"  Module Type  : {module_type}\n"
        msg += f"  Input Shapes : {input_shapes}\n"
        msg += f"  Input DTypes : {input_dtypes}\n"
        msg += f"  Output Shapes: {output_shapes}\n"
        msg += f"  Output DTypes: {output_dtypes}\n"
        msg += "-" * 80 + "\n"
        
        print(msg.strip())
        log_file.write(msg)
        log_file.flush()
    return hook

unet = pipe.unet

# hook 主要模块
unet.conv_in.register_forward_hook(make_hook("conv_in"))
unet.conv_out.register_forward_hook(make_hook("conv_out"))

for idx, block in enumerate(unet.down_blocks):
    block.register_forward_hook(make_hook(f"down_blocks.{idx}"))

if hasattr(unet, "mid_block"):
    unet.mid_block.register_forward_hook(make_hook("mid_block"))

for idx, block in enumerate(unet.up_blocks):
    block.register_forward_hook(make_hook(f"up_blocks.{idx}"))

# 推理
prompt = "a fantasy castle at sunset, highly detailed, realistic"
start_time = time.time()
with torch.autocast("cuda"):
    _ = pipe(prompt, num_inference_steps=1).images[0]
end_time = time.time()

msg = f"\nInference Time: {end_time - start_time:.2f} seconds\n"
log_file.write(msg)
log_file.close()