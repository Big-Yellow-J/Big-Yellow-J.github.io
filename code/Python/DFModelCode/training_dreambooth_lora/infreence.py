import torch
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype= torch.float16,
                cache_dir= '/data/huangjie/'
            )
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype= torch.float16,
        cache_dir= '/data/huangjie/'
    )
    pipeline.load_lora_weights('/data/huangjie/DreamBooth-SDXL-Lora-LOL/checkpoint-500/pytorch_lora_weights.safetensors')
    pipeline = pipeline.to(device)
    pipeline_config = {"prompt": "a photo of Rengar the Pridestalker drive the car", "num_inference_steps": 50}
    images = [pipeline(**pipeline_config).images[0] 
                    for _ in range(4)]
    for i, _ in enumerate(images):
        images[i].save(f"./out/image-{i}.png")