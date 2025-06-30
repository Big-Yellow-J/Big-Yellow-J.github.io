from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

prompt = "A futuristic city at sunset, cyberpunk style, highly detailed, cinematic lighting"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("output.png")