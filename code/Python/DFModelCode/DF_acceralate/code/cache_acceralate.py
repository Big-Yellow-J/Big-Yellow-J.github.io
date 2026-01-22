'''
首先安装需要的模块：pip3 install -U cache-dit
'''
import time
import torch
import cache_dit
from diffusers import ZImagePipeline, AutoModel, PyramidAttentionBroadcastConfig
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

def load_model(model_name= 'Tongyi-MAI/Z-Image-Turbo', cache_dir='/root/autodl-tmp/Model'):
    device = "cuda"
    quantization_config = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
    )
    transformer = AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        mirror='https://hf-mirror.com'
    )
    # transformer = transformer.to("cpu")

    quantization_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    text_encoder = AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        subfolder="text_encoder",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        mirror='https://hf-mirror.com'
    )
    # text_encoder = text_encoder.to("cpu")

    pipe = ZImagePipeline.from_pretrained(
        model_name,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    # pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    return pipe

def image_generate(pipeline, prompt, special_name, seed=10086):
    s_time = time.time()
    image = pipeline(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=10,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    e_time = time.time()
    # image.save(f"./{special_name}-{e_time- s_time:.2f}.png")

if __name__ == '__main__':
    import time
    # print(cache_dit.supported_pipelines())
    dit_pipeline = load_model()
    prompt = "Realistic mid-aged male image"

    # image_generate(dit_pipeline, prompt, 'Normal')
    z_image_pipeline = load_model()
    s_time = time.time()
    # z_image_pipeline.transformer.compile()
    for i in range(5):
        image_generate(z_image_pipeline,
                    "Realistic mid-aged male image",
                    None)
    print(f"Used Time: {time.time()- s_time:.2f}")
    
    # cache_dit.enable_cache(dit_pipeline)
    # image_generate(dit_pipeline, prompt, 'DBCache')
    # stats = cache_dit.summary(dit_pipeline)
    # cache_dit.disable_cache(dit_pipeline)