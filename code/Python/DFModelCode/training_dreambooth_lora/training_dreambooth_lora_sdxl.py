import gc
import os
import json
import math
import shutil
import logging
import warnings
import argparse
import itertools
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import (
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
    is_peft_version,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from config.dreambooth_lora_sdxl import DreamboothSDXL
from build_data import DreamBoothDataset, PromptDataset, collate_fn
warnings.filterwarnings('ignore')
logger = get_logger(__name__)
config = DreamboothSDXL()

def determine_scheduler_type(pretrained_model_name_or_path):
    model_index_filename = "model_index.json"
    if os.path.isdir(pretrained_model_name_or_path):
        model_index = os.path.join(pretrained_model_name_or_path, model_index_filename)
    else:
        model_index = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=model_index_filename
        )

    with open(model_index, "r") as f:
        scheduler_type = json.load(f)["scheduler"][1]
    return scheduler_type

def log_validation(
    pipeline,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {config.num_validation_images} images with prompt:"
        f" {config.validation_prompt}." #mark: 这个参数使用自定义的
    )

    scheduler_args = {}
    if not config.do_edm_style_training:
        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed) if config.seed is not None else None
    if torch.backends.mps.is_available() or "playground" in config.pretrained_model_name_or_path:
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] 
                  for _ in range(config.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    '''加载模型'''
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, cache_dir= config.cache_dir
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")
    
def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    '''将prompt进行编码'''
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def main(validation_prompt= None, instance_prompt= None):
    # prompt设置
    config.validation_prompt = validation_prompt if validation_prompt else config.validation_prompt
    config.instance_prompt = instance_prompt if instance_prompt else config.instance_prompt

    logging_dir = Path(config.output_dir, config.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir= config.output_dir, logging_dir= logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps= config.gradient_accumulation_steps,
        mixed_precision= config.mixed_precision,
        log_with= config.report_to,
        project_config= accelerator_project_config,
        kwargs_handlers= [kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    if config.seed is not None:
        set_seed(config.seed)
    
    # Prior Preservation Loss处理
    if config.with_prior_preservation:
        class_images_dir = Path(config.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < config.num_class_images:
            has_supported_fp16_accelerator = torch.cuda.is_available() or torch.backends.mps.is_available()
            torch_dtype = torch.float16 if has_supported_fp16_accelerator else torch.float32
            if config.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif config.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif config.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                config.pretrained_model_name_or_path,
                torch_dtype=torch_dtype, cache_dir= config.cache_dir
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = config.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(config.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, 
                                                            batch_size= config.batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
    
    # 加载tokenizer
    tokenizer_one = AutoTokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        cache_dir= config.cache_dir,
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        cache_dir= config.cache_dir,
        use_fast=False,
    )
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        config.pretrained_model_name_or_path,
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        config.pretrained_model_name_or_path, 
        subfolder="text_encoder_2",
    )

    # 加载模型
    scheduler_type = determine_scheduler_type(config.pretrained_model_name_or_path)
    if "EDM" in scheduler_type:
        config.do_edm_style_training = True
        noise_scheduler = EDMEulerScheduler.from_pretrained(config.pretrained_model_name_or_path, 
                                                            subfolder="scheduler",
                                                            cache_dir= config.cache_dir)
        logger.info("Performing EDM-style training!")
    elif config.do_edm_style_training:
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            config.pretrained_model_name_or_path, 
            subfolder="scheduler",
            cache_dir= config.cache_dir
        )
        logger.info("Performing EDM-style training!")
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, 
                                                        cache_dir= config.cache_dir,
                                                        subfolder="scheduler")

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        config.pretrained_model_name_or_path, 
        subfolder="text_encoder",
        cache_dir= config.cache_dir
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        config.pretrained_model_name_or_path, 
        subfolder="text_encoder_2",
        cache_dir= config.cache_dir
    )
    vae_path = (
        config.pretrained_model_name_or_path
        if config.pretrained_vae_model_name_or_path is None
        else config.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if config.pretrained_vae_model_name_or_path is None else None, cache_dir= config.cache_dir
    )
    latents_mean = latents_std = None
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
    if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
        latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)

    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="unet", cache_dir= config.cache_dir
    )
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                    "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if config.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()
    
    def get_lora_config(rank, dropout, use_dora, target_modules):
        '''lora config'''
        base_config = {
            "r": rank,
            "lora_alpha": rank,
            "lora_dropout": dropout,
            "init_lora_weights": "gaussian",
            "target_modules": target_modules,
        }
        if use_dora:
            if is_peft_version("<", "0.9.0"):
                raise ValueError(
                    "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
                )
            else:
                base_config["use_dora"] = True

        return LoraConfig(**base_config)
    
    # lora config
    unet_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    unet_lora_config = get_lora_config(
        rank= config.rank,
        dropout= config.lora_dropout,
        use_dora= config.use_dora,
        target_modules= unet_target_modules,
    )
    unet.add_adapter(unet_lora_config)
    if config.train_text_encoder:
        text_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        text_lora_config = get_lora_config(
            rank= config.rank,
            dropout= config.lora_dropout,
            use_dora= config.use_dora,
            target_modules= text_target_modules,
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)

    # 保存模型
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(model, type(unwrap_model(text_encoder_two))):
                    text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers= unet_lora_layers_to_save,
                text_encoder_lora_layers= text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers= text_encoder_two_lora_layers_to_save,
            )
    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")
        lora_state_dict, network_alphas = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)

        unet_state_dict = {f"{k.replace('unet.', '')}": v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate * config.gradient_accumulation_steps * config.batch_size * accelerator.num_processes
        )
    
    if config.mixed_precision == "fp16":
        models = [unet]
        if config.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])
        cast_training_params(models, dtype=torch.float32)

    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if config.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
        text_lora_parameters_two = list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))

    # 优化模型参数
    unet_lora_parameters_with_lr = {"params": unet_lora_parameters, "lr": config.learning_rate}
    if config.train_text_encoder:
        text_lora_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": config.adam_weight_decay_text_encoder,
            "lr": config.text_encoder_lr if config.text_encoder_lr else config.learning_rate,
        }
        text_lora_parameters_two_with_lr = {
            "params": text_lora_parameters_two,
            "weight_decay": config.adam_weight_decay_text_encoder,
            "lr": config.text_encoder_lr if config.text_encoder_lr else config.learning_rate,
        }
        params_to_optimize = [
            unet_lora_parameters_with_lr,
            text_lora_parameters_one_with_lr,
            text_lora_parameters_two_with_lr,
        ]
    else:
        params_to_optimize = [unet_lora_parameters_with_lr]
    
    if config.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
            params_to_optimize,
            weight_decay= config.adam_weight_decay)
    
    # 数据集构建
    train_dataset = DreamBoothDataset(
        config= config,
        data_root= config.instance_data_dir,
        prompt= config.instance_prompt,
        class_prompt= config.class_prompt,
        class_data_root= config.class_data_dir if config.with_prior_preservation else None,
        class_num= config.num_class_images,
        size= config.resolution,
        repeats= config.repeats,
        center_crop= config.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= config.batch_size,
        shuffle= True,
        collate_fn= lambda examples: collate_fn(examples, config.with_prior_preservation),
        num_workers= config.num_workers,
    )

    def compute_time_ids(original_size, crops_coords_top_left):
        target_size = (config.resolution, config.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    if not config.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds
        
    if not config.train_text_encoder and not train_dataset.custom_instance_prompts:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(
            config.instance_prompt, text_encoders, tokenizers
        )

    # Handle class prompt for prior-preservation.
    if config.with_prior_preservation:
        if not config.train_text_encoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds = compute_text_embeddings(
                config.class_prompt, text_encoders, tokenizers
            )

    # Clear the memory here
    if not config.train_text_encoder and not train_dataset.custom_instance_prompts:
        del tokenizers, text_encoders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not train_dataset.custom_instance_prompts:
        if not config.train_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            unet_add_text_embeds = instance_pooled_prompt_embeds
            if config.with_prior_preservation:
                prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
                unet_add_text_embeds = torch.cat([unet_add_text_embeds, class_pooled_prompt_embeds], dim=0)
        else:
            tokens_one = tokenize_prompt(tokenizer_one, config.instance_prompt)
            tokens_two = tokenize_prompt(tokenizer_two, config.instance_prompt)
            if config.with_prior_preservation:
                class_tokens_one = tokenize_prompt(tokenizer_one, config.class_prompt)
                class_tokens_two = tokenize_prompt(tokenizer_two, config.class_prompt)
                tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
                tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)
    
    num_warmup_steps_for_scheduler = config.lr_warmup_steps * accelerator.num_processes
    if config.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / config.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            config.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = config.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer= optimizer,
        num_warmup_steps= num_warmup_steps_for_scheduler,
        num_training_steps= num_training_steps_for_scheduler,
        num_cycles= config.lr_num_cycles,
        power= config.lr_power,
    )

    if config.train_text_encoder:
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != config.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_name = (
            "dreambooth-lora-sd-xl"
            if "playground" not in config.pretrained_model_name_or_path
            else "dreambooth-lora-playground"
        )
        accelerator.init_trackers(tracker_name, config=vars(config))
    
    # 训练
    total_batch_size = config.batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    for epoch in range(first_epoch, config.num_train_epochs):
        unet.train()
        if config.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()
            accelerator.unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)
            accelerator.unwrap_model(text_encoder_two).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                prompts = batch["prompts"]

                # encode batch prompts when custom prompts are provided for each image -
                if train_dataset.custom_instance_prompts:
                    if not config.train_text_encoder:
                        prompt_embeds, unet_add_text_embeds = compute_text_embeddings(
                            prompts, text_encoders, tokenizers
                        )
                    else:
                        tokens_one = tokenize_prompt(tokenizer_one, prompts)
                        tokens_two = tokenize_prompt(tokenizer_two, prompts)

                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()

                if latents_mean is None and latents_std is None:
                    model_input = model_input * vae.config.scaling_factor
                    if config.pretrained_vae_model_name_or_path is None:
                        model_input = model_input.to(weight_dtype)
                else:
                    latents_mean = latents_mean.to(device=model_input.device, dtype=model_input.dtype)
                    latents_std = latents_std.to(device=model_input.device, dtype=model_input.dtype)
                    model_input = (model_input - latents_mean) * vae.config.scaling_factor / latents_std
                    model_input = model_input.to(dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                if not config.do_edm_style_training:
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    timesteps = timesteps.long()
                else:
                    indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                    timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                if config.do_edm_style_training:
                    sigmas = get_sigmas(timesteps, len(noisy_model_input.shape), noisy_model_input.dtype)
                    if "EDM" in scheduler_type:
                        inp_noisy_latents = noise_scheduler.precondition_inputs(noisy_model_input, sigmas)
                    else:
                        inp_noisy_latents = noisy_model_input / ((sigmas**2 + 1) ** 0.5)

                # time ids
                add_time_ids = torch.cat(
                    [
                        compute_time_ids(original_size=s, crops_coords_top_left=c)
                        for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                    ]
                )

                # Calculate the elements to repeat depending on the use of prior-preservation and custom captions.
                if not train_dataset.custom_instance_prompts:
                    elems_to_repeat_text_embeds = bsz // 2 if config.with_prior_preservation else bsz
                else:
                    elems_to_repeat_text_embeds = 1

                # Predict the noise residual
                if not config.train_text_encoder:
                    unet_added_conditions = {
                        "time_ids": add_time_ids,
                        "text_embeds": unet_add_text_embeds.repeat(elems_to_repeat_text_embeds, 1),
                    }
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                    model_pred = unet(
                        inp_noisy_latents if config.do_edm_style_training else noisy_model_input,
                        timesteps,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions,
                        return_dict=False,
                    )[0]
                else:
                    unet_added_conditions = {"time_ids": add_time_ids}
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[tokens_one, tokens_two],
                    )
                    unet_added_conditions.update(
                        {"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)}
                    )
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                    model_pred = unet(
                        inp_noisy_latents if config.do_edm_style_training else noisy_model_input,
                        timesteps,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions,
                        return_dict=False,
                    )[0]

                weighting = None
                if config.do_edm_style_training:
                    # Similar to the input preconditioning, the model predictions are also preconditioned
                    # on noised model inputs (before preconditioning) and the sigmas.
                    # Follow: Section 5 of https://huggingface.co/papers/2206.00364.
                    if "EDM" in scheduler_type:
                        model_pred = noise_scheduler.precondition_outputs(noisy_model_input, model_pred, sigmas)
                    else:
                        if noise_scheduler.config.prediction_type == "epsilon":
                            model_pred = model_pred * (-sigmas) + noisy_model_input
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            model_pred = model_pred * (-sigmas / (sigmas**2 + 1) ** 0.5) + (
                                noisy_model_input / (sigmas**2 + 1)
                            )
                    # We are not doing weighting here because it tends result in numerical problems.
                    # See: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
                    # There might be other alternatives for weighting as well:
                    # https://github.com/huggingface/diffusers/pull/7126#discussion_r1505404686
                    if "EDM" not in scheduler_type:
                        weighting = (sigmas**-2.0).float()

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = model_input if config.do_edm_style_training else noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = (
                        model_input
                        if config.do_edm_style_training
                        else noise_scheduler.get_velocity(model_input, noise, timesteps)
                    )
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if config.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute prior loss
                    if weighting is not None:
                        prior_loss = torch.mean(
                            (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                                target_prior.shape[0], -1
                            ),
                            1,
                        )
                        prior_loss = prior_loss.mean()
                    else:
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                if config.snr_gamma is None:
                    if weighting is not None:
                        loss = torch.mean(
                            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                                target.shape[0], -1
                            ),
                            1,
                        )
                        loss = loss.mean()
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://huggingface.co/papers/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    base_weight = (
                        torch.stack([snr, config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective needs to be floored to an SNR weight of one.
                        mse_loss_weights = base_weight + 1
                    else:
                        # Epsilon and sample both use the same loss weights.
                        mse_loss_weights = base_weight

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                if config.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + config.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet_lora_parameters, text_lora_parameters_one, text_lora_parameters_two)
                        if config.train_text_encoder
                        else unet_lora_parameters
                    )
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % config.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(config.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(config.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= config.max_train_steps:
                break

        if accelerator.is_main_process:
            if config.validation_prompt is not None and epoch % config.validation_epochs == 0:
                # create pipeline
                if not config.train_text_encoder:
                    text_encoder_one = text_encoder_cls_one.from_pretrained(
                        config.pretrained_model_name_or_path,
                        subfolder="text_encoder",
                        cache_dir= config.cache_dir
                    )
                    text_encoder_two = text_encoder_cls_two.from_pretrained(
                        config.pretrained_model_name_or_path,
                        subfolder="text_encoder_2",
                        cache_dir= config.cache_dir
                    )
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    config.pretrained_model_name_or_path,
                    vae= vae,
                    text_encoder= accelerator.unwrap_model(text_encoder_one),
                    text_encoder_2= accelerator.unwrap_model(text_encoder_two),
                    unet= accelerator.unwrap_model(unet),
                    torch_dtype= weight_dtype,
                    cache_dir= config.cache_dir
                )
                pipeline_config = {"prompt": config.validation_prompt}

                images = log_validation(
                    pipeline,
                    accelerator,
                    pipeline_config,
                    epoch,
                )

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        if config.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_one.to(torch.float32))
            )
            text_encoder_two = unwrap_model(text_encoder_two)
            text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_two.to(torch.float32))
            )
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory= config.output_dir,
            unet_lora_layers= unet_lora_layers,
            text_encoder_lora_layers= text_encoder_lora_layers,
            text_encoder_2_lora_layers= text_encoder_2_lora_layers,
        )
        if config.output_kohya_format:
            lora_state_dict = load_file(f"{config.output_dir}/pytorch_lora_weights.safetensors")
            peft_state_dict = convert_all_state_dict_to_peft(lora_state_dict)
            kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
            save_file(kohya_state_dict, f"{config.output_dir}/pytorch_lora_weights_kohya.safetensors")

        # Final inference
        # Load previous pipeline
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if config.pretrained_vae_model_name_or_path is None else None,
            torch_dtype=weight_dtype,
            cache_dir= config.cache_dir
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            config.pretrained_model_name_or_path,
            vae=vae,
            torch_dtype=weight_dtype,
            cache_dir= config.cache_dir
        )

        # load attention processors
        pipeline.load_lora_weights(config.output_dir)

        # run inference
        images = []
        for i, prompt in enumerate(["A photo of Rengar the Pridestalker driving a car", "A photo of Rengar the Pridestalker in a car",
                       "A photo of Rengar the Pridestalker sitting on a chair"]):
        # if config.validation_prompt and config.num_validation_images > 0:
            # pipeline_config = {"prompt": config.validation_prompt, "num_inference_steps": 25}
            pipeline_config = {"prompt": prompt, "num_inference_steps": 25}
            images = log_validation(
                pipeline,
                accelerator,
                pipeline_config,
                epoch,
                is_final_validation=True,
            )
            for j, tmp_image in enumerate(images):
                tmp_image.save(f"image-{i}-{j}.png")

    accelerator.end_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run main function with parameters")
    parser.add_argument('--validation_prompt', 
                        default="A photo of sks dog in a bucket", 
                        type= str, 
                        help='A prompt that is used during validation to verify that the model is learning.')
    parser.add_argument('--instance_prompt', 
                        default= "a photo of sks dog", 
                        type= str, 
                        help= "The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'")
    args = parser.parse_args()
    # CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes=2 --main_process_port=29600 training_dreambooth_lora_sdxl.py
    main()