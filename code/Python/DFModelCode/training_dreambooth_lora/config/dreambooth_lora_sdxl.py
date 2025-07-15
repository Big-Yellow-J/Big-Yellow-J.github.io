from dataclasses import dataclass

@dataclass
class DreamboothSDXL:
    # config
    seed = 1234
    output_dir= '/data/huangjie/DreamBooth-SDXL-Lora-LOL'
    cache_dir= '/data/huangjie'
    logging_dir= 'log'
    instance_data_dir= '/root/gqh/intern/tmp_code/dreambooth_lora/lol-image/'
    enable_xformers_memory_efficient_attention= False
    gradient_checkpointing= False
    output_kohya_format= None

    # config Prior Preservation Loss
    #mark: 使用 with_prior_preservation 必须指定 class_data_dir 以及 class_prompt
    with_prior_preservation= False
    class_data_dir= None                           # 存储训练的类别图片
    class_prompt= None                             # 图片的类别描述prompt
    num_class_images= 100
    prior_generation_precision= 'fp16'
    allow_tf32= False

    # config data
    random_flip= False                             # 随机翻转图像
    center_crop= False                             # 中心裁剪到指定分辨率
    resolution= 1024
    repeats= 1                                     # 训练输出重复次数
    num_workers= 4

    # config validation
    validation_prompt= "A photo of Rengar the Pridestalker driving a car" # a [identifier] [class noun] 
    instance_prompt= "a photo of Rengar the Pridestalker"               # a [class noun] 
    num_validation_images= 4
    resume_from_checkpoint= False
    checkpointing_steps= 500                            # checkpointing_steps 步保存一次模型
    checkpoints_total_limit= 2                          # 只保存 checkpoints_total_limit 个checkpoint
    validation_epochs= 50                               # 50次验证一次

    # config model
    pretrained_model_name_or_path= "stabilityai/stable-diffusion-xl-base-1.0"
    pretrained_vae_model_name_or_path= "madebyollin/sdxl-vae-fp16-fix"
    train_text_encoder= False

    # config lora
    rank= 4
    lora_dropout= 0.0
    use_dora= False

    # config train
    batch_size= 4
    num_train_epochs= 1
    gradient_accumulation_steps= 4

    # config edm https://arxiv.org/pdf/2206.00364
    do_edm_style_training= False
    snr_gamma= None                            # 推荐直接设置 5.0 https://arxiv.org/pdf/2303.09556

    # config optim
    adam_weight_decay= 1e-4
    adam_weight_decay_text_encoder= 1e-3
    learning_rate= 1e-4
    lr_warmup_steps= 0
    lr_num_cycles= 1
    lr_power= 1
    lr_scheduler= 'constant'
    max_grad_norm= 1
    max_train_steps= 500
    scale_lr= False                            # 直接通过bs等参数来处理lr
    use_8bit_adam= False

    # config accelerate
    report_to= 'tensorboard'
    mixed_precision= 'bf16'
    

