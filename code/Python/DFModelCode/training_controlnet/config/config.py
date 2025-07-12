from dataclasses import dataclass

@dataclass
class ControlnetSD15:
    # basic config
    seed = 1234
    cache_dir = '/data/huangjie/'
    output_dir = '/data/huangjie/SD15-ControlNet'
    log_dir = '/data/huangjie/SD15-ControlNet/logs'
    tracker_project_name = "SD15-Controlnet"

    # config data
    dataset_name = 'raulc0399/open_pose_controlnet'
    column_image = 'image'
    column_text = 'text'
    column_conditioning_image = 'conditioning_image'
    resolution = 512
    proportion_empty_prompts= 0 #"Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement)."
    max_train_samples = None                                                              # 用小批量数据来测试 100
    
    # config accelerator
    gradient_accumulation_steps = 1
    mixed_precision = "fp16"
    report_to = "tensorboard"
    num_validation_images = 4                                                             # 对于每组 validation_image validation_prompt需要生成的图像数量
    validation_image = ['/home/huangjie/Code/DFModelCode/training_controlnet/validation/image_1.jpg', 
                        '/home/huangjie/Code/DFModelCode/training_controlnet/validation/image_2.jpg']
    validation_prompt = [
                         "A sexy woman in athletic wear bikini in a dimly lit gym, holding a ball.",
                         "A sexy young woman is caught in a splash of water, with droplets adhering to their face and hands, set against a backdrop of a wooden"]


    # config model
    revision = None
    tokenizer_name = None
    scale_lr = False
    controlnet_model_name_or_path = None
    pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    variant = None                                                                 # 加载不同精度的模型

    # 精度设置来加速训练节约显存
    enable_xformers_memory_efficient_attention = None
    gradient_checkpointing = False                                                 # 通过gradient_checkpointing来节约显存但是速度减慢
    allow_tf32 = True                                                              # 通过启用allow_tf32（如 RTX 30/40 系列、A100支持的一种混合精度计算方式）来加速
    use_8bit_adam = False
    
    # config training
    train_batch_size = 52
    num_train_epochs = 50
    dataloader_num_workers = 4
    learning_rate = 1e-4
    lr_warmup_steps = 500                    # lr scheduler的迭代次数
    max_train_steps = None                   # 总共需要迭代的步数
    lr_scheduler = "constant"
    lr_num_cycles = 1
    lr_power = 1
    adamw_config = {
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_weight_decay': 1e-2,
        'adam_epsilon': 1e-8
    }
    max_grad_norm = 1.0
    set_grads_to_none = False
    resume_from_checkpoint = None

    checkpointing_steps= 500
    checkpoints_total_limit = None # 只保存多少checkpoint
    validation_steps = 100