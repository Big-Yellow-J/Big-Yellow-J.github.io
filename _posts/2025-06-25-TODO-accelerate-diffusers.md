---
layout: mypost
title: 深入浅出了解生成模型-5：diffuser/accelerate库学习
categories: python
extMath: true
images: true
address: 武汉🏯
show_footer_image: true
tags: [生成模型,diffusion model,python]
show: true
stickie: true
description: 工欲善其事必先利其器，介绍再多的生成模型没有一个好的工具是不行的，因此本位主要介绍几个在生成模型中常用的python库：diffuser/accelerate的基本使用以及代码操作。
---

工欲善其事，必先利其器。即便介绍了再多生成模型，没有趁手的工具也难以施展才华。因此，本文将重点介绍几个在生成模型开发中常用的 Python 库，着重讲解 **Diffusers** 和 **Accelerate** 的基本使用。感谢 Hugging Face 为无数算法工程师提供了强大的开源支持！需要注意的是，官方文档对这两个库已有详尽的说明，本文仅作为一篇简明的使用笔记，抛砖引玉，供参考和交流。

## accelerate
> 推荐直接阅读官方文档：[https://huggingface.co/docs/accelerate/index](https://huggingface.co/docs/accelerate/index)
> [`pip install accelerate`](https://huggingface.co/docs/accelerate/basic_tutorials/install)

介绍之前了解一下这个库是干什么的：这个库主要提供一个快速的分布式训练（避免了直接用torch进行手搓）并且支持各类加速方法：[混合精度训练](https://www.big-yellow-j.top/posts/2025/01/01/mixed-precision.html)、[Deepspeed](https://www.big-yellow-j.top/posts/2025/02/24/deepspeed.html)、梯度累计等

## 一个基本使用场景
一般任务中一个常见的应用场景是：需要实现一个多显卡（这里假设为双显卡）分布式训练，并且使用梯度累计、混合精度训练，并且训练得到的结果通过tensorboard/wandb进行记录，除此之外还需要使用warm-up学习率调整策略，并且我的模型不同模块使用的学习率不同，训练完成之后所有的模型权重要进行保存/读取权重进行测试。那么可以直接通过下面代码进行实现（部分库的导入以及一些参数比如说config直接忽略）

```python
from accelerate import Accelerator
kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)] # 不是必须的
# Step-1 首先初始化 accelerate
accelerator = Accelerator(mixed_precision= 'fp16', 
                            gradient_accumulation_steps= 2,
                            log_with= ['tensorboard', 'wandb'], # 一般来说用一个即可
                            project_dir=os.path.join(config.output_dir, "logs"),
                            kwargs_handlers= kwargs_handlers
                            )
# 仅在主线程上创建文件夹
if accelerator.is_main_process: 
    os.makedirs(config.output_dir, exist_ok=True)
    # 初始化一个实验记录器（此处内容需要注意⭐）
    # accelerator.init_trackers(f"Train-{config.training}")
    log_name = 'Model-Test'
    accelerator.init_trackers(
        project_name= f"Page-Layout-Analysis-{config.pred_heads}",
        init_kwargs={
            "wandb": {
                "name": log_name,
                "dir": os.path.join(config.output_dir, "logs"),
                "config": vars(config)
            }
        }
        )
 
# Step-2 初始化完成之后可以直接将我们需要的内容通过 accelerator.prepare 进行处理
optimizer = torch.optim.AdamW([
        {'params': model.image_model.parameters(), 'lr': 2e-5, 'weight_decay': 1e-4},
        {'params': model.text_model.parameters(), 'lr': 4e-5},
        {'params': [p for n, p in model.named_parameters() 
                    if 'image_model' not in n and 'text_model' not in n], 
        'lr': config.learning_rate, 'weight_decay': 1e-4}, 
    ])
total_steps = config.epochs * len(train_dataloader)
warmup_steps = int(0.15 * total_steps)

# Warmup 调度器：从 0.1*lr 线性增加到 lr
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                        start_factor=0.1, 
                                                        total_iters=warmup_steps
)

# 余弦退火调度器：添加 eta_min 防止学习率过低
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                T_max=total_steps - warmup_steps, 
                                                                eta_min=1e-6
)

cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                T_max=total_steps - warmup_steps)
lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps] 
)
dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

# Step-3 模型训练以及模型优化
total_data = len(dataloader)
for i, batch in enumerate(dataloader):
    with accelerator.accumulate(model): # 梯度累计
        inputs, targets = batch

        # 下面两句可以不用，但是习惯还是直接使用
        inputs = inputs.to(accelerator.device)
        targets = targets.to(accelerator.device)

        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        accelerator.backward(loss)
        if accelerator.sync_gradients: # 进行梯度裁剪
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # 记录一下实验结果
        logs = {
                "Train/loss": loss.item(),
                "Train/lr": optimizer.param_groups[0]['lr'], # 这里是假设模型使用的优化学习率不同 或者直接使用 scheduler.get_last_lr()[0]
                "Train/ACC": acc,
            }
            progress_bar.set_postfix(
                loss=loss.item(),
                acc=acc, f1=f1)
            accelerator.log(logs, step= epoch* total_data+ i)

# Step-3 同步不同进程
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    model = accelerator.unwrap_model(model)
    model.save_pretrained(os.path.join(args.output_dir, "model"))
accelerator.end_training()
```

不过对于上面的代码需要注意如下几个内容
1、追踪器使用：一般多显卡使用过程中通过使用 `accelerator.end_training()` 去结束 `tracker`
2、tqdm使用：一般只需要主进程进行显示进度条，因此一般直接：`tqdm(..., disable=not accelerator.is_local_main_process)`

## diffuser
> 推荐直接阅读官方文档：[https://huggingface.co/docs/diffusers/main/en/index](https://huggingface.co/docs/diffusers/main/en/index)
> [`pip install git+https://github.com/huggingface/diffusers`](https://huggingface.co/docs/diffusers/main/en/installation?install=Python)

### 基本使用
对于[Diffusion Model原理](https://www.big-yellow-j.top/posts/2025/05/19/DiffusionModel.html)理解可以参考，以及直接通过下面[训练一个Diffusion Model代码](https://github.com/shangxiaaabb/ProjectCode/blob/main/code/Python/DFModelTraining/df_training.py)（代码不一定很规范）进行解释。

```python
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps= config.num_train_timesteps,
                            beta_start= config.beta_start, # 两个beta代表加噪权重
                            beta_end= config.beta_end,
                            beta_schedule= 'scaled_linear')
...
# training
for epoch in range(config.epochs):
    for i, batch in enumerate(train_dataloader):
        image = batch["images"]
        ...
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                    (image.shape[0],), 
                                    device=image.device, 
                                    dtype=torch.int64)
            
        noise = torch.randn(image.shape, device= accelerator.device)
        noise_image = noise_scheduler.add_noise(image, noise, timesteps)
        ...
        noise_pred = model(noise_image, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        ...
# eva
def evaluate(..., noise_scheduler, ):
    ...
    noise = torch.randn((config.eval_batch_size, config.channel, config.image_size, config.image_size)) # 可以选择固定随机数种子
    for t in noise_scheduler.timesteps:
        t_tensor = torch.full((noise.shape[0],), 
                                t, 
                                dtype=torch.long, 
                                device= device)
        predicted_noise = model(noise, t_tensor, text_label)
        noise = noise_scheduler.step(predicted_noise, t, noise).prev_sample
    images = (noise.clamp(-1, 1) + 1) / 2
    ...
```

训练过程
**1、加噪处理**：通过选择使用DDPM/DDIM而后将生成的"确定的噪声"添加到图片上 `noise_scheduler.add_noise(image, noise, timesteps)`
![image.png](https://s2.loli.net/2025/06/27/yLPrx7tkdOh3AiD.webp)

**2、模型预测**：通过模型去预测所添加的噪声并且计算loss
生成过程
**3、逐步解噪**：训练好的模型逐步预测噪声之后将其从噪声图片中将噪声剥离出来

### 1、Scheduler
> https://huggingface.co/docs/diffusers/api/schedulers/overview

以[DDPMScheduler](https://github.com/huggingface/diffusers/blob/d7dd924ece56cddf261cd8b9dd901cbfa594c62c/src/diffusers/schedulers/scheduling_ddpm.py#L129)为例主要使用两个功能：
**1、add_noise**（[输入](https://github.com/huggingface/diffusers/blob/d7dd924ece56cddf261cd8b9dd901cbfa594c62c/src/diffusers/schedulers/scheduling_ddpm.py#L501)：`sample、noise、timesteps`）：这个比较简单就是直接：$x=\sqrt{\alpha}x+ \sqrt{1-\alpha}\epsilon$
**2、step**（[输入](https://github.com/huggingface/diffusers/blob/d7dd924ece56cddf261cd8b9dd901cbfa594c62c/src/diffusers/schedulers/scheduling_ddpm.py#L398)：`model_output、timestep、sample`）：step做的就是将上面的add_noise进行逆操作。具体代码处理
* [**Step-1**](https://github.com/huggingface/diffusers/blob/d7dd924ece56cddf261cd8b9dd901cbfa594c62c/src/diffusers/schedulers/scheduling_ddpm.py#L437) 首先计算几个参数：$\alpha、\beta$

```python
alpha_prod_t = self.alphas_cumprod[t]
alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
beta_prod_t = 1 - alpha_prod_t
beta_prod_t_prev = 1 - alpha_prod_t_prev
current_alpha_t = alpha_prod_t / alpha_prod_t_prev
current_beta_t = 1 - current_alpha_t
```

* [**Step-2**](https://github.com/huggingface/diffusers/blob/d7dd924ece56cddf261cd8b9dd901cbfa594c62c/src/diffusers/schedulers/scheduling_ddpm.py#L445) 根据计算得到参数反推$t-1$的计算结果（提供3种类，介绍“epsilon”）$x_0=\frac{x_T- \sqrt{1- \alpha_t}\epsilon}{\alpha_t}$

```pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)```

* [**Step-3**](https://github.com/huggingface/diffusers/blob/d7dd924ece56cddf261cd8b9dd901cbfa594c62c/src/diffusers/schedulers/scheduling_ddpm.py#L469C9-L474C109)：从数学公式上在上一步就可以计算得到，但是在[论文](https://arxiv.org/pdf/2006.11239)中为了更加近似预测结果还会计算：

$$
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_{t}}{1-\bar{\alpha}_{t}}\mathbf{x}_{0}+\frac{\sqrt{\alpha_{t}}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_{t}}\mathbf{x}_{t}
$$

```python
pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
```

区别DDIM的处理过程将DDPM的马尔科夫链替换为非马尔科夫链过程而后进行采样，这样我们就可以每次迭代中跨多个step，从而减少推理迭代次数和时间：

$$
x_{t-1}=\sqrt{\alpha_{t-1}}\left(\frac{x_t-\sqrt{1-\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\alpha_t}}\right)+\sqrt{1-\alpha_{t-1}-\sigma_t^2}\epsilon_\theta(x_t,t)+\sigma_tz
$$

```python
std_dev_t = eta * variance ** (0.5)
pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
 
prev_sample = prev_sample + variance
```

### 2、pipeline
> https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/README.md
> https://huggingface.co/docs/diffusers/v0.34.0/en/api/pipelines/overview#diffusers.DiffusionPipeline

很多论文里面基本都是直接去微调训练好的模型比如说StableDiffusion等，使用别人训练后的就少不了看到 `pipeline`的影子，直接介绍[`StableDiffusionPipeline`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)的构建（文生图pipeline）。

![image.png](https://s2.loli.net/2025/06/21/5eTfQwG6tLDpycv.webp)

参考：
1、https://github.com/huggingface/diffusers/blob/v0.34.0/src/diffusers/pipelines/pipeline_utils.py#L180
