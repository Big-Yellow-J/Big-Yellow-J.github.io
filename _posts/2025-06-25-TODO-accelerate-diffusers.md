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

https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py#L342