import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from dataclasses import dataclass

from data_loader import CustomDataset
from loss_function import perceptual_loss, fourier_loss, mask_aware_l1, VGG16FeatureExtractor
from Model.ffc import FFCResNetGenerator, FFCNLayerDiscriminator

@dataclass
class LaMaConfig:
    # 存储
    output_dir = '/data/huangjie/'

    # accerate 初始化
    mixed_precision = "fp16"
    log_writing = "tensorboard"
    gradient_accumulation_steps = 1

    batch_size = 32
    epochs = 10
    lr1 = 0.001
    lr2 = 0.0001

    # 模型初始化
    input_nc = 4
    output_nc = 3

    # loss
    w1 = 10
    w2 = 10
    w3 = 10

config = LaMaConfig()
def train(epoch, data_loader, generate_model, discriminate_model, accelerator, 
          g_optimizer, d_optimizer, lr_scheduler_g=None, lr_scheduler_d=None):
    generate_model.train()
    discriminate_model.train()

    total_data = len(data_loader)
    progress_bar = tqdm(total= total_data, disable= not accelerator.is_main_process, 
                        desc= f"Epoch-TRAIN {epoch}")
    for i, batch in enumerate(data_loader):
        gt = batch['gt'].to(accelerator.device)
        mask = batch['mask'].to(accelerator.device)
        input_tensor = batch['input'].to(accelerator.device)
        with accelerator.accumulate(generate_model):
            fake_img = generate_model(input_tensor)
            comp_img = fake_img * mask + input_tensor * (1 - mask)

            real_logits = discriminate_model(gt)
            fake_logits = discriminate_model(comp_img.detach())

            d_loss_real = loss_function(real_logits, torch.ones_like(real_logits))
            d_loss_fake = loss_function(fake_logits, torch.zeros_like(fake_logits))
            d_loss = (d_loss_real + d_loss_fake) * 0.5

            d_optimizer.zero_grad()
            accelerator.backward(d_loss)
            d_optimizer.step()
            if lr_scheduler_d is not None:
                lr_scheduler_d.step()

            fake_logits_for_g = discriminate_model(comp_img)
            g_adv_loss = loss_function(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
            g_l1_loss = torch.nn.functional.l1_loss(comp_img, gt)  # 可以加重构损失
            g_loss = g_adv_loss + 100 * g_l1_loss  # 权重可调

            g_optimizer.zero_grad()
            accelerator.backward(g_loss)
            g_optimizer.step()
            if lr_scheduler_g is not None:
                lr_scheduler_g.step()

        progress_bar.update(1)
        if accelerator.is_main_process:
            logs = {'Train/D-Loss', d_loss.item(), 'Train/G-Loss', g_loss.item()}
            progress_bar.set_postfix({
                'd_loss': logs['Train/D-Loss'],
                'g_loss': logs['Train/G-Loss']
            })
            accelerator.log(logs)
    progress_bar.close()

# def train(epoch, data_loader, generate_model, discriminate_model, accerator, loss_function, optim):
#     pass

def main():
    # accerate 初始化
    accelerator = Accelerator(mixed_precision= config.mixed_precision, 
                              gradient_accumulation_steps= config.gradient_accumulation_steps,
                              log_with= config.log_writing,
                              project_dir= os.path.join(config.output_dir, f"logs"),)
    
    # data 初始化
    dataset = CustomDataset()
    train_dataloader = DataLoader(dataset, batch_size= config.batch_size, 
                                  num_workers= 8, shuffle= True)
    
    # model 初始化
    generate_model = FFCResNetGenerator(config.input_nc, config.output_nc)
    discriminate_model = FFCNLayerDiscriminator(input_nc= config.output_nc)

    # 优化器 loss函数
    g_optimizer = torch.optim.AdamW(generate_model.parameters(), lr=config.lr1)
    d_optimizer = torch.optim.AdamW(discriminate_model.parameters(), lr=config.lr2)

    total_steps = config.epochs * len(train_dataloader)
    warmup_steps = int(0.1 * total_steps)

    g_warmup = torch.optim.lr_scheduler.LinearLR(g_optimizer, start_factor=0.2, total_iters=warmup_steps)
    g_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    g_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(g_optimizer, schedulers=[g_warmup, g_cosine], milestones=[warmup_steps])

    d_warmup = torch.optim.lr_scheduler.LinearLR(d_optimizer, start_factor=0.2, total_iters=warmup_steps)
    d_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    d_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(d_optimizer, schedulers=[d_warmup, d_cosine], milestones=[warmup_steps])

    vgg_extractor = VGG16FeatureExtractor().to(accelerator.device)

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(f"Train", config= vars(config))
    generate_model, discriminate_model, g_optimizer, d_optimizer, g_lr_scheduler, d_lr_scheduler, vgg_extractor, train_dataloader = accelerator.prepare(
        generate_model,
        discriminate_model,
        g_optimizer,
        d_optimizer,
        g_lr_scheduler,
        d_lr_scheduler,
        vgg_extractor,
        train_dataloader
    )
    
    for epoch in range(config.epochs):
        train(
            epoch,
            train_dataloader,
            generate_model,
            discriminate_model,
            accelerator,
            g_optimizer,
            d_optimizer,
            g_lr_scheduler,
            d_lr_scheduler
        )

if __name__ == '__main__':
    main()