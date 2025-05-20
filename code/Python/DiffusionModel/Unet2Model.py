import os
import re
import imageio
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from diffusers import UNet2DModel, DDPMPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from tqdm.auto import tqdm

@dataclass
class TrainingConfig:
    image_size = 128
    train_batch_size = 16
    eval_batch_size = 16
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"
    output_dir = "ddpm-butterflies-128"
    seed = 0
    dataset_name = "huggan/smithsonian_butterflies_subset"

def transform(examples):
    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return {"images": [preprocess(image.convert("RGB")) for image in examples["image"]]}

def generate(model, nois_image, scheduler, device='cuda', store_path='./Butterfly/'):
    '''无条件生成器：从噪声图像生成最终图像'''
    def save_samples(samples, title="Generated Samples"):
        fig, axes = plt.subplots(1, len(samples), figsize=(len(samples) * 2, 2))
        for i, img in enumerate(samples):
            axes[i].imshow(img.permute(1, 2, 0).numpy())  # RGB 图像，[C, H, W] -> [H, W, C]
            axes[i].axis('off')
        plt.title(title)
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.close()
    
    def generate_gif(path_dir):
        '''生成gif'''
        images = []
        path_list = os.listdir(path_dir)
        def sort_key(filename):
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else filename
        # 按数字排序
        path_list = sorted(path_list, key=sort_key, reverse= True)
        for path in path_list:
            img_path = os.path.join(path_dir, path)
            if os.path.exists(img_path):
                images.append(imageio.v2.imread(img_path))
        if images:
            name = os.path.basename(os.path.normpath(path_dir))
            imageio.mimsave('./Generate_image.gif', images, fps=2)
    os.makedirs(store_path, exist_ok= True)
    model.eval()
    with torch.no_grad():
        x = nois_image.to(device)
        # 采样循环
        for t in scheduler.timesteps:
            t_tensor = torch.full((x.shape[0],), t, dtype=torch.long, device=device)
            pred_noise = model(x, t_tensor).sample
            x = scheduler.step(pred_noise, t, x).prev_sample

            tmp_x = (x.clamp(-1, 1) + 1) / 2
            save_samples(tmp_x.cpu(), f'{store_path}/{t.item()}')
        # 反归一化到 [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        generate_gif(path_dir= store_path)
        return x.cpu()

config = TrainingConfig()
dataset = load_dataset(config.dataset_name, split="train", cache_dir= '/data/DFModelDataset/').with_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, 
                                               shuffle=True)

model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.lr_warmup_steps, 
                                               num_training_steps=len(train_dataloader) * config.num_epochs)

def evaluate(config, epoch, pipeline):
    images = pipeline(batch_size=config.eval_batch_size, generator=torch.Generator(device='cpu').manual_seed(config.seed)).images
    image_grid = make_image_grid(images, rows=4, cols=4)
    os.makedirs(f"{config.output_dir}/samples", exist_ok=True)
    image_grid.save(f"{config.output_dir}/samples/{epoch:04d}.png")

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(mixed_precision=config.mixed_precision, 
                              gradient_accumulation_steps=config.gradient_accumulation_steps, 
                              log_with="tensorboard", project_dir=f"{config.output_dir}/logs")
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)
    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), 
                            disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch}")
        for batch in train_dataloader:
            clean_images = batch["images"]
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                      (clean_images.shape[0],), device=clean_images.device, dtype=torch.int64)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)
    
    n_samples, channels, image_size, image_size = 5, 3, 128, 128
    scheduler = DDPMScheduler.from_pretrained(f'{config.output_dir}/scheduler')
    scheduler.set_timesteps(50)  # DDIM 采样，50 步
    nois_image = torch.randn(n_samples, channels, image_size, image_size).to(clean_images.device)
    generate(model, nois_image, scheduler)

if __name__ == '__main__':
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)