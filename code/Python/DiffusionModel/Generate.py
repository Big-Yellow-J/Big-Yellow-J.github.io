from diffusers import UNet2DModel, DDPMScheduler
import torch
import imageio
import os
import re
import matplotlib.pyplot as plt

def generate_gif(path_dir: list):
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

def generate(model, nois_image, scheduler, device='cuda'):
    '''无条件生成器：从噪声图像生成最终图像'''
    model.eval()
    with torch.no_grad():
        x = nois_image.to(device)
        # 采样循环
        for t in scheduler.timesteps:
            t_tensor = torch.full((x.shape[0],), t, dtype=torch.long, device=device)
            pred_noise = model(x, t_tensor).sample
            x = scheduler.step(pred_noise, t, x).prev_sample

            tmp_x = (x.clamp(-1, 1) + 1) / 2
            save_samples(tmp_x.cpu(), f'./Butterfly/{t.item()}')
        # 反归一化到 [0, 1]
        x = (x.clamp(-1, 1) + 1) / 2
        generate_gif('./Butterfly/')
        return x.cpu()

def save_samples(samples, title="Generated Samples"):
    fig, axes = plt.subplots(1, len(samples), figsize=(len(samples) * 2, 2))
    for i, img in enumerate(samples):
        axes[i].imshow(img.permute(1, 2, 0).numpy())  # RGB 图像，[C, H, W] -> [H, W, C]
        axes[i].axis('off')
    plt.title(title)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

if __name__ == '__main__':
    # 设备和超参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = 128  # 蝴蝶数据集
    channels = 3      # RGB 图像
    n_samples = 5

    # 加载调度器和模型
    scheduler = DDPMScheduler.from_pretrained('./ddpm-butterflies-128/scheduler')
    scheduler.set_timesteps(50)  # DDIM 采样，50 步
    model = UNet2DModel.from_pretrained('./ddpm-butterflies-128/unet').to(device)

    nois_image = torch.randn(n_samples, channels, image_size, image_size).to(device)
    samples = generate(model, nois_image, scheduler, device)
    save_samples(samples, "Generated Butterfly Samples")