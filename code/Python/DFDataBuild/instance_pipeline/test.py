from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


pipe = pipeline(task="depth-estimation", 
                model="LiheYoung/depth-anything-base-hf",
                cache_dir= '/data/huangjie/')

# 加载图像
image_path = "../image/sa_325551.jpg"  # 请替换为您的图像路径
image = Image.open(image_path).convert("RGB")

# 进行深度估计
result = pipe(image)
depth_map = result["depth"]  # PIL Image 类型

# 转换为 numpy 数组以便可视化
depth_np = np.array(depth_map)

# 归一化深度图到 [0, 1] 区间用于显示
depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6)

# 可视化原图和深度图
plt.figure(figsize=(12, 6))

# 原图
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

# 深度图
plt.subplot(1, 2, 2)
plt.title("Depth Map")
im = plt.imshow(depth_norm, cmap="plasma")  # 保存 imshow 返回值以添加 colorbar
plt.colorbar(im, fraction=0.046, pad=0.04)  # 控制位置和大小
plt.axis("off")


# 保存和显示
plt.savefig("depth_estimation_output.jpg")