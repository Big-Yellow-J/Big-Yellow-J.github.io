import torch
import clip
from PIL import Image

# 加载 CLIP 模型（ViT-B/32 模型，速度和效果适中）
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def compute_clip_score(image_path: str, text: str) -> float:
    # 加载图像并预处理
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    # 编码文本
    text_tokens = clip.tokenize([text]).to(device)

    # 计算特征
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        # 归一化
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 计算余弦相似度（CLIP score）
        similarity = (image_features @ text_features.T).item()

    return similarity

# 🔧 示例用法
image_path = "./instance/car-97.png"
text = "a car with wheels and headlights"
score = compute_clip_score(image_path, text)
print(f"CLIP Score: {score:.4f}")
