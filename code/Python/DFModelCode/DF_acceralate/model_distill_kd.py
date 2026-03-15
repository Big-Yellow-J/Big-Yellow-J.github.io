import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from transformers import ViTForImageClassification, ViTImageProcessor

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class StudentModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=100,
                 dim=192, depth=6, heads=3, mlp_dim=384, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.head(x)

# -------------------------- 损失函数 --------------------------
def kd_loss(logits_student, logits_teacher, labels, temperature=4.0, alpha=0.5, distill=True):
    soft_student = F.log_softmax(logits_student / temperature, dim=1)
    ce_loss = F.cross_entropy(logits_student, labels)
    if distill:
        soft_teacher = F.softmax(logits_teacher / temperature, dim=1)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
        return alpha * kl_loss + (1 - alpha) * ce_loss
    return  ce_loss

def dkd_loss(
    logits_student,
    logits_teacher,
    labels,
    temperature=4.0,
    alpha=0.7,     # TCKD 权重，论文常取小值让 NCKD 主导
    beta=2.0       # NCKD 权重，论文推荐 1.0~5.0
):
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = F.softmax(logits_student / temperature, dim=1)

    one_hot_labels = F.one_hot(labels, num_classes=pred_teacher.size(1)).float()

    p_t_teacher = torch.sum(pred_teacher * one_hot_labels, dim=1)  # [B]
    p_t_student  = torch.sum(pred_student  * one_hot_labels, dim=1)  # [B]

    tckd_teacher = torch.stack([p_t_teacher, 1 - p_t_teacher], dim=1)     # [B, 2]
    tckd_student  = torch.stack([p_t_student,  1 - p_t_student],  dim=1)  # [B, 2]

    tckd_loss = F.kl_div(
        torch.log(tckd_student + 1e-8),  # log
        tckd_teacher,
        reduction='batchmean'
    ) * (temperature ** 2)

    non_target_teacher = pred_teacher * (1 - one_hot_labels)          # [B, C]
    non_target_student  = pred_student  * (1 - one_hot_labels)        # [B, C]

    sum_non_target_teacher = non_target_teacher.sum(dim=1, keepdim=True).clamp(min=1e-8)
    sum_non_target_student  = non_target_student.sum(dim=1, keepdim=True).clamp(min=1e-8)

    norm_teacher = non_target_teacher / sum_non_target_teacher
    norm_student  = non_target_student  / sum_non_target_student

    nckd_loss = F.kl_div(
        torch.log(norm_student + 1e-8),
        norm_teacher,
        reduction='batchmean'
    ) * (temperature ** 2)

    total_loss = alpha * tckd_loss + beta * nckd_loss

    ce_loss = F.cross_entropy(logits_student, labels)
    total_loss += ce_loss  # 或用权重：total_loss = alpha*tckd + beta*nckd + gamma*ce

    return total_loss

# -------------------------- 评估函数 --------------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100. * correct / total
    return accuracy

# -------------------------- 主程序 --------------------------
def main(distill=True, distill_way='kd'):
    print(distill, distill_way)
    teacher_model_name = "pkr7098/cifar100-vit-base-patch16-224-in21k"
    cache_dir = "/root/autodl-fs/Model"

    teacher_model = ViTForImageClassification.from_pretrained(
        teacher_model_name, cache_dir=cache_dir
    ).to(device)
    processor = ViTImageProcessor.from_pretrained(
        teacher_model_name, cache_dir=cache_dir
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    # 数据集
    trainset = datasets.CIFAR100(root=f'{cache_dir}/data', train=True, download=True, transform=transform)
    testset  = datasets.CIFAR100(root=f'{cache_dir}/data', train=False, download=True, transform=transform)
    # from torch.utils.data import Subset, random_split
    # trainset = Subset(trainset, range(100))
    # testset = Subset(testset, range(100))

    trainloader = DataLoader(trainset, batch_size=1000, shuffle=True,  num_workers=16, pin_memory=True)
    testloader  = DataLoader(testset,  batch_size=1000, shuffle=False, num_workers=16, pin_memory=True)

    student_model = StudentModel(
        img_size=224, patch_size=16, num_classes=100,
        dim=192, depth=6, heads=3, mlp_dim=384
    ).to(device)

    optimizer = optim.AdamW(student_model.parameters(), lr=1e-6, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    epochs = 100
    temperature = 4.0
    alpha = 0.7
    best_acc = 0.0
    print(f"Start distilling... Teacher: {teacher_model_name}")
    teacher_model.eval()

    for epoch in range(epochs):
        student_model.train()
        total_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            student_logits = student_model(images)

            with torch.no_grad():
                if distill:
                    teacher_outputs = teacher_model(images)
                    teacher_logits = teacher_outputs.logits
                else:
                    teacher_logits = None
            if distill_way == 'kd':
                loss = kd_loss(student_logits, teacher_logits, labels, temperature, alpha, distill)
            elif distill_way == 'dkd':
                loss = dkd_loss(student_logits, teacher_logits, labels)
            else:
                loss = kd_loss(student_logits, teacher_logits, labels, temperature, alpha, distill)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            acc = evaluate(student_model, testloader, device)
            print(f"Epoch {epoch+1:3d} | Avg Loss: {total_loss/len(trainloader):.4f} | Test Acc: {acc:.2f}%")

    print("\nTraining finished.")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    return best_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CIFAR-100 蒸馏实验")
    parser.add_argument('--distill_way', type=str, default='kd',
                        choices=['kd', 'dkd', 'none'],
                        help='蒸馏方式: kd / dkd / none (无蒸馏)')

    parser.add_argument('--output', type=str, default='./out_distill.txt',
                        help='结果保存文件路径')

    args = parser.parse_args()

    print(f"开始实验: distill_way = {args.distill_way}")
    if args.distill_way == 'none':
        distill = False
    else:
        distill = True
    acc = main(distill= distill, distill_way=args.distill_way)
    result_line = f"{args.distill_way:6s} : {acc:.4f}%\n"
    with open(args.output, 'a', encoding='utf-8') as f:
        f.write(result_line)

    print(f"实验完成: {result_line.strip()}")
    print(f"结果已追加到: {args.output}")
    torch.cuda.empty_cache()
