import torch
import torch.nn as nn
import torch.fft
import torchvision.models as models

# -----------------------------
# Perceptual Loss (VGG16)
# -----------------------------
class VGG16FeatureExtractor(nn.Module):
    def __init__(self, layers=(3, 8, 15, 22)):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slices = nn.ModuleList()
        last_idx = 0
        for l in layers:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(last_idx, l)]))
            last_idx = l
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        features = []
        for slice_ in self.slices:
            x = slice_(x)
            features.append(x)
        return features

# -----------------------------
# Mask-aware L1
# -----------------------------
def mask_aware_l1(pred, target, mask):
    return torch.sum(torch.abs(pred - target) * mask) / (mask.sum() + 1e-8)

# -----------------------------
# Multi-scale Fourier Loss
# -----------------------------
def fourier_loss(pred, target, mask=None, scales=[1, 0.5, 0.25, 0.125], weights=[1.0, 0.5, 0.25, 0.125]):
    loss = 0.0
    for scale, w in zip(scales, weights):
        if scale != 1:
            pred_s = nn.functional.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
            target_s = nn.functional.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            mask_s = nn.functional.interpolate(mask.float(), scale_factor=scale, mode='nearest') if mask is not None else None
        else:
            pred_s, target_s, mask_s = pred, target, mask

        pred_fft = torch.fft.fft2(pred_s, norm='ortho')
        target_fft = torch.fft.fft2(target_s, norm='ortho')

        freq_loss = torch.abs(pred_fft - target_fft)
        if mask_s is not None:
            freq_loss = freq_loss * mask_s

        loss += w * freq_loss.mean()
    return loss

# -----------------------------
# VGG Perceptual Loss
# -----------------------------
def perceptual_loss(pred, target, vgg_extractor, mask=None):
    pred_feats = vgg_extractor(pred)
    target_feats = vgg_extractor(target)
    loss = 0.0
    for pf, tf in zip(pred_feats, target_feats):
        if mask is not None:
            mask_resized = nn.functional.interpolate(mask, size=(pf.shape[2], pf.shape[3]), mode='nearest')
            loss += torch.mean(torch.abs(pf - tf) * mask_resized)
        else:
            loss += torch.mean(torch.abs(pf - tf))
    return loss