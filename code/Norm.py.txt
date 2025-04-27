import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(1, *normalized_shape))
            self.beta = nn.Parameter(torch.zeros(1, *normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x: torch.Tensor):
        '''layer norm: 直接处理batch之外维度的均值/标准差'''
        norm_dims = list(range(x.dim() - len(self.normalized_shape), x.dim()))
        mean = x.mean(dim=norm_dims, keepdim=True)
        var = x.var(dim=norm_dims, unbiased=False, keepdim=True)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_normalized = x_normalized * self.gamma + self.beta
        return x_normalized

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(RMSNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(1, *normalized_shape))
            self.beta = nn.Parameter(torch.zeros(1, *normalized_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x: torch.Tensor):
        norm_dims = list(range(x.dim() - len(self.normalized_shape), x.dim()))
        rms = torch.sqrt(torch.mean(x**2, dim=norm_dims, keepdim=True) + self.eps)
        x_normalized = x / rms
        if self.elementwise_affine:
            x_normalized = x_normalized * self.gamma + self.beta

        return x_normalized

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, x: torch.Tensor):
        '''直接处理 channel 之外均值/标准差'''
        training = self.training

        if x.dim() == 3:  # (B, T, C) -> 转换为 (B, C, T, 1) 以统一处理
            x = x.transpose(1, 2).unsqueeze(-1)  # (B, C, T, 1)
            is_1d = True
        else:  # (B, C, H, W)
            is_1d = False

        if training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)  # 沿着 B, H, W（或 T）平均
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

            if self.track_running_stats:
                self.num_batches_tracked += 1
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean if self.track_running_stats else x.mean(dim=(0, 2, 3), keepdim=True)
            var = self.running_var if self.track_running_stats else x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_normalized = x_normalized * self.gamma + self.beta

        if is_1d:
            x_normalized = x_normalized.squeeze(-1).transpose(1, 2)  # (B, T, C)

        return x_normalized

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x0 = torch.randn(32, 100, 512).to(device)
    x1 = torch.randn(8, 3, 512, 512).to(device)

    layer_norm0 = LayerNorm((512)).to(device)
    layer_norm1 = LayerNorm((3, 512, 512)).to(device)

    out0 = layer_norm0(x0)
    out1 = layer_norm1(x1)
    print(f"Layer Norm:{out0.shape} {out1.shape}")

    batch_norm0 = BatchNorm(512).to(device)
    batch_norm1 = BatchNorm(3).to(device)
    out0 = batch_norm0(x0)
    out1 = batch_norm1(x1)
    print(f"Batch Norm: {out0.shape} {out1.shape}")

    rmse_norm0 = RMSNorm((512))
    rmse_norm1 = RMSNorm((3, 512, 512))
    rmse_norm0.to(device)
    rmse_norm1.to(device)

    out0 = rmse_norm0(x0)
    out1 = rmse_norm1(x1)
    print("RMSNorm:", out0.shape, out1.shape)
