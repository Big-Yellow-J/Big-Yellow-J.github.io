import time
import torch
import torch.nn.functional as F

device = "cuda"
dtype = torch.float16
B, H, L, D = 4, 16, 2048, 128

q = torch.randn(B, H, L, D, device=device, dtype=dtype)
k = torch.randn(B, H, L, D, device=device, dtype=dtype)
v = torch.randn(B, H, L, D, device=device, dtype=dtype)

N = 50

torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
torch.cuda.synchronize()
print(f"SDPA (很可能 Flash) : {(time.time()-t0)/N*1000:.2f} ms")

torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=False)
torch.cuda.synchronize()
print(f"Naive torch.matmul+mask : {(time.time()-t0)/N*1000:.2f} ms")