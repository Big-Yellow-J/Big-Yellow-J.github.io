import torch

# 输入
X1 = torch.tensor(2.0, requires_grad=True)
X2 = torch.tensor(3.0, requires_grad=True)

# forward 计算图
a = X1 * X2
y1 = torch.log(a)
y2 = torch.sin(X2)
w = y1 * y2
z = w 

# 对中间变量启用梯度保存
a.retain_grad()
y1.retain_grad()
y2.retain_grad()
w.retain_grad()
z.retain_grad()

# 打印 forward 输出
print("z:", z.item())

# backward
z.backward()

# 打印每个节点梯度
print("dz/dX1:", X1.grad.item())
print("dz/dX2:", X2.grad.item())
print("dz/da:", a.grad.item())
print("dz/dy1:", y1.grad.item())
print("dz/dy2:", y2.grad.item())
print("dz/dw:", w.grad.item())
print("dz/dz:", z.grad.item())  # 通常为 1，因为 dz/dz = 1