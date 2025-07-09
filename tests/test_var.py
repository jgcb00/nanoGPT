import torch

D = 256

x = torch.randn(D, D)
w = torch.randn(D, D)

y = x @ w.T

print(y.std(), y.mean()) # sqrt(D), 0