import torch

from config import NanoConfig
from arch.gpt import GPT

config = NanoConfig()
config.d_model = 768
config.n_heads = 6
config.n_layers = 3
config.vocab_size = 50304

model = GPT(config).to("cuda")

x = torch.randint(0, 50, (16, 1024)).to("cuda")
y = torch.randint(0, 50, (16, 1024)).to("cuda")

print(x.shape)
print(y.shape)

loss = model(x, targets=y)
print(loss.item())
