import torch
import torch.nn as nn

from config import NanoConfig
from arch.dragon import Dragon

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

config = NanoConfig()
config.d_model = 768
config.n_layers = 2
config.n_heads = 6
config.attn_type = "normal"
config.lin_attn_type = "mamba2"
config.expand_factor = 2
config.rmsnorm = False
config.d_state = 128
config.d_conv = 4
config.headdim = 64
config.ngroups = 8
config.norm_before_gate = False
config.vocab_size = 1024

model = Dragon(config)
model.to("cuda")

B, L = 8, 512
x = torch.randint(0, 1024, (B, L), device="cuda")

# parallel mode
with ctx:
    y_full = model(x) # (B, L, vocab_size)

print(y_full.shape)

# inference mode
y_step = []
caches = [block.get_empty_cache() for block in model.transformer.h]

with ctx:
    out, caches = model(x[:, 0:10], caches=caches)
y_chunk = out.clone()

for i in range(10, L):
    token = x[:, i:i+1]
    with ctx:
        out, caches = model(token, caches=caches)
    y_step.append(out)

y_step = torch.cat(y_step, dim=1)
y_inference = torch.cat([y_chunk, y_step], dim=1)
print(f"Token-by-token output shape: {y_inference.shape}")

max_diff = (y_full - y_inference).abs().max().item()
print(f"Maximum absolute difference: {max_diff}")

generated_full = y_full.argmax(dim=-1)
generated_inference = y_inference.argmax(dim=-1)
print((generated_full == generated_inference).all())
