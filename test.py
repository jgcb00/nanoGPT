import torch
import torch.nn as nn

from config import NanoConfig
from arch.mixer.mixer_attention import MixerAttention

# Set up configuration
config = NanoConfig()
config.d_model = 128
config.n_heads = 4
config.n_layers = 2
config.n_kv_heads = 2  # For GQA
config.expand_factor = 1
config.qk_norm = True
config.scalable_softmax = True

# Create model and set seed for reproducibility
torch.manual_seed(42)
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.blocks = nn.ModuleList([
            MixerAttention(config, swa=False),
            MixerAttention(config, swa=False),
            MixerAttention(config, swa=False),
            MixerAttention(config, swa=False)
        ])

    def forward(self, x, caches=None):
        for i, block in enumerate(self.blocks):
            x, cache = block(x, cache=caches[i] if caches else None)

            if caches is not None:
                caches[i] = cache
        return x, caches
    
model = MyModel(config).to('cuda')

batch_size = 1
seq_len = 16
x = torch.randn(batch_size, seq_len, config.d_model, device='cuda')

# Process the full sequence at once (training mode)
with ctx:
    y_full, _ = model(x)
print(f"Full sequence output shape: {y_full.shape}")

# Process the sequence token-by-token (inference mode)
y_step = []
caches = [(None, None, 0) for _ in range(len(model.blocks))]

for i in range(seq_len):
    token = x[:, i:i+1, :] # (1, 1, d_model)
    with ctx:
        out, caches = model(token, caches=caches)
    y_step.append(out)

y_step = torch.cat(y_step, dim=1)
print(f"Token-by-token output shape: {y_step.shape}")

max_diff = (y_full - y_step).abs().max().item()
print(f"Maximum absolute difference: {max_diff}")
print(f"Outputs are equivalent: {max_diff < 1e-5}")
