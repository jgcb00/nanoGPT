import torch
import torch.nn as nn

from config import NanoConfig
from arch.mixer.mixer_mamba2 import MixerMamba2, Mamba2

# Set up configuration
config = NanoConfig()
config.d_model = 768
config.expand_factor = 2
config.rmsnorm = False
config.d_state = 128
config.d_conv = 4
config.headdim = 64
config.ngroups = 8
config.norm_before_gate = False

# Create model and set seed for reproducibility
torch.manual_seed(42)
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.blocks = nn.ModuleList(
            [
                MixerMamba2(config),
            ]
        )

    def forward(self, x, caches=None):
        for i, block in enumerate(self.blocks):
            x, cache = block(x, cache=caches[i] if caches else None)

            if caches is not None:
                caches[i] = cache
        return x, caches


model = MyModel(config).to("cuda")

batch_size = 1
seq_len = 1024
x = torch.randn(batch_size, seq_len, config.d_model, device="cuda")

# Process the full sequence at once (training mode)
with ctx:
    y_full, _ = model(x)
print(f"Full sequence output shape: {y_full.shape}")

# Process the sequence token-by-token (inference mode)
y_step = []
caches = [block.get_empty_cache() for block in model.blocks]

with ctx:
    out, caches = model(x[:, 0:3, :], caches=caches)
y_chunk = out.clone()

for i in range(3, seq_len):
    token = x[:, i : i + 1, :]  # (1, 1, d_model)
    with ctx:
        out, caches = model(token, caches=caches)
    y_step.append(out)

y_step = torch.cat(y_step, dim=1)
y_inference = torch.cat([y_chunk, y_step], dim=1)
print(f"Token-by-token output shape: {y_inference.shape}")

max_diff = (y_full - y_inference).abs().max().item()
print(f"Maximum absolute difference: {max_diff}")
print(f"Outputs are equivalent: {max_diff < 1e-5}")
