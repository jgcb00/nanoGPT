from typing import Tuple
import torch
import torch.nn as nn

from config import NanoConfig
from arch.utils import get_model

B, L = 2, 2048
d_model, n_heads, n_layers = 256, 4, 20
nconfig = NanoConfig(model="dragon", use_uscaling=True, init_std=1.0, uscaling_tau=0.2, d_model=d_model, n_heads=n_heads, n_layers=n_layers, n_global_layers=3, global_attn_repart="middle", attn_type="diff", lin_attn_type="gdn")

model = get_model(nconfig)
model.to("cuda")

def show_layer_stats(layer: nn.Module, input_shape: Tuple[int, ...]) -> None:
    input = torch.randn(*input_shape, device="cuda", requires_grad=True)
    output = layer(input)
    output.backward(torch.randn_like(output))
    print(f"# {type(layer).__name__}:")
    for k, v in {
        "output": output.std(),
        "input.grad": input.grad.std(),
        **{f"{name}": param.std() for name, param in layer.named_parameters()},
        **{f"{name}.grad": param.grad.std() for name, param in layer.named_parameters()},
    }.items():
        print(f"{k:>20}.std = {v.item():.2f}")

show_layer_stats(model.transformer.h[0], (B, L, d_model))

