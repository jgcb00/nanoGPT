import torch
import torch.nn as nn

# old, kept for evaluating old runs
"""
class HeadWiseRMSNorm(nn.Module):
    def __init__(self, n_heads, d_head, eps=1e-5):
        super().__init__()
        self.eps = eps
        # poids distinct par tête
        self.weight = nn.Parameter(torch.ones(n_heads, d_head))

    def forward(self, x):
        # x: (B, T, H, D)
        var = x.pow(2).mean(dim=-1, keepdim=True)               # (B, T, H, 1)
        x_norm = x * torch.rsqrt(var + self.eps)               # normalisation RMS
        return x_norm * self.weight.unsqueeze(0).unsqueeze(0)  # (1,1,H,D) → broadcast
"""


class HeadWiseRMSNorm(torch.nn.Module):
    def __init__(self, n_heads, d_head, eps=1e-5):
        super().__init__()
        self.rms = torch.nn.RMSNorm(d_head, eps=eps, elementwise_affine=False)
        self.weight = torch.nn.Parameter(torch.ones(n_heads, d_head))

    def forward(self, x):
        B, L, H, D = x.shape
        y = self.rms(x) * self.weight.view(1, 1, H, D)
        return y.view(B, L, H, D)


def get_model(nconfig):
    match nconfig.model:
        case "gpt":
            from arch.gpt import GPT

            model = GPT(nconfig)
        case "dragon":
            from arch.dragon import Dragon

            model = Dragon(nconfig)
            pass
        case "gated-delta-net":
            from arch.gated_delta_net import GatedDeltaNetModel

            model = GatedDeltaNetModel(nconfig)
        case "mamba2":
            from arch.mamba2 import Mamba2Model

            model = Mamba2Model(nconfig)
        case _:
            raise ValueError(f"Model {nconfig.model} not supported")
    return model
