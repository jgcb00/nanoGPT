import torch.nn as nn
import torch.nn.functional as F

from config import NanoConfig


class MLP(nn.Module):

    def __init__(self, config: NanoConfig):
        super().__init__()
        self.up_proj = nn.Linear(
            config.d_model, config.mlp_expand * config.d_model, bias=False
        )
        self.down_proj = nn.Linear(
            config.mlp_expand * config.d_model, config.d_model, bias=False
        )

    def forward(self, x):
        x = self.up_proj(x)
        x = F.relu(
            x
        ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.down_proj(x)
        return x
