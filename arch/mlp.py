import torch.nn as nn
import torch.nn.functional as F

from arch.utils import ScaledLinear
from config import NanoConfig

class MLP(nn.Module):

    def __init__(self, config: NanoConfig):
        super().__init__()
        self.c_fc    = ScaledLinear(config, config.d_model, config.mlp_expand * config.d_model, bias=False)
        self.c_proj  = ScaledLinear(config, config.mlp_expand * config.d_model, config.d_model, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x