import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NanoConfig

import transformer_engine as te

class MLP(nn.Module):

    def __init__(self, config: NanoConfig, tp_group: torch.distributed.ProcessGroup, tp_size: int, device: torch.device):
        super().__init__()
        # todo: pass init method
        self.c_fc = te.pytorch.Linear(config.d_model, config.mlp_expand * config.d_model, bias=False, parallel_mode="column", tp_group=tp_group, tp_size=tp_size, device=device)
        self.c_proj = te.pytorch.Linear(config.mlp_expand * config.d_model, config.d_model, bias=False, parallel_mode="row", tp_group=tp_group, tp_size=tp_size, device=device)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x
