import torch.nn as nn
import torch.nn.functional as F

from config import NanoConfig

class MLP(nn.Module):

    def __init__(self, config: NanoConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.d_model, 4 * config.d_model, bias=False)
        self.c_proj  = nn.Linear(4 * config.d_model, config.d_model, bias=False)
        #self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x