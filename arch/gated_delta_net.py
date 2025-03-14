from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedLinearCrossEntropyLoss

from config import NanoConfig
from arch.mlp import MLP
from arch.mixer.mixer_gnd import GatedDeltaNet

"""not compatible for inference"""
    
class Block(nn.Module):
    def __init__(self, config: NanoConfig, layer_depth: int = 0):
        """
        swa: whether to use local attention/SWA for this block, or global
        kv_source: layer to get KV from, if any
        """
        super().__init__()
        self.mixer = GatedDeltaNet(config)
        self.mlp = MLP(config)
        # register here to not break torch_dynamo
        self.register_buffer("layer_norm_scaling", torch.tensor(1 / math.sqrt(layer_depth) if config.layer_norm_scaling else 1.0))

    def forward(self, x, cache=None):
        h, _ = self.mixer(self.layer_norm_scaling * F.rms_norm(x, (x.size(-1),)), cache=cache)
        x = x + h
        x = x + self.mlp(self.layer_norm_scaling * F.rms_norm(x, (x.size(-1),)))
        return x

class GatedDeltaNetModel(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config
        blocks: List[Block] = []
        for i in range(config.n_layers):
            layer_depth = i + 1
            blocks.append(Block(config, layer_depth=layer_depth))
                
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            h = nn.ModuleList(blocks),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, idx, targets=None, return_logits=True):
        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, d_model)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            if self.config.fused_loss_computation:
                criterion = FusedLinearCrossEntropyLoss(ignore_index=-1)
                loss = criterion(x, targets, self.lm_head.weight)
            else:
                logits = self.lm_head(x)
                logits = logits.float() # use tf32/fp32 for logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None
            return logits