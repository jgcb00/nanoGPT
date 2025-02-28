import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NanoConfig
from arch.mlp import MLP
from arch.mixer.mixer_attention import MixerAttention, MixerDiffAttention
from arch.mixer.mixer_mamba2 import MixerMamba2
from arch.mixer.mixer_gnd import MixerGatedDeltaNet

class Block(nn.Module):
    def __init__(self, config : NanoConfig, layer_depth: int = 0):
        super().__init__()
        match config.attn_type:
            case "normal":
                self.attn = MixerAttention(config)
            case "diff":
                self.attn = MixerDiffAttention(config, layer_depth)
            case _:
                raise ValueError(f"Unknown attention type {config.attn_type}")

        match config.lin_attn_type:
            case "mamba2":
                self.lin_attn = MixerMamba2(config=config)
            case "gdn":
                self.lin_attn = MixerGatedDeltaNet(config=config)
            case _:
                raise ValueError(f"Unknown linear attention type {config.lin_attn_type}")
        
        self.out_proj = nn.Linear(config.expand_factor * config.d_model, config.d_model, bias=False)
        self.out_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.attn_norm = torch.nn.Parameter(torch.ones(config.expand_factor * config.d_model))
        self.mamba_norm = torch.nn.Parameter(torch.ones(config.expand_factor * config.d_model))
        self.mlp = MLP(config)
        self.expand_factor = config.expand_factor

    def forward(self, x):
        hidden = F.rms_norm(x, (x.size(-1),))
        y = F.rms_norm(self.attn(hidden), (hidden.size(-1) * self.expand_factor,), self.attn_norm) + F.rms_norm(self.lin_attn(hidden), (hidden.size(-1) * self.expand_factor,), self.mamba_norm)
        x = x + self.out_proj(y / 2)
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

class Dragon(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            h = nn.ModuleList([Block(config, layer_depth=i+1) for i in range(config.n_layers)]),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, idx, targets=None, return_logits=True):
        # forward the Dragon model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, d_model)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss
    