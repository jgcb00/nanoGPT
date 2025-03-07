from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import NanoConfig
from arch.mlp import MLP
from arch.mixer.mixer_attention import MixerAttention, MixerDiffAttention
from arch.mixer.mixer_mamba2 import MixerMamba2
from arch.mixer.mixer_gnd import MixerGatedDeltaNet

class Block(nn.Module):
    def __init__(self, config : NanoConfig, swa: bool = False, layer_depth: int = 0, kv_source=None):
        """
        swa: whether to use local attention/SWA for this block, or global
        kv_source: layer to get KV from, if any
        """
        super().__init__()

        match config.attn_type:
            case "normal":
                self.attn = MixerAttention(config, swa=swa, kv_share=(kv_source is not None))
            case "diff":
                self.attn = MixerDiffAttention(config, swa=swa, kv_share=(kv_source is not None), layer_depth=layer_depth)
            case _:
                raise ValueError(f"Unknown attention type {config.attn_type}")

        match config.lin_attn_type:
            case "mamba2":
                self.lin_attn = MixerMamba2(config=config)
            case "gdn":
                self.lin_attn = MixerGatedDeltaNet(config=config)
            case _:
                raise ValueError(f"Unknown linear attention type {config.lin_attn_type}")
        
        self.kv_source = kv_source
        self.out_proj = nn.Linear(config.expand_factor*config.d_model, config.d_model, bias=False)
        self.out_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.attn_norm = torch.nn.Parameter(torch.ones(config.expand_factor*config.d_model))
        self.lin_attn_norm = torch.nn.Parameter(torch.ones(config.expand_factor*config.d_model))
        self.mlp = MLP(config)
        self.expand_factor = config.expand_factor
        # register here to not break torch_dynamo
        self.register_buffer("layer_norm_scaling", torch.tensor(1 / math.sqrt(layer_depth) if config.layer_norm_scaling else 1.0))

    def forward(self, x, cache=None):
        external_kv = None
        if self.kv_source is not None:
            external_kv = self.kv_source.attn.get_kv()

        if cache is not None:
            attn_cache, lin_attn_cache = cache
        else:
            attn_cache, lin_attn_cache = None, None

        hidden = self.layer_norm_scaling * F.rms_norm(x, (x.size(-1),))

        y_attn,     attn_cache     = self.attn(hidden, external_kv=external_kv, cache=attn_cache)
        y_lin_attn, lin_attn_cache = self.lin_attn(hidden, cache=lin_attn_cache)
        y = F.rms_norm(y_attn, (hidden.size(-1) * self.expand_factor,), self.attn_norm) + F.rms_norm(y_lin_attn, (hidden.size(-1) * self.expand_factor,), self.lin_attn_norm)
        x = x + self.out_proj(y / 2)
        x = x + self.mlp(self.layer_norm_scaling * F.rms_norm(x, (x.size(-1),)))
        return x if cache is None else (x, (attn_cache, lin_attn_cache))

    def get_empty_cache(self):
        # ((k_cache, v_cache, pos), (h_cache, conv_cache))
        return (self.attn.get_empty_cache(), self.lin_attn.get_empty_cache())

class Dragon(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config

        # TODO: fuse the two loops?
        
        swas : List[bool] = [] # whether to use swa for each layer
        for i in range(config.n_layers):
            layer_depth = i + 1

            if config.use_swa:
                is_first = layer_depth == 1
                is_middle = layer_depth == (config.n_layers + 1) // 2
                is_last = layer_depth == config.n_layers
                swa = not (is_first or is_middle or is_last)
            else:
                swa = False
                
            swas.append(swa)
        
        blocks : List[Block] = []
        for i in range(config.n_layers):
            layer_depth = i + 1
            is_local = swas[i]
            kv_source = None

            if not config.use_kv_sharing:
                blocks.append(Block(config, swa=is_local, layer_depth=layer_depth, kv_source=kv_source))
                continue
            
            # KV sharing strategy
            if config.use_swa:
                # global/local attn: share kv between consecutive local layers (globals are isolated kv-wise)
                if is_local and i > 0 and swas[i-1]: # prev is local
                    if blocks[i-1].kv_source is None: # prev doesn't have kv source
                        kv_source = blocks[i-1]
            else:
                # full global attn: share between every 2 layers
                if i > 0 and i % 2 == 1: # odd layers get KV from previous even layer
                    kv_source = blocks[i-1]
                
            blocks.append(Block(config, swa=is_local, layer_depth=layer_depth, kv_source=kv_source))
            
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            h = nn.ModuleList(blocks),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        #self.lm_head.weight.data.zero_()

    def forward(self, idx, targets=None, caches=None):
        # forward the Dragon model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, d_model)
        x = F.rms_norm(x, (x.size(-1),))

        if caches is None:
            # regular forward pass
            for block in self.transformer.h:
                x = block(x)
        else:
            # forward pass with caching
            for i, block in enumerate(self.transformer.h):
                x, cache = block(x, cache=caches[i] if caches else None)

                if caches is not None:
                    caches[i] = cache

        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None: # training
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return loss
        elif caches is None: # inference without caching (not recommended)
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            return logits
        else: # inference
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            return logits, caches
