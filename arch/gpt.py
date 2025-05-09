from typing import List, Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedLinearCrossEntropyLoss, FusedCrossEntropyLoss

from config import NanoConfig
from arch.mlp import MLP
from arch.mixer.mixer_attention import Attention, DiffAttention, NativeSparseAttention
    
class Block(nn.Module):
    def __init__(self, config: NanoConfig, swa: bool = False, layer_depth: int = 0, kv_source=None):
        """
        swa: whether to use local attention/SWA for this block, or global
        kv_source: layer to get KV from, if any
        """
        super().__init__()

        match config.attn_type:
            case "normal":
                self.attn = Attention(config, swa=swa, kv_share=(kv_source is not None))
            case "diff":
                self.attn = DiffAttention(config, swa=swa, kv_share=(kv_source is not None), layer_depth=layer_depth)
            case "nsa":
                self.attn = NativeSparseAttention(config)
            case _:
                raise ValueError(f"Unknown attention type {config.attn_type}")
        
        self.kv_source = kv_source
        self.mlp = MLP(config)
        # register here to not break torch_dynamo
        self.register_buffer("layer_norm_scaling", torch.tensor(1 / math.sqrt(layer_depth) if config.layer_norm_scaling else 1.0))

    def forward(self, x, cache=None):
        external_kv = None
        if self.kv_source is not None:
            external_kv = self.kv_source.attn.get_kv()

        h, cache = self.attn(self.layer_norm_scaling * F.rms_norm(x, (x.size(-1),)), external_kv=external_kv, cache=cache)
        x = x + h
        x = x + self.mlp(self.layer_norm_scaling * F.rms_norm(x, (x.size(-1),)))

        return x if cache is None else (x, cache)
    
    def get_empty_cache(self):
        return self.attn.get_empty_cache()

class GPT(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config

        if not self.config.is_scorer:
            swas : List[bool] = [] # whether to use swa for each layer
            for i in range(config.n_layers):
                swa = (i%2 == 1)
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
        else:
            blocks : List[Block] = []
            for i in range(config.n_layers):
                blocks.append(Block(config, swa=config.use_swa, layer_depth=i+1, kv_source=None))
            
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            h = nn.ModuleList(blocks),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        #self.lm_head.weight.data.zero_()

        if self.config.is_scorer:
            self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, caches=None, just_logits=False, scores=None):
        B, L = idx.size()

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (B, L, d_model)

        if self.config.use_patch_level_training:
            x = x.view(B, L//self.config.patch_size, self.config.patch_size, -1).mean(2) # (B, num_patches, D)
            x = x[:, :-1] # remove the last patch
            targets = targets[:, self.config.patch_size-1:-1] # targets is already shifted by one
            # so we remove only the first patch_size-1 tokens, as well as the last

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

        if just_logits:
            logits = self.lm_head(x)
            return logits
        
        if self.config.scoring:
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='none')
            return loss

        if targets is not None: # if we are given some desired targets also calculate the loss
            if self.config.use_patch_level_training:
                if self.config.fused_loss_computation:

                    # regular, modified
                    """
                    logits = self.lm_head(x)
                    logits = logits.float() # use tf32/fp32 for logits

                    targets = targets.reshape(-1, self.config.patch_size)

                    loss = 0
                    for i in range(self.config.patch_size):
                        loss += F.cross_entropy(logits.view(-1, logits.size(-1)), targets[:, i], ignore_index=-1)
                    loss /= self.config.patch_size"
                    """

                    # FusedLinearCrossEntropyLoss
                    
                    criterion = FusedLinearCrossEntropyLoss(ignore_index=-1)
                    targets = targets.reshape(-1, self.config.patch_size)

                    loss = 0
                    for i in range(self.config.patch_size):
                        loss += criterion(x, targets[:, i], self.lm_head.weight)
                    loss /= self.config.patch_size
                    

                    # FusedCrossEntropyLoss
                    """
                    criterion = FusedCrossEntropyLoss(ignore_index=-1)
                    logits = self.lm_head(x)
                    logits = logits.float() # use tf32/fp32 for logits
                    targets = targets.reshape(-1, self.config.patch_size)
                    loss = 0
                    for i in range(self.config.patch_size):
                        loss += criterion(logits.view(-1, logits.size(-1)), targets[:, i])
                    loss /= self.config.patch_size
                    """
                else:
                    logits = self.lm_head(x)
                    logits = logits.float() # use tf32/fp32 for logits

                    targets = targets.reshape(-1, self.config.patch_size)

                    loss = 0
                    log_probs = F.log_softmax(logits.view(-1, logits.size(-1)), dim=1)
                    for i in range(self.config.patch_size):
                        loss += F.nll_loss(log_probs, targets[:, i], ignore_index=-1)
                    loss /= self.config.patch_size
            else:
                if self.config.fused_loss_computation:
                    criterion = FusedLinearCrossEntropyLoss(ignore_index=-1)
                    loss = criterion(x, targets, self.lm_head.weight)
                    
                    #criterion = FusedCrossEntropyLoss(ignore_index=-1)
                    #logits = self.lm_head(x)
                    #logits = logits.float() # use tf32/fp32 for logits
                    #loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                else:
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
