from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedLinearCrossEntropyLoss

from config import NanoConfig
from arch.mlp import MLP
from arch.mixer.mixer_attention import (
    MixerAttention,
    MixerDiffAttention,
    MixerGroupedTiedAttention,
    MixerGroupedTiedDifferentialAttention
)
from arch.mixer.mixer_mamba2 import MixerMamba2
from arch.mixer.mixer_gnd import MixerGatedDeltaNet
from arch.utils import HeadWiseRMSNorm

ATTN_CLASSES = {
    "normal": MixerAttention,
    "diff":   MixerDiffAttention,
    "gta":    MixerGroupedTiedAttention,
    "gtda":   MixerGroupedTiedDifferentialAttention,
}

LIN_ATTN_CLASSES = {
    "mamba2": MixerMamba2,
    "gdn":    MixerGatedDeltaNet,
}

class Block(nn.Module):
    def __init__(self, config: NanoConfig, swa: bool = False, layer_depth: int = 1, kv_source=None):
        """
        swa: whether to use local attention/SWA for this block, or global
        kv_source: layer to get KV from, if any
        """
        super().__init__()

        attn_type = config.local_attn_type if swa else config.attn_type
        cls = ATTN_CLASSES.get(attn_type)
        if cls is None:
            raise ValueError(f"Unknown attention type {attn_type}")
        self.attn = cls(config, swa=swa, kv_share=(kv_source is not None), layer_depth=layer_depth)

        cls = LIN_ATTN_CLASSES.get(config.lin_attn_type)
        if cls is None:
            raise ValueError(f"Unknown linear attention type {config.lin_attn_type}")
        self.lin_attn = cls(config)
            
        self.expand_factor = config.expand_factor

        self.kv_source = kv_source
        self.out_proj = nn.Linear(int(self.expand_factor*config.d_model), config.d_model, bias=False)
        
        if isinstance(self.attn, (MixerDiffAttention, MixerGroupedTiedDifferentialAttention)):
            self.attn_group_norm = HeadWiseRMSNorm(n_heads=self.attn.n_heads//2, d_head=2*self.attn.head_dim, eps=config.eps_rmsnorm)
        else:
            self.attn_group_norm = HeadWiseRMSNorm(n_heads=self.attn.n_heads, d_head=self.attn.d_head, eps=config.eps_rmsnorm)
        self.lin_attn_group_norm = HeadWiseRMSNorm(n_heads=self.lin_attn.n_heads, d_head=self.lin_attn.head_v_dim, eps=config.eps_rmsnorm)

        self.input_norm = nn.RMSNorm(config.d_model, elementwise_affine=config.rmsnorm_weights, eps=config.eps_rmsnorm)
        self.postmixer_norm = nn.RMSNorm(config.d_model, elementwise_affine=config.rmsnorm_weights, eps=config.eps_rmsnorm)
        self.mlp = MLP(config)
        
        self.register_buffer("layer_norm_scaling", torch.tensor(1 / math.sqrt(layer_depth+1)) if config.layer_norm_scaling else 1.)

    def forward(self, x, cache=None):
        external_kv = None
        if self.kv_source is not None:
            external_kv = self.kv_source.attn.get_kv()

        if cache is not None:
            attn_cache, lin_attn_cache = cache
        else:
            attn_cache, lin_attn_cache = None, None

        hidden = self.layer_norm_scaling * self.input_norm(x) # (B, L, d_model)
        
        # MIXER.
        y_attn,     attn_cache     = self.attn(hidden, external_kv=external_kv, cache=attn_cache) # (B, L, E*D)
        y_lin_attn, lin_attn_cache = self.lin_attn(hidden, cache=lin_attn_cache) # (B, L, E*D)

        y_attn = self.attn_group_norm(y_attn).view(y_attn.size(0), y_attn.size(1), -1)
        y_lin_attn = self.lin_attn_group_norm(y_lin_attn).view(y_lin_attn.size(0), y_lin_attn.size(1), -1)

        x = x + self.out_proj((y_attn + y_lin_attn) / 2)

        # MLP.
        x = x + self.mlp(self.layer_norm_scaling * self.postmixer_norm(x))
        return x if cache is None else (x, (attn_cache, lin_attn_cache))

    def get_empty_cache(self):
        # ((k_cache, v_cache, pos), (h_cache, conv_cache))
        return (self.attn.get_empty_cache(), self.lin_attn.get_empty_cache())

class Dragon(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config

        if self.config.global_attn_repart == "hymba":
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
        elif self.config.global_attn_repart == "middle":
            n = config.n_layers
            base, rem = divmod(n, 3)
            group_sizes = [base + (1 if i < rem else 0) for i in range(3)]
            starts = [sum(group_sizes[:i]) for i in range(3)]
            mids = {starts[i] + group_sizes[i] // 2 for i in range(3)}
            swas = [i not in mids for i in range(n)]
            blocks: List[Block] = []
            for i in range(n):
                layer_depth = i + 1
                is_local = swas[i]
                kv_source = None
                if not config.use_kv_sharing:
                    blocks.append(Block(config, swa=is_local, layer_depth=layer_depth, kv_source=kv_source))
                    continue
                if is_local and i > 0 and swas[i-1]: # share kv cache between consecutive local layers
                    prev = blocks[i-1]
                    if prev.kv_source is None:
                        kv_source = prev
                blocks.append(Block(config, swa=is_local, layer_depth=layer_depth, kv_source=kv_source))
        elif self.config.global_attn_repart == "megatron":
            n = config.n_layers
            G = config.n_global_layers
            swas = self.allocate_swas(n, G)

            blocks: List[Block] = []
            layer_depth = 1

            for is_local in swas:
                if is_local:
                    # first local
                    first = Block(config, swa=True, layer_depth=layer_depth, kv_source=None)
                    blocks.append(first)
                    layer_depth += 1

                    # second local, sharing kv with the first
                    kv_src = first if config.use_kv_sharing else None
                    second = Block(config, swa=True, layer_depth=layer_depth, kv_source=kv_src)
                    blocks.append(second)
                    layer_depth += 1
                else:
                    # one global‐attention layer
                    blk = Block(config, swa=False, layer_depth=layer_depth, kv_source=None)
                    blocks.append(blk)
                    layer_depth += 1
        else:
            raise NotImplementedError
            
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            h = nn.ModuleList(blocks),
        ))
        if self.config.input_norm:
            self.input_norm = nn.RMSNorm(config.d_model, elementwise_affine=config.rmsnorm_weights, eps=config.eps_rmsnorm)
        self.final_norm = nn.RMSNorm(config.d_model, elementwise_affine=config.rmsnorm_weights, eps=config.eps_rmsnorm)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def allocate_swas(self, total_layers_count: int, num_global_attention_layers: int):
        base, rem = divmod(total_layers_count - num_global_attention_layers, num_global_attention_layers)
        assert base % 4 == 0, "Locals must be divisible by 4 for equal kv-sharing"
        assert rem == 0, "Improper number of layers"
        swas = []
        for _ in range(num_global_attention_layers):
            swas += [True] * (base // 4)   # local run
            swas += [False]                # global
            swas += [True] * (base // 4)   # local run
        return swas

    def _init_weights(self, module: nn.Module):

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.006)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.006)

    def forward(self, idx, targets=None, scores=None, caches=None, just_logits=False):
        B, L = idx.size()

        # forward the Dragon model itself
        x = self.transformer.wte(idx) # token embeddings of shape (B, L, d_model)

        if self.config.input_norm:
            x = self.input_norm(x)

        if self.config.use_patch_level_training:
            x = x.view(B, L//self.config.patch_size, self.config.patch_size, -1).mean(2) # (B, num_patches, D)
            x = x[:, :-1] # remove the last patch
            targets = targets[:, self.config.patch_size-1:-1] # targets is already shifted by one
            # so we remove only the first patch_size-1 tokens, as well as the last

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

        x = self.final_norm(x)

        if just_logits:
            logits = self.lm_head(x)
            return logits            

        if targets is not None: # if we are given some desired targets also calculate the loss
            if self.config.use_patch_level_training:
                if self.config.fused_loss_computation:
                    # FusedLinearCrossEntropyLoss
                    criterion = FusedLinearCrossEntropyLoss(ignore_index=-1)
                    targets = targets.reshape(-1, self.config.patch_size)

                    loss = 0
                    for i in range(self.config.patch_size):
                        loss += criterion(x, targets[:, i], self.lm_head.weight)
                    loss /= self.config.patch_size
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
