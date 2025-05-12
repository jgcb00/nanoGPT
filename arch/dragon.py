from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedLinearCrossEntropyLoss, FusedCrossEntropyLoss
try:
    from cut_cross_entropy import linear_cross_entropy
except ImportError:
    linear_cross_entropy = None

from config import NanoConfig
from arch.mlp import MLP
from arch.mixer.mixer_attention import MixerAttention, MixerMetaTokensAttention, MixerDiffAttention, MixerNativeSparseAttention
from arch.mixer.mixer_mamba2 import MixerMamba2
from arch.mixer.mixer_gnd import MixerGatedDeltaNet
from arch.utils import HeadWiseRMSNorm

class Block(nn.Module):
    def __init__(self, config : NanoConfig, swa: bool = False, layer_depth: int = 0, kv_source=None):
        """
        swa: whether to use local attention/SWA for this block, or global
        kv_source: layer to get KV from, if any
        """
        super().__init__()

        if not swa:
            match config.attn_type:
                case "normal":
                    if config.num_meta_tokens == 0:
                        self.attn = MixerAttention(config, swa=swa, kv_share=(kv_source is not None))
                    else:
                        self.attn = MixerMetaTokensAttention(config, swa=swa, kv_share=(kv_source is not None))
                case "diff":
                    self.attn = MixerDiffAttention(config, swa=swa, kv_share=(kv_source is not None), layer_depth=layer_depth)
                case "nsa":
                    self.attn = MixerNativeSparseAttention(config, swa=swa)
                case _:
                    raise ValueError(f"Unknown attention type {config.attn_type}")
        else:
            match config.local_attn_type:
                case "normal":
                    self.attn = MixerAttention(config, swa=swa, kv_share=(kv_source is not None))
                case "diff":
                    self.attn = MixerDiffAttention(config, swa=swa, kv_share=(kv_source is not None), layer_depth=layer_depth)
                case "nsa":
                    self.attn = MixerNativeSparseAttention(config, swa=swa)
                case "metatokens":
                    self.attn = MixerMetaTokensAttention(config, swa=swa, kv_share=(kv_source is not None))
                case _:
                    raise ValueError(f"Unknown attention type {config.local_attn_type}")

        match config.lin_attn_type:
            case "mamba2":
                self.lin_attn = MixerMamba2(config=config)
            case "gdn":
                self.lin_attn = MixerGatedDeltaNet(config=config, expand_factor=self.attn.expand_factor)
            case _:
                raise ValueError(f"Unknown linear attention type {config.lin_attn_type}")
            
        self.expand_factor = self.attn.expand_factor

        self.kv_source = kv_source
        self.out_proj = nn.Linear(int(self.expand_factor*config.d_model), config.d_model, bias=False)
        #self.out_proj.weight.data.zero_() # zero init suggested by @Grad62304977

        #self.attn_norm = torch.nn.Parameter(torch.ones(int(self.expand_factor*config.d_model)))
        #self.is_lin_attn_norm = config.rmsnorm
        #if not config.rmsnorm:
        #    self.lin_attn_norm = torch.nn.Parameter(torch.ones(int(self.expand_factor*config.d_model)))
        self.input_norm = nn.RMSNorm(config.d_model, elementwise_affine=True)
        self.postmixer_norm = nn.RMSNorm(config.d_model, elementwise_affine=True)
        if swa:
            self.attn_norm = HeadWiseRMSNorm(n_heads=self.lin_attn.n_heads, d_head=self.lin_attn.head_v_dim, eps=1e-5)
        else:
            self.attn_norm = HeadWiseRMSNorm(n_heads=self.attn.n_heads//2, d_head=2*self.attn.head_dim, eps=1e-5)
        self.lin_attn_norm = HeadWiseRMSNorm(n_heads=self.lin_attn.n_heads, d_head=self.lin_attn.head_v_dim, eps=1e-5)
        self.mlp = MLP(config)
        
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

        hidden = self.layer_norm_scaling * self.input_norm(x) # (B, L, d_model)
        
        # y_attn and y_lin_attn are (B, L, E*d_model)
        y_attn,     attn_cache     = self.attn(hidden, external_kv=external_kv, cache=attn_cache)
        y_lin_attn, lin_attn_cache = self.lin_attn(hidden, cache=lin_attn_cache)
        y_attn = self.attn_norm(y_attn).view(y_attn.size(0), y_attn.size(1), -1)
        y_lin_attn = self.lin_attn_norm(y_lin_attn).view(y_lin_attn.size(0), y_lin_attn.size(1), -1)
        #y_attn = y_attn.view(y_attn.size(0), y_attn.size(1), -1)
        #y_lin_attn = y_lin_attn.view(y_lin_attn.size(0), y_lin_attn.size(1), -1)
        #y = F.rms_norm(y_attn, (int((hidden.size(-1)*self.expand_factor)),), self.attn_norm)
        #y = y + F.rms_norm(y_lin_attn, (int(hidden.size(-1)*self.expand_factor),), self.lin_attn_norm)
        y = y_attn + y_lin_attn
        x = x + self.out_proj(y / 2)
        x = x + self.mlp(self.layer_norm_scaling * self.postmixer_norm(x))
        return x if cache is None else (x, (attn_cache, lin_attn_cache))

    def get_empty_cache(self):
        # ((k_cache, v_cache, pos), (h_cache, conv_cache))
        return (self.attn.get_empty_cache(), self.lin_attn.get_empty_cache())

class Dragon(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config

        # TODO: fuse the two loops?

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
            """
            block_1 = Block(config, swa=False, layer_depth=1)
            block_2 = Block(config, swa=True, layer_depth=2, kv_source=block_1)
            block_3 = Block(config, swa=True, layer_depth=3)
            block_4 = Block(config, swa=True, layer_depth=4, kv_source=block_3)
            block_5 = Block(config, swa=False, layer_depth=5)
            block_6 = Block(config, swa=True, layer_depth=6)
            block_7 = Block(config, swa=True, layer_depth=7, kv_source=block_6)
            block_8 = Block(config, swa=True, layer_depth=8)
            block_9 = Block(config, swa=True, layer_depth=9, kv_source=block_8)
            blocks = [block_1, block_2, block_3, block_4, block_5, block_6, block_7, block_8, block_9]
            """

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
        else:
            raise NotImplementedError
            
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            h = nn.ModuleList(blocks),
        ))
        self.final_norm = nn.RMSNorm(config.d_model, elementwise_affine=True)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size,  dtype=torch.bfloat16, bias=False)
        #self.lm_head.weight.data.zero_()

        if self.config.num_meta_tokens > 0:
            self.meta_tokens = nn.Parameter(torch.randn(self.config.num_meta_tokens, self.config.d_model))

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.006)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.006)

    def forward(self, x):
        for block in self.transformer.h:
            x = block(x)
        #x = self.final_norm(x)
        return x
