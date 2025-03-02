import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NanoConfig
from arch.mlp import MLP
from arch.mixer.mixer_attention import Attention, DiffAttention
    
class Block(nn.Module):
    def __init__(self, config: NanoConfig, swa: bool = False, layer_depth: int = 0, kv_source=None):
        super().__init__()

        match config.attn_type:
            case "normal":
                self.attn = Attention(config, swa=swa, kv_share=(kv_source is not None))
            case "diff":
                self.attn = DiffAttention(config, swa=swa, kv_share=(kv_source is not None), layer_depth=layer_depth)
            case _:
                raise ValueError(f"Unknown attention type {config.attn_type}")
        
        self.kv_source = kv_source # layer to get KV from, if any
        self.mlp = MLP(config)

    def forward(self, x):
        external_kv = None
        if self.kv_source is not None:
            external_kv = self.kv_source.attn.get_kv()
            
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)), external_kv=external_kv)
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

class GPT(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config

        temp_blocks = []
        for i in range(config.n_layers):
            layer_depth = i + 1

            if config.use_swa:
                is_first = layer_depth == 1
                is_middle = layer_depth == (config.n_layers + 1) // 2
                is_last = layer_depth == config.n_layers
                swa = not (is_first or is_middle or is_last)
            else:
                swa = False
                
            temp_blocks.append((Block(config, swa=swa, layer_depth=layer_depth), swa))
        
        blocks = []
        for i in range(config.n_layers):
            block, is_local = temp_blocks[i]

            if not config.use_kv_sharing:
                break
            
            # KV sharing strategy
            if config.use_swa:
                # global/local attn: share kv between consecutive local layers (globals are isolated kv-wise)
                if is_local and i > 0 and temp_blocks[i-1][1]: # prev is local
                    if blocks[i-1].kv_source is None: # prev doesn't have kv source
                        block.kv_source = blocks[i-1]
            else:
                # full global attn: share between every 2 layers
                if i > 0 and i % 2 == 1: # odd layers get KV from previous even layer
                    block.kv_source = blocks[i-1]
                
            blocks.append(block)
            
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
