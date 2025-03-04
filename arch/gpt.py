from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NanoConfig
from arch.mlp import MLP
from arch.mixer.mixer_attention import Attention, DiffAttention
    
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
    
    def get_empty_cache(self, size):
        return self.attn.get_empty_cache()

class GPT(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config
        
        # TODO: fuse the two loops?

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
            
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            h = nn.ModuleList(blocks),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        #self.lm_head.weight.data.zero_()

    def forward(self, idx, targets=None, caches=None):
        #assert (targets is None) ^ (caches is None), "Either targets or caches must be provided"

        # forward the GPT model itself
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

        x = F.rms_norm(x, (x.size(-1),)) # final norm

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
    
    def generate_naive(self, prompt, n_tokens: int, sample: bool = True, top_k: int = None, temperature: float = 1.0):
        # prompt: (b, T) tensor
        # outputs: (b, t) tensor

        if top_k is not None:
            top_k = min(top_k, self.config.vocab_size)
        
        input_device = prompt.device
        prompt = prompt.to(self.transformer.wte.weight.device)

        self.eval()
        generated = prompt.clone()

        with torch.no_grad():
            for _ in range(n_tokens):
                logits = self.forward(generated, targets=None, caches=None) # (B, L, vocab_size)
                next_token_logits = logits[:, -1]

                if sample:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    
                    if top_k is not None:
                        values, _ = torch.topk(probs, k=top_k) # (B, k) ordered from lowest to biggest
                        probs[probs < values[:, -1, None]] = 0 # zero-out all probs except the k first
                        probs = probs / probs.sum(axis=1, keepdims=True)

                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)

        self.train()

        return generated.to(input_device)[:, -n_tokens:]
    
    def generate(self, prompts, n_tokens: List[int], sample: bool = True, top_k: int = None, temperature: float = 1.0):
        # prompts : list of B x (L) tensors

        B = len(prompts)
        min_len = min(prompt.size(0) for prompt in prompts)
        max_len = max(prompt.size(0) for prompt in prompts)

        max_num_tokens = max(n_tokens)

        max_len_generation = max([len(p) + nt for (p, nt) in zip(prompts, n_tokens)]) # max timestep that wil be reached during generation

        assert min_len >= self.config.d_conv
            
        if top_k is not None:
            top_k = min(top_k, self.config.vocab_size)

        input_device = prompts[0].device
        model_device = self.transformer.wte.weight.device
        
        self.eval()
        
        padded_prompts = [F.pad(prompt, (0, max_len-prompt.size(0))) for prompt in prompts]
        padded_prompts = torch.stack(padded_prompts) # to : model_device?
        
        batched_generated = torch.zeros(B, max_len+max_num_tokens, dtype=torch.long, device=model_device)
        batched_generated[:, :max_len] = padded_prompts
        
        prompt_lengths = torch.tensor([p.size(0) for p in prompts], device=input_device)
        position_ids = torch.arange(max_len+max_num_tokens, device=input_device).unsqueeze(0).expand(B, -1)
        active_mask = position_ids < prompt_lengths.unsqueeze(1)
        active_mask = active_mask.to(model_device)

        # caches is a list of cache, one per layer
        # cache is composed of : - if Mamba(2) layer : the hidden state, and the last d_conv-1 inputs (see more in mamba_lm.py)
        #                        - if attention layer : the KV cache, ie 2 tensors of shape (B, num_kv_heads, L, head_dim)
        caches = [layer.get_empty_cache(len(prompts)) for layer in self.transformer.h]

        with torch.no_grad():
            # process prompt in one go
            logits, caches = self.forward(batched_generated[:, :min_len], targets=None, caches=caches) # (B, L, vocab_size)
            next_token_logits = logits[:, -1] # (B, vocab_size)

            for t in range(min_len, max_len_generation):
                if sample:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)

                    if top_k is not None:
                        values, _ = torch.topk(probs, k=top_k) # (B, k) ordered from lowest to biggest
                        probs[probs < values[:, -1, None]] = 0 # zero-out all probs except the k first
                        probs = probs / probs.sum(axis=1, keepdims=True)

                    next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True) # (B, 1)

                # here, choose if modify batched_generated[:, t] with next_token or leave it as is

                update_mask = ~active_mask[:, t]
                batched_generated[:, t] = torch.where(update_mask, next_token.squeeze(1), batched_generated[:, t])

                next_token_logits, caches = self.forward(batched_generated[:, [t]], targets=None, caches=caches) # (B, 1, vocab_size), caches
                next_token_logits = next_token_logits.squeeze(1) # (B, vocab_size)

        self.train()

        generated = [seq[prompts[i].size(0):prompts[i].size(0) + nt].to(input_device) for i, (seq, nt) in enumerate(zip(batched_generated, n_tokens))]
        return generated
