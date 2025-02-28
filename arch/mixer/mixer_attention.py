import torch
import torch.nn as nn
import torch.nn.functional as F
import flash_attn
import flex_head_fa
from config import NanoConfig
import math

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class MixerAttention(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.d_model = config.d_model
        self.d_head = self.d_model // self.n_heads * config.expand_factor
        self.expand_factor = config.expand_factor
        assert self.d_model % self.n_heads == 0
        self.c_q = nn.Linear(self.d_model, self.n_heads*self.d_head, bias=False)
        self.c_k = nn.Linear(self.d_model, self.n_kv_heads*self.d_head, bias=False)
        self.c_v = nn.Linear(self.d_model, self.n_kv_heads*self.d_head, bias=False)
        self.rotary = Rotary(self.d_head)

    def forward(self, x):
        # x: (B,T,D) -> y: (B,T,D)
        B, T, _ = x.size() # batch size, sequence length, embedding dimensionality (d_model)
        q = self.c_q(x).view(B, T, self.n_heads, self.d_head)
        k = self.c_k(x).view(B, T, self.n_kv_heads, self.d_head)
        v = self.c_v(x).view(B, T, self.n_kv_heads, self.d_head)
        
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        k, v = repeat_kv(k, self.n_kv_groups), repeat_kv(v, self.n_kv_groups) # GQA

        y = flash_attn.flash_attn_func(q.bfloat16(), k.bfloat16(), v.bfloat16(), causal=True)
        y = y.contiguous().view(B, T, self.d_model*self.expand_factor)
        return y
    
class Attention(MixerAttention):
    def __init__(self, config):
        super().__init__(config)
        
        # output projection
        self.c_proj = nn.Linear(config.expand_factor * self.d_model, self.d_model, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
    
    def forward(self, x):
        y = super().forward(x)
        y = self.c_proj(y)
        return y
    
class MixerDiffAttention(nn.Module):
    def __init__(self, config: NanoConfig, layer_depth: int = 0):
        super().__init__()
        self.n_heads = config.n_heads
        #self.n_kv_heads = config.n_kv_heads
        #self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.d_model = config.d_model
        self.head_dim = self.d_model // self.n_heads * config.expand_factor
        self.expand_factor = config.expand_factor
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_depth)
        
        head_dim = self.head_dim // 2
        self.lambda_q1 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        assert self.d_model % self.n_heads == 0
        self.c_q = nn.Linear(self.d_model, self.n_heads*self.head_dim, bias=False)
        self.c_k = nn.Linear(self.d_model, self.n_heads*self.head_dim, bias=False)
        self.c_v = nn.Linear(self.d_model, self.n_heads*self.head_dim, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        # x: (B,T,D) -> y: (B,T,D)
        B, T, _ = x.size() # batch size, sequence length, embedding dimensionality (d_model)
        q = self.c_q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_heads//2, 2*self.head_dim)

        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        
        # split q,k heads into two groups
        q, k = q.view(B, T, 2, self.n_heads//2, self.head_dim), k.view(B, T, 2, self.n_heads//2, self.head_dim)        
        q1, q2, k1, k2 = q[:, :, 0], q[:, :, 1], k[:, :, 0], k[:, :, 1]
        
        y1 = flex_head_fa.flash_attn_func(q1.bfloat16(), k1.bfloat16(), v.bfloat16(), causal=True)
        y2 = flex_head_fa.flash_attn_func(q2.bfloat16(), k2.bfloat16(), v.bfloat16(), causal=True)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1*self.lambda_k1, dim=-1).float()).type_as(y1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2*self.lambda_k2, dim=-1).float()).type_as(y2)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        y = (y1 - lambda_full * y2).contiguous().view(B, T, self.d_model*self.expand_factor)
        return y

class DiffAttention(MixerDiffAttention):
    def __init__(self, config, layer_depth: int = 0):
        super().__init__(config, layer_depth=layer_depth)
        
        # output projection
        self.c_proj = nn.Linear(config.expand_factor * self.d_model, self.d_model, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
    
    def forward(self, x):
        y = super().forward(x)
        y = self.c_proj(y)
        return y

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)