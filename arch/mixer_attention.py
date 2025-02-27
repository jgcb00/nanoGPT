import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func

from config import NanoConfig

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
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_head = self.d_model // self.n_head
        assert self.d_model % self.n_head == 0
        self.c_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_v = nn.Linear(self.d_model, self.d_model, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.d_head)

    def forward(self, x):
        # x: (B,T,D) -> y: (B,T,D)
        B, T, _ = x.size() # batch size, sequence length, embedding dimensionality (d_model)
        q = self.c_q(x).view(B, T, self.n_head, self.d_head)
        k = self.c_k(x).view(B, T, self.n_head, self.d_head)
        v = self.c_v(x).view(B, T, self.n_head, self.d_head)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        #y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        #y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = flash_attn_func(q, k, v, causal=True)
        y = y.contiguous().view_as(x)
        y = self.c_proj(y)
        return y
    
    
class MixerDiffAttention(nn.Module):
    def __init__(self, config: NanoConfig, layer_depth: int):
        super().__init__()
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.head_dim = self.d_model // self.n_head
        
        head_dim = self.head_dim / 2
        self.lambda_q1 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        assert self.d_model % self.n_head == 0
        self.c_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_v = nn.Linear(self.d_model, self.d_model, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        # x: (B,T,D) -> y: (B,T,D)
        B, T, _ = x.size() # batch size, sequence length, embedding dimensionality (d_model)
        q = self.c_q(x).view(B, T, 2, self.n_head // 2, self.head_dim)
        k = self.c_k(x).view(B, T, 2, self.n_head // 2 , self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head // 2, 2 * self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        
        q1, q2 = q[:, :, 0], q[:, :, 1]
        k1, k2 = k[:, :, 0], k[:, :, 1]
        
        y1 = F.scaled_dot_product_attention(q1.transpose(1, 2), k1.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y2 = F.scaled_dot_product_attention(q2.transpose(1, 2), k2.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y1 = y1.transpose(1, 2).contiguous()
        y2 = y2.transpose(1, 2).contiguous()
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        y = (y1 - lambda_full * y2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y
