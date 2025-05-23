import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

import transformer_engine as te

try:
    from flash_attn import flash_attn_func # FA2
    FLASH_ATTN_TYPE = "FA2"
except ImportError:
    try:
        import flash_attn_interface # FA3
        flash_attn_func = flash_attn_interface.flash_attn_func
        FLASH_ATTN_TYPE = "FA3"
    except ImportError:
        flash_attn_func = None
        FLASH_ATTN_TYPE = None

try:
    import flex_head_fa
except ImportError:
    pass

try:
    from native_sparse_attention.ops import compressed_attention, topk_sparse_attention, linear_compress, avgpool_compress, weightedpool_compress
    from native_sparse_attention.module.rope import RopeConfig, RotaryEmbedding
except ImportError:
    pass

from config import NanoConfig
from arch.utils import HeadWiseRMSNorm
import math

#todo: rename head_dim in diffattn to d_head just like in mixer_attention
#todo: convert back to float32 after attn?

#todo: is the pos in the cache really useful??

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, start_pos=0):
        seq_len = x.shape[1]
        total_len = start_pos + seq_len
        
        if total_len > self.seq_len_cached:
            self.seq_len_cached = max(2*total_len, 16) # each time we encounter a new seq_len, we cache the next 2*seq_len
            t = torch.arange(self.seq_len_cached, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
            
        cos = self.cos_cached[start_pos:start_pos+seq_len]
        sin = self.sin_cached[start_pos:start_pos+seq_len]
        return cos[None, :, None, :], sin[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class MixerAttention(nn.Module):
    def __init__(self, config: NanoConfig, tp_group: torch.distributed.ProcessGroup, tp_size: int, device: torch.device, swa: bool = False, kv_share: bool = False):
        super().__init__()

        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_repeats = self.n_heads // self.n_kv_heads
        self.d_model = config.d_model
        self.expand_factor = config.expand_factor
        self.d_head = (self.d_model*self.expand_factor) // self.n_heads # todo: change this
        self.swa, self.swa_window_size = swa, config.swa_window_size
        self.rope = self.swa or not config.rope_to_nope
        self.qk_norm = config.qk_norm
        assert self.d_model % self.n_heads == 0

        self.n_heads_local = self.n_heads // tp_size
        self.n_kv_heads_local = self.n_kv_heads // tp_size

        self.c_q = te.pytorch.Linear(config.d_model, self.n_heads*self.d_head, bias=False, parallel_mode="column", tp_group=tp_group, tp_size=tp_size, device=device)
        if not kv_share: # only define kv projs if not sharing
            self.c_k = te.pytorch.Linear(config.d_model, self.n_kv_heads*self.d_head, bias=False, parallel_mode="column", tp_group=tp_group, tp_size=tp_size, device=device)
            self.c_v = te.pytorch.Linear(config.d_model, self.n_kv_heads*self.d_head, bias=False, parallel_mode="column", tp_group=tp_group, tp_size=tp_size, device=device)
        if self.rope:
            if self.swa:
                self.rotary = Rotary(self.d_head, base=self.config.rope_theta_local) # 477=3k/(2pi)
            else:
                self.rotary = Rotary(self.d_head, base=self.config.rope_theta_global)

        if self.qk_norm:
            self.q_norm = nn.RMSNorm(self.d_head, elementwise_affine=config.rmsnorm_weights, eps=config.eps_rmsnorm)
            if not kv_share:
                self.k_norm = nn.RMSNorm(self.d_head, elementwise_affine=config.rmsnorm_weights, eps=config.eps_rmsnorm)

        if self.config.groupnorm:
            self.group_norm = HeadWiseRMSNorm(n_heads=self.n_heads_local, d_head=self.d_head, eps=config.eps_rmsnorm)

        self.last_k = None
        self.last_v = None

    def forward(self, hidden_states, external_kv=None, cache=None):
        # hidden_states: (B,T,D) -> y: (B,T,D)
        # external_kv: used in training, to use the kv from the previous layer
        # cache: used at inference, to store the kv from the past (k, v, pos)

        x = hidden_states

        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_heads_local, self.d_head)

        start_pos = cache[2] if cache is not None else 0
        
        if external_kv is not None: # kv-sharing path
            k, v = external_kv

            if self.rope:
                cos, sin = self.rotary(q, start_pos)
            q = self.q_norm(q) if self.qk_norm else q
            if self.rope:
                q = apply_rotary_emb(q, cos, sin) # RoPE
        else: # regular path
            k = self.c_k(x).view(B, T, self.n_kv_heads_local, self.d_head)
            v = self.c_v(x).view(B, T, self.n_kv_heads_local, self.d_head)
            
            if self.rope:
                cos, sin = self.rotary(q, start_pos)
            q, k = (self.q_norm(q), self.k_norm(k)) if self.qk_norm else (q, k) # QK norm suggested by @Grad62304977
            if self.rope:
                q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # RoPE
            
            self.last_k, self.last_v = k, v
        
        new_pos = start_pos + T
        if cache is not None:
            past_k, past_v, _ = cache
            if past_k is not None: # not first token
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)
            
            cache = (k, v, new_pos)
        
        k, v = repeat_kv(k, self.n_kv_repeats), repeat_kv(v, self.n_kv_repeats) # GQA (todo: can be handled by FA)

        wsize = self.swa_window_size if self.swa else -1
        if self.config.slw_window > 0:
            if self.swa:
                wsize = min(self.config.slw_window, self.swa_window_size)
            else:
                if self.config.slw_window < self.config.sequence_length:
                    wsize = self.config.slw_window
                else:
                    wsize = -1
        
        if FLASH_ATTN_TYPE == "FA2":
            y = flash_attn_func(q.bfloat16(), k.bfloat16(), v.bfloat16(), causal=True, window_size=(wsize, wsize))
        elif FLASH_ATTN_TYPE == "FA3":
            y, _ = flash_attn_func(q.bfloat16(), k.bfloat16(), v.bfloat16(), causal=True, window_size=(wsize, wsize))
        else:
            raise ValueError

        y = self.group_norm(y)
        return y, cache
    
    def get_kv(self):
        """ used for cross-layer kv sharing """
        return self.last_k, self.last_v
    
    def get_empty_cache(self):
        return (None, None, 0) # (k_cache, v_cache, pos)
    
class MixerDiffAttention(nn.Module):
    def __init__(self, config: NanoConfig, tp_group: torch.distributed.ProcessGroup, tp_size: int, device: torch.device, swa: bool = False, kv_share: bool = False, layer_depth: int = 0):
        super().__init__()

        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_repeats = self.n_heads // self.n_kv_heads
        self.d_model = config.d_model
        self.expand_factor = config.expand_factor
        self.head_dim = (self.d_model * self.expand_factor) // self.n_heads # todo: change this
        self.swa, self.swa_window_size = swa, config.swa_window_size
        self.rope = self.swa or not config.rope_to_nope
        self.qk_norm = config.qk_norm
        self.scalable_softmax = config.scalable_softmax
        if config.disable_scalable_softmax_for_local and self.swa:
            self.scalable_softmax = False
        self.register_buffer("lambda_init", torch.tensor(0.8 - 0.6 * math.exp(-0.3 * layer_depth)))

        self.n_heads_local = self.n_heads // tp_size
        self.n_kv_heads_local = self.n_kv_heads // tp_size
        
        head_dim = self.head_dim // 2
        self.lambda_q1 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        if self.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=config.rmsnorm_weights, eps=config.eps_rmsnorm)
            if not kv_share:
                self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=config.rmsnorm_weights, eps=config.eps_rmsnorm)

        assert self.d_model % self.n_heads == 0
        self.c_q = te.pytorch.Linear(config.d_model, self.n_heads*self.head_dim, bias=False, parallel_mode="column", tp_group=tp_group, tp_size=tp_size, device=device)
        if not kv_share: # only define kv projs if not sharing
            self.c_k = te.pytorch.Linear(config.d_model, self.n_kv_heads*self.head_dim, bias=False, parallel_mode="column", tp_group=tp_group, tp_size=tp_size, device=device)
            self.c_v = te.pytorch.Linear(config.d_model, self.n_kv_heads*self.head_dim, bias=False, parallel_mode="column", tp_group=tp_group, tp_size=tp_size, device=device)
        
        if self.rope:
            if self.swa:
                self.rotary = Rotary(self.head_dim, base=self.config.rope_theta_local)
            else:
                self.rotary = Rotary(self.head_dim, base=self.config.rope_theta_global)
        
        if self.scalable_softmax:
            self.softmax_scaler = nn.Parameter(torch.ones(self.n_heads_local, dtype=torch.float32))

        if self.config.groupnorm:
            self.group_norm = HeadWiseRMSNorm(n_heads=self.n_heads_local//2, d_head=2*self.head_dim, eps=config.eps_rmsnorm)

        self.last_k1 = None
        self.last_k2 = None
        self.last_v = None

    def forward(self, hidden_states, external_kv=None, cache=None):
        # hidden_states: (B,T,D) -> y: (B,T,D)
        # external_kv: used in training, to use the kv from the previous layer
        # cache: used at inference, to store the kv from the past (k1, k2, v, pos)

        x = hidden_states

        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_heads_local, self.head_dim)
        
        start_pos = cache[3] if cache is not None else 0
        
        if external_kv is not None: # kv-sharing path
            k1, k2, v = external_kv

            if self.rope:
                cos, sin = self.rotary(q, start_pos)
            q = self.q_norm(q) if self.qk_norm else q
            if self.rope:
                q = apply_rotary_emb(q, cos, sin) # RoPE
        else: # regular path
            k = self.c_k(x).view(B, T, self.n_kv_heads_local, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_heads_local//2, 2*self.head_dim)

            if self.rope:
                cos, sin = self.rotary(q, start_pos)
            q, k = (self.q_norm(q), self.k_norm(k)) if self.qk_norm else (q, k) # QK norm suggested by @Grad62304977
            if self.rope:
                q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # RoPE
            
            # split k heads into two groups
            k = k.view(B, T, 2, self.n_kv_heads_local//2, self.head_dim)
            k1, k2 = k[:, :, 0], k[:, :, 1]
            
            self.last_k1, self.last_k2, self.last_v = k1, k2, v

        wsize = self.swa_window_size if self.swa else -1
        if self.config.slw_window > 0:
            if self.swa:
                wsize = min(self.config.slw_window, self.swa_window_size)
            else:
                if self.config.slw_window < self.config.sequence_length:
                    wsize = self.config.slw_window
                else:
                    wsize = -1

        # TODO: optimize this (cache it)
        if self.scalable_softmax:
            # scalable-softmax (https://arxiv.org/abs/2501.19399): multiply q by s*log(n)
            pos = torch.arange(start_pos+1, start_pos+T+1, device=q.device).view(1, T, 1, 1)
            log_pos = pos.float().log() if wsize <= 0 else torch.clamp_max(pos.float(), wsize).log()
            q = (self.softmax_scaler.view(1, 1, -1, 1) * log_pos) * q
            
        # split q heads into two groups
        q = q.view(B, T, 2, self.n_heads_local//2, self.head_dim)
        q1, q2 = q[:, :, 0], q[:, :, 1]
        
        new_pos = start_pos + T
        if cache is not None:
            past_k1, past_k2, past_v, _ = cache
            if past_k1 is not None: # not first token
                k1 = torch.cat([past_k1, k1], dim=1)
                k2 = torch.cat([past_k2, k2], dim=1)
                v = torch.cat([past_v, v], dim=1)
            
            cache = (k1, k2, v, new_pos)
        
        k1, k2, v = repeat_kv(k1, self.n_kv_repeats), repeat_kv(k2, self.n_kv_repeats), repeat_kv(v, self.n_kv_repeats) # GQA todo: can be handled by FA

        y1 = flex_head_fa.flash_attn_func(q1.bfloat16(), k1.bfloat16(), v.bfloat16(), causal=True, window_size=(wsize, wsize))
        y2 = flex_head_fa.flash_attn_func(q2.bfloat16(), k2.bfloat16(), v.bfloat16(), causal=True, window_size=(wsize, wsize))
        lambda_1 = torch.exp(torch.sum(self.lambda_q1*self.lambda_k1, dim=-1).float()).type_as(y1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2*self.lambda_k2, dim=-1).float()).type_as(y2)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        y = (y1 - lambda_full * y2).contiguous()

        y = self.group_norm(y)
        return y, cache
        
    def get_kv(self):
        return self.last_k1, self.last_k2, self.last_v
    
    def get_empty_cache(self):
        return (None, None, None, 0) # (k1_cache, k2_cache, v_cache, pos)

# classic helper function for GQA, although the shapes are different
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    #From (batch, seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    #where num_attention_heads = num_key_value_heads * n_rep.
    """
    batch, seqlen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, seqlen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, seqlen, num_key_value_heads * n_rep, head_dim)
