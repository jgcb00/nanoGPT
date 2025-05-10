import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

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
    def __init__(self, config: NanoConfig, swa: bool = False, kv_share: bool = False):
        super().__init__()

        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_repeats = self.n_heads // self.n_kv_heads
        self.d_model = config.d_model
        #self.expand_factor = config.expand_factor//2 if swa else config.expand_factor
        #assert self.expand_factor==2, "should be 2 here"
        self.expand_factor = config.expand_factor
        self.d_head = (self.d_model*self.expand_factor) // self.n_heads
        self.swa, self.swa_window_size = swa, config.swa_window_size
        self.rope = self.swa or not config.rope_to_nope
        self.qk_norm = config.qk_norm
        self.scalable_softmax = config.scalable_softmax
        if config.disable_scalable_softmax_for_local and self.swa:
            self.scalable_softmax = False
        assert self.d_model % self.n_heads == 0

        self.c_q = nn.Linear(self.d_model, self.n_heads*self.d_head, bias=False)
        if not kv_share: # only define kv projs if not sharing
            self.c_k = nn.Linear(self.d_model, self.n_kv_heads*self.d_head, bias=False)
            self.c_v = nn.Linear(self.d_model, self.n_kv_heads*self.d_head, bias=False)
        if self.rope:
            if self.swa:
                self.rotary = Rotary(self.d_head, base=self.config.rope_theta_local) # 477=3k/(2pi)
            else:
                self.rotary = Rotary(self.d_head, base=self.config.rope_theta_global)
        if self.scalable_softmax:
            self.softmax_scaler = nn.Parameter(torch.ones(self.n_heads))
        self.last_k = None
        self.last_v = None

    def forward(self, x, external_kv=None, cache=None):
        # x: (B,T,D) -> y: (B,T,D)
        # external_kv: used in training, to use the kv from the previous layer
        # cache: used at inference, to store the kv from the past (k, v, pos)

        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_heads, self.d_head)

        start_pos = cache[2] if cache is not None else 0
        
        if external_kv is not None: # kv-sharing path
            k, v = external_kv

            if self.rope:
                cos, sin = self.rotary(q, start_pos)
            q = F.rms_norm(q, (q.size(-1),)) if self.qk_norm else q # QK norm (only for q)
            if self.rope:
                q = apply_rotary_emb(q, cos, sin) # RoPE
        else: # regular path
            k = self.c_k(x).view(B, T, self.n_kv_heads, self.d_head)
            v = self.c_v(x).view(B, T, self.n_kv_heads, self.d_head)
            
            if self.rope:
                cos, sin = self.rotary(q, start_pos)
            q, k = (F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))) if self.qk_norm else (q, k) # QK norm suggested by @Grad62304977
            if self.rope:
                q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # RoPE
            
            self.last_k, self.last_v = k, v

        #return q,k,v
        
        if self.scalable_softmax:
            # scalable-softmax (https://arxiv.org/abs/2501.19399): multiply q by s*log(n)
            log_pos = torch.arange(start_pos+1, start_pos+T+1, device=q.device).view(1, T, 1, 1).float().log()
            q = (self.softmax_scaler.view(1, 1, -1, 1) * log_pos) * q

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
        #y = y.contiguous().view(B, T, self.d_model*self.expand_factor)
        return y, cache
    
    def get_kv(self):
        """ used for cross-layer kv sharing """
        return self.last_k, self.last_v
    
    def get_empty_cache(self):
        return (None, None, 0) # (k_cache, v_cache, pos)
    
class MixerMetaTokensAttention(nn.Module):
    def __init__(self, config: NanoConfig, swa: bool = False, kv_share: bool = False):
        super().__init__()

        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_repeats = self.n_heads // self.n_kv_heads
        self.d_model = config.d_model
        #self.expand_factor = config.expand_factor//2 if swa else config.expand_factor
        #assert self.expand_factor==2, "should be 2 here"
        self.expand_factor = config.expand_factor
        self.d_head = (self.d_model*self.expand_factor) // self.n_heads
        self.swa, self.swa_window_size = swa, config.swa_window_size
        self.rope = self.swa or not config.rope_to_nope
        self.qk_norm = config.qk_norm
        self.scalable_softmax = config.scalable_softmax
        if config.disable_scalable_softmax_for_local and self.swa:
            self.scalable_softmax = False
        assert self.d_model % self.n_heads == 0

        self.c_q = nn.Linear(self.d_model, self.n_heads*self.d_head, bias=False)
        if not kv_share: # only define kv projs if not sharing
            self.c_k = nn.Linear(self.d_model, self.n_kv_heads*self.d_head, bias=False)
            self.c_v = nn.Linear(self.d_model, self.n_kv_heads*self.d_head, bias=False)
        if self.rope:
            if self.swa:
                self.rotary = Rotary(self.d_head, base=self.config.rope_theta_local) # 477=3k/(2pi)
            else:
                self.rotary = Rotary(self.d_head, base=self.config.rope_theta_global)
        
        if self.scalable_softmax:
            self.softmax_scaler = nn.Parameter(torch.ones(self.n_heads))

        wsize = config.swa_window_size
        def attn_mask(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            swa = q_idx - kv_idx <= wsize
            prefix = kv_idx < config.num_meta_tokens
            return causal & (swa | prefix)
        block_mask_local = create_block_mask(attn_mask, B=None, H=None, Q_LEN=config.sequence_length+config.num_meta_tokens, KV_LEN=config.sequence_length+config.num_meta_tokens, _compile=True)
        self.block_mask = block_mask_local

        self.last_k = None
        self.last_v = None

    def forward(self, x, external_kv=None, cache=None):
        # x: (B,T,D) -> y: (B,T,D)
        # external_kv: used in training, to use the kv from the previous layer
        # cache: used at inference, to store the kv from the past (k, v, pos)

        B, T, _ = x.size()

        q = self.c_q(x).view(B, T, self.n_heads, self.d_head)

        start_pos = cache[2] if cache is not None else 0
        
        if external_kv is not None: # kv-sharing path
            k, v = external_kv

            if self.rope:
                cos, sin = self.rotary(q, start_pos)
            q = F.rms_norm(q, (q.size(-1),)) if self.qk_norm else q # QK norm (only for q)
            if self.rope:
                q = apply_rotary_emb(q, cos, sin) # RoPE
        else: # regular path
            k = self.c_k(x).view(B, T, self.n_kv_heads, self.d_head)
            v = self.c_v(x).view(B, T, self.n_kv_heads, self.d_head)
            
            if self.rope:
                cos, sin = self.rotary(q, start_pos)
            q, k = (F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))) if self.qk_norm else (q, k) # QK norm suggested by @Grad62304977
            if self.rope:
                q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # RoPE
            
            self.last_k, self.last_v = k, v
        
        if self.scalable_softmax:
            # scalable-softmax (https://arxiv.org/abs/2501.19399): multiply q by s*log(n)
            log_pos = torch.arange(start_pos+1, start_pos+T+1, device=q.device).view(1, T, 1, 1).float().log()
            q = (self.softmax_scaler.view(1, 1, -1, 1) * log_pos) * q

        new_pos = start_pos + T
        if cache is not None:
            past_k, past_v, _ = cache
            if past_k is not None: # not first token
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)
            
            cache = (k, v, new_pos)
        
        k, v = repeat_kv(k, self.n_kv_repeats), repeat_kv(v, self.n_kv_repeats) # GQA (todo: can be handled by FA)

        #todo:make compatible with skyladder
        """
        wsize = self.swa_window_size if self.swa else -1
        if self.config.slw_window > 0:
            if self.swa:
                wsize = min(self.config.slw_window, self.swa_window_size)
            else:
                if self.config.slw_window < self.config.sequence_length:
                    wsize = self.config.slw_window
                else:
                    wsize = -1
        """

        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=self.block_mask).transpose(1, 2)
        y = y.contiguous().view(B, T, self.d_model*self.expand_factor)
        return y, cache
    
    def get_kv(self):
        """ used for cross-layer kv sharing """
        return self.last_k, self.last_v
    
    def get_empty_cache(self):
        return (None, None, 0) # (k_cache, v_cache, pos)

class Attention(MixerAttention):
    def __init__(self, config: NanoConfig, swa: bool = False, kv_share: bool = False):
        super().__init__(config, swa=swa, kv_share=kv_share)
        
        # output projection
        self.c_proj = nn.Linear(self.expand_factor * self.d_model, self.d_model, bias=False)
        #self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
    
    def forward(self, x, external_kv=None, cache=None):
        y, cache = super().forward(x, external_kv, cache)
        y = self.c_proj(y)
        return y, cache
    
class MixerDiffAttention(nn.Module):
    def __init__(self, config: NanoConfig, swa: bool = False, kv_share: bool = False, layer_depth: int = 0):
        super().__init__()

        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_repeats = self.n_heads // self.n_kv_heads
        self.d_model = config.d_model
        #self.expand_factor = config.expand_factor//2 if swa else config.expand_factor
        #assert self.expand_factor==2, "should be 2 here"
        self.expand_factor = config.expand_factor
        self.head_dim = (self.d_model * self.expand_factor) // self.n_heads
        self.swa, self.swa_window_size = swa, config.swa_window_size
        self.rope = self.swa or not config.rope_to_nope
        self.qk_norm = config.qk_norm
        self.scalable_softmax = config.scalable_softmax
        if config.disable_scalable_softmax_for_local and self.swa:
            self.scalable_softmax = False
        self.register_buffer("lambda_init", torch.tensor(0.8 - 0.6 * math.exp(-0.3 * layer_depth)))
        
        head_dim = self.head_dim // 2
        self.lambda_q1 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = torch.nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        assert self.d_model % self.n_heads == 0
        self.c_q = nn.Linear(self.d_model, self.n_heads*self.head_dim, bias=False)
        if not kv_share: # only define kv projs if not sharing
            self.c_k = nn.Linear(self.d_model, self.n_kv_heads*self.head_dim, bias=False)
            self.c_v = nn.Linear(self.d_model, self.n_kv_heads*self.head_dim, bias=False)
        if self.rope:
            if self.swa:
                self.rotary = Rotary(self.head_dim, base=self.config.rope_theta_local) # 477=3k/(2pi)
            else:
                self.rotary = Rotary(self.head_dim, base=self.config.rope_theta_global)
        if self.scalable_softmax:
            self.softmax_scaler = nn.Parameter(torch.ones(self.n_heads))
        self.last_k1 = None
        self.last_k2 = None
        self.last_v = None

    def forward(self, x, external_kv=None, cache=None):
        # x: (B,T,D) -> y: (B,T,D)
        # external_kv: used in training, to use the kv from the previous layer
        # cache: used at inference, to store the kv from the past (k1, k2, v, pos)
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_heads, self.head_dim)
        
        start_pos = cache[3] if cache is not None else 0
        
        if external_kv is not None: # kv-sharing path
            k1, k2, v = external_kv

            if self.rope:
                cos, sin = self.rotary(q, start_pos)
            q = F.rms_norm(q, (q.size(-1),)) if self.qk_norm else q # QK norm (only for q)
            if self.rope:
                q = apply_rotary_emb(q, cos, sin) # RoPE
        else: # regular path
            k = self.c_k(x).view(B, T, self.n_kv_heads, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_heads//2, 2*self.head_dim)

            if self.rope:
                cos, sin = self.rotary(q, start_pos)
            q, k = (F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))) if self.qk_norm else (q, k) # QK norm suggested by @Grad62304977
            if self.rope:
                q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # RoPE
            
            # split k heads into two groups
            k = k.view(B, T, 2, self.n_kv_heads//2, self.head_dim)
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

        if self.scalable_softmax:
            # scalable-softmax (https://arxiv.org/abs/2501.19399): multiply q by s*log(n)
            #log_pos = torch.arange(start_pos+1, start_pos+T+1, device=q.device).view(1, T, 1, 1).float().log()
            #q = (self.softmax_scaler.view(1, 1, -1, 1) * log_pos) * q

            pos = torch.arange(start_pos+1, start_pos+T+1, device=q.device).view(1, T, 1, 1)
            log_pos = pos.float().log() if wsize <= 0 else torch.clamp_max(pos.float(), wsize).log()
            q = (self.softmax_scaler.view(1, 1, -1, 1) * log_pos) * q
            
        # split q heads into two groups
        q = q.view(B, T, 2, self.n_heads//2, self.head_dim)
        q1, q2 = q[:, :, 0], q[:, :, 1]
        
        new_pos = start_pos + T
        if cache is not None:
            past_k1, past_k2, past_v, _ = cache
            if past_k1 is not None: # not first token
                k1 = torch.cat([past_k1, k1], dim=1)
                k2 = torch.cat([past_k2, k2], dim=1)
                v = torch.cat([past_v, v], dim=1)
            
            cache = (k1, k2, v, new_pos)
        
        k1, k2, v = repeat_kv(k1, self.n_kv_repeats), repeat_kv(k2, self.n_kv_repeats), repeat_kv(v, self.n_kv_repeats) # GQA

        y1 = flex_head_fa.flash_attn_func(q1.bfloat16(), k1.bfloat16(), v.bfloat16(), causal=True, window_size=(wsize, wsize))
        y2 = flex_head_fa.flash_attn_func(q2.bfloat16(), k2.bfloat16(), v.bfloat16(), causal=True, window_size=(wsize, wsize))
        #y = (y1 - y2).contiguous()
        lambda_1 = torch.exp(torch.sum(self.lambda_q1*self.lambda_k1, dim=-1).float()).type_as(y1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2*self.lambda_k2, dim=-1).float()).type_as(y2)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        y = (y1 - lambda_full * y2).contiguous()
        # We found that group norm doesn't improve on long scale the results
        # y = F.rms_norm(y, (2*self.head_dim,)) * (1 - self.lambda_init) 
        #y = y.view(B, T, self.d_model*self.expand_factor)
        return y, cache
        
    def get_kv(self):
        return self.last_k1, self.last_k2, self.last_v
    
    def get_empty_cache(self):
        return (None, None, None, 0) # (k1_cache, k2_cache, v_cache, pos)

class DiffAttention(MixerDiffAttention):
    def __init__(self, config: NanoConfig, swa: bool = False, kv_share: bool = False, layer_depth: int = 0):
        super().__init__(config, swa=swa, kv_share=kv_share, layer_depth=layer_depth)
        
        # output projection
        self.c_proj = nn.Linear(self.expand_factor * self.d_model, self.d_model, bias=False)
        #self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
    
    def forward(self, x, external_kv=None, cache=None):
        y, cache = super().forward(x, external_kv, cache)
        y = self.c_proj(y)
        return y, cache
    
class MixerNativeSparseAttention(nn.Module):
    def __init__(self, config: NanoConfig, swa=False):
        assert swa==False

        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_model = config.d_model
        self.expand_factor = config.expand_factor
        #assert self.expand_factor==4, "should be 4 here"
        #self.expand_factor = config.expand_factor
        self.d_head = min(128, (self.d_model*self.expand_factor) // self.n_heads) # cap d_head to 128
        self.expand_factor = (self.n_heads*self.d_head)/self.d_model # and recompute expand_factor
        self.kernel_size = config.nsa_kernel_size
        self.kernel_stride = config.nsa_kernel_stride
        self.block_size = config.nsa_block_size
        self.topn = config.nsa_topn
        self.window_size = config.nsa_swa

        assert self.d_model % self.n_heads == 0
        
        self.c_q = nn.Linear(self.d_model, self.n_heads*self.d_head, bias=False)
        self.c_k = nn.Linear(self.d_model, self.n_kv_heads*self.d_head, bias=False)
        self.c_v = nn.Linear(self.d_model, self.n_kv_heads*self.d_head, bias=False)
        self.c_g = nn.Linear(self.d_model, 3*self.n_heads, bias=False)
        self.wk = torch.nn.Parameter(torch.zeros(self.n_kv_heads, self.kernel_size))
        self.wv = torch.nn.Parameter(torch.zeros(self.n_kv_heads, self.kernel_size))
        self.pe = torch.nn.Parameter(torch.zeros(self.n_kv_heads, self.kernel_size, self.d_head))
        self.rotary = RotaryEmbedding(RopeConfig(head_dim=self.d_head, rope_theta=10000))

    def forward(self, x, external_kv=None, cache=None):
        B, T, _ = x.size()

        # here, pass into B*L mode and create cu_seqlens
        x = x.view(B*T, -1)
        cu_seqlens = torch.arange(0, B*T+1, T, dtype=torch.int32, device=x.device)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

        # qkv proj
        q = self.c_q(x).view(B*T, self.n_heads, self.d_head)
        k = self.c_k(x).view(B*T, self.n_kv_heads, self.d_head)
        v = self.c_v(x).view(B*T, self.n_kv_heads, self.d_head)
        g_cmp, g_slc, g_swa = self.c_g(x).sigmoid().view(B*T, self.n_heads, 3).unbind(-1)

        # no need to replicate k/v for GQA, this is handled in the attentions

        # compression attention
        #k_cmp, cu_seqlens_cmp = linear_compress(k, self.wk, cu_seqlens, self.kernel_size, self.kernel_stride, self.pe)
        #v_cmp, _ = linear_compress(v, self.wv, cu_seqlens, self.kernel_size, self.kernel_stride, None)

        k_cmp, cu_seqlens_cmp = weightedpool_compress(k, self.wk, cu_seqlens, self.kernel_size, self.kernel_stride, self.pe)
        v_cmp, _ = weightedpool_compress(v, self.wv, cu_seqlens, self.kernel_size, self.kernel_stride, self.pe)

        #k_cmp, cu_seqlens_cmp = avgpool_compress(k, None, cu_seqlens, self.kernel_size, self.kernel_stride, self.pe)
        #v_cmp, _ = avgpool_compress(v, None, cu_seqlens, self.kernel_size, self.kernel_stride, self.pe)

        q = self.rotary(q, cu_seqlens)
        k_cmp = self.rotary(k_cmp, cu_seqlens_cmp, stride=self.kernel_stride)
        
        compressed_seqlens = cu_seqlens_cmp[1:] - cu_seqlens_cmp[:-1]
        o_cmp, topn_idx = compressed_attention(q, k_cmp, v_cmp, self.kernel_size, self.kernel_stride, self.block_size, self.topn, cu_seqlens, cu_seqlens_cmp, seqlens.max().item(), compressed_seqlens.max().item(), None, init_blocks=1, local_blocks=2)
        o = g_cmp[..., None] * o_cmp

        # selection attention
        k = self.rotary(k, cu_seqlens)
        o_slc = topk_sparse_attention(q, k, v, topn_idx, self.block_size, cu_seqlens, None)
        o += g_slc[..., None] * o_slc

        # local attention
        o_swa = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, seqlens.max().item(), seqlens.max().item(), causal=True, window_size=(self.window_size, -1))
        o += g_swa[..., None] * o_swa

        o = o.view(B, T, int(self.d_model*self.expand_factor))
        return o, None

class NativeSparseAttention(MixerNativeSparseAttention):
    def __init__(self, config: NanoConfig):
        super().__init__(config)

        # output projection
        self.c_proj = nn.Linear(int(self.expand_factor*self.d_model), self.d_model, bias=False)
        #self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x, external_kv=None, cache=None):
        y, cache = super().forward(x)
        y = self.c_proj(y)
        return y, cache

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
