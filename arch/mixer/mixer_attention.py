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

from transformer_engine.pytorch.attention import _SplitAlongDim
SplitAlongDim = _SplitAlongDim.apply

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
    def __init__(self, config: NanoConfig, swa: bool = False, kv_share: bool = False):
        super().__init__()

        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_repeats = self.n_heads // self.n_kv_heads
        self.d_model = config.d_model
        self.kv_share = kv_share
        self.expand_factor = config.expand_factor
        self.d_head = (self.d_model*self.expand_factor) // self.n_heads
        self.swa, self.swa_window_size = swa, config.swa_window_size
        self.rope = self.swa or not config.rope_to_nope
        self.qk_norm = config.qk_norm
        self.scalable_softmax = config.scalable_softmax
        if config.disable_scalable_softmax_for_local and self.swa:
            self.scalable_softmax = False
        self.use_gate = config.use_gate_attn
        assert self.d_model % self.n_heads == 0

        if self.config.groupnorm:
            if self.config.model == "dragon":
                if self.config.groupnorm_unique:
                    if not self.config.groupnorm_unique_independent:
                        self.group_norm = nn.RMSNorm((self.n_heads, self.d_head), elementwise_affine=config.groupnorm_weights, eps=config.eps_rmsnorm)
                    else:
                        self.group_norm = HeadWiseRMSNorm(n_heads=self.n_heads, d_head=self.d_head, eps=config.eps_rmsnorm)
                else:
                    self.group_norm = nn.RMSNorm(self.d_head, elementwise_affine=config.groupnorm_weights, eps=config.eps_rmsnorm)

        proj_dim = self.d_head * (self.n_heads + 2 * (0 if kv_share else self.n_kv_heads))
        self.linear_qkv = nn.Linear(self.d_model, proj_dim, bias=False)

        if self.rope:
            if self.swa:
                self.rotary = Rotary(self.d_head, base=self.config.rope_theta_local) # 477=3k/(2pi)
            else:
                self.rotary = Rotary(self.d_head, base=self.config.rope_theta_global)
        if self.scalable_softmax:
            self.softmax_scaler = nn.Parameter(torch.ones(self.n_heads))

        if self.use_gate:
            # gate projection
            if self.config.gate_type_attn == "elementwise":
                self.g_proj = nn.Linear(self.d_model, self.d_model*self.expand_factor, bias=False)
            elif self.config.gate_type_attn == "headwise":
                self.g_proj = nn.Linear(self.d_model, self.n_heads, bias=False)
            else:
                raise ValueError(f"Unknown gate type: {self.config.gate_type_attn}")
            
            # activation function
            if self.config.gate_act_attn == "silu":
                self.act_func_gate = F.silu
            elif self.config.gate_act_attn == "srelu":
                self.act_func_gate = lambda g: F.relu(g).square()
            elif self.config.gate_act_attn == "sigmoid":
                self.act_func_gate = F.sigmoid
            else:
                raise ValueError(f"Unknown gate activation: {self.config.gate_act_attn}")

        if self.qk_norm:
            self.q_norm = nn.RMSNorm(self.d_head, elementwise_affine=config.rmsnorm_weights, eps=config.eps_rmsnorm)
            if not kv_share:
                self.k_norm = nn.RMSNorm(self.d_head, elementwise_affine=config.rmsnorm_weights, eps=config.eps_rmsnorm)
        self.last_k = None
        self.last_v = None

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        if self.kv_share:
            # reshape to [..., num_query_groups, heads_per_group * head_dim]
            q_dim = (self.n_heads // self.n_kv_heads) * self.d_head
            new_shape = mixed_qkv.size()[:-1] + (self.n_kv_heads, q_dim)
            query = mixed_qkv.view(*new_shape)
            # final shape [seq, batch, num_heads, head_dim]
            query = query.reshape(query.size(0), query.size(1), -1, self.d_head)

            if self.qk_norm:
                query = self.q_norm(query)

            return query

        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.n_kv_heads,
            (
                (self.n_heads // self.n_kv_heads + 2)
                * self.d_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.n_heads
                // self.n_kv_heads
                * self.d_head
            ),
            self.d_head,
            self.d_head,
        ]

        if SplitAlongDim is not None:
            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:
            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.d_head)

        if self.qk_norm:
            query = self.q_norm(query)
        if self.qk_norm:
            key = self.k_norm(key)

        return query, key, value

    def forward(self, hidden_states, external_kv=None, cache=None):
        # hidden_states: (B,T,D) -> y: (B,T,D)
        # external_kv: used in training, to use the kv from the previous layer
        # cache: used at inference, to store the kv from the past (k, v, pos)

        x = hidden_states

        B, T, _ = x.size()

        start_pos = cache[2] if cache is not None else 0
        
        if external_kv is not None: # kv-sharing path
            q = self.get_query_key_value_tensors(x)
            k, v = external_kv

            if self.rope:
                cos, sin = self.rotary(q, start_pos)
                q = apply_rotary_emb(q, cos, sin) # RoPE
        else: # regular path
            q, k, v = self.get_query_key_value_tensors(x)
            
            if self.rope:
                cos, sin = self.rotary(q, start_pos)
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

        if self.use_gate:
            # gate
            if self.config.gate_type_attn == "elementwise":
                g = self.g_proj(hidden_states).view(B, T, y.size(2), y.size(3)) # (B, L, H, D)
            elif self.config.gate_type_attn == "headwise":
                g = self.g_proj(hidden_states).view(B, T, y.size(2), 1) # (B, L, H, 1)
            else:
                raise ValueError(f"Unknown gate type: {self.config.gate_type_attn}")
            y = y * self.act_func_gate(g)

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
        self.expand_factor = config.expand_factor
        self.head_dim = (self.d_model * self.expand_factor) // self.n_heads
        self.swa, self.swa_window_size = swa, config.swa_window_size
        self.rope = self.swa or not config.rope_to_nope
        self.qk_norm = config.qk_norm
        self.scalable_softmax = config.scalable_softmax
        if config.disable_scalable_softmax_for_local and self.swa:
            self.scalable_softmax = False
        self.use_gate = config.use_gate_attn
        self.register_buffer("lambda_init", torch.tensor(0.8 - 0.6 * math.exp(-0.3 * layer_depth)))
        
        head_dim = self.head_dim // 2
        self.lambda_q1 = torch.nn.Parameter(torch.zeros(self.n_heads//2, head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = torch.nn.Parameter(torch.zeros(self.n_heads//2, head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = torch.nn.Parameter(torch.zeros(self.n_heads//2, head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = torch.nn.Parameter(torch.zeros(self.n_heads//2, head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        if self.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=config.rmsnorm_weights, eps=config.eps_rmsnorm)
            if not kv_share:
                self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=config.rmsnorm_weights, eps=config.eps_rmsnorm)

        if self.config.groupnorm:
            if self.config.model == "dragon":
                if self.config.groupnorm_unique:
                    if not self.config.groupnorm_unique_independent:
                        self.group_norm = nn.RMSNorm((self.n_heads//2, 2*self.head_dim), elementwise_affine=config.groupnorm_weights, eps=config.eps_rmsnorm)
                    else:
                        self.group_norm = HeadWiseRMSNorm(n_heads=self.n_heads//2, d_head=2*self.head_dim, eps=config.eps_rmsnorm)
                else:
                    self.group_norm = nn.RMSNorm(2*self.head_dim, elementwise_affine=config.groupnorm_weights, eps=config.eps_rmsnorm)
            else:
                self.group_norm = nn.RMSNorm(2*self.head_dim, elementwise_affine=config.groupnorm_weights, eps=config.eps_rmsnorm)
        
        assert self.d_model % self.n_heads == 0
        self.linear_qkv = nn.Linear(self.d_model, self.n_heads*self.head_dim + 2*self.n_kv_heads*self.head_dim, bias=False)
        if self.rope:
            if self.swa:
                self.rotary = Rotary(self.head_dim, base=self.config.rope_theta_local) # 477=3k/(2pi)
            else:
                self.rotary = Rotary(self.head_dim, base=self.config.rope_theta_global)
        if self.scalable_softmax:
            self.softmax_scaler = nn.Parameter(torch.ones(self.n_heads, dtype=torch.float32))

        if self.use_gate:
            # gate projection
            if self.config.gate_type_attn == "elementwise":
                self.g_proj = nn.Linear(self.d_model, self.d_model*self.expand_factor, bias=False)
            elif self.config.gate_type_attn == "headwise":
                self.g_proj = nn.Linear(self.d_model, self.n_heads//2, bias=False)
            else:
                raise ValueError(f"Unknown gate type: {self.config.gate_type_attn}")
            
            # activation function
            if self.config.gate_act_attn == "silu":
                self.act_func_gate = F.silu
            elif self.config.gate_act_attn == "srelu":
                self.act_func_gate = lambda g: F.relu(g).square()
            elif self.config.gate_act_attn == "sigmoid":
                self.act_func_gate = F.sigmoid
            else:
                raise ValueError(f"Unknown gate activation: {self.config.gate_act_attn}")
    
    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.n_kv_heads,
            (
                (self.n_heads // self.n_kv_heads + 2)
                * self.head_dim
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.n_heads
                // self.n_kv_heads
                * self.head_dim
            ),
            self.head_dim,
            self.head_dim,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.head_dim)

        # diff attn reshaping of v: 2 times less heads, dimension doubled
        value = value.reshape(value.size(0), value.size(1), value.size(2)//2, 2*value.size(3))

        if self.qk_norm:
            query = self.q_norm(query)

        if self.qk_norm:
            key = self.k_norm(key)

        return query, key, value

    def forward(self, hidden_states, external_kv=None, cache=None):
        # hidden_states: (B,T,D) -> y: (B,T,D)
        # external_kv: used in training, to use the kv from the previous layer
        # cache: used at inference, to store the kv from the past (k1, k2, v, pos)

        x = hidden_states

        B, T, _ = x.size()
        
        start_pos = cache[3] if cache is not None else 0
        
        q, k, v = self.get_query_key_value_tensors(x)

        if self.rope:
            cos, sin = self.rotary(q, start_pos)
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # RoPE

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
            
        # split q,k heads into two groups
        B, L, n_heads, d_head = q.size(0), q.size(1), q.size(2), q.size(3)
        n_kv_heads = k.size(2)

        q = q.view(B, L, n_heads, d_head)
        q1, q2 = q[:, :, torch.arange(0, n_heads, 2)].contiguous(), q[:, :, torch.arange(1, n_heads, 2)].contiguous()
        k = k.view(B, L, n_kv_heads, d_head)
        k1, k2 = k[:, :, torch.arange(0, n_kv_heads, 2)].contiguous(), k[:, :, torch.arange(1, n_kv_heads, 2)].contiguous()
        
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
        lambda_1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum(-1).float()) # (H)
        lambda_2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum(-1).float()) # (H)
        lambda_full = (lambda_1 - lambda_2 + self.lambda_init).view(1, 1, -1, 1).type_as(y1)
        y = (y1 - lambda_full * y2).contiguous()

        if self.use_gate:
            # gate
            if self.config.gate_type_attn == "elementwise":
                g = self.g_proj(hidden_states).view(B, T, y.size(2), y.size(3)) # (B, L, H, D)
            elif self.config.gate_type_attn == "headwise":
                g = self.g_proj(hidden_states).view(B, T, y.size(2), 1) # (B, L, H, 1)
            else:
                raise ValueError(f"Unknown gate type: {self.config.gate_type_attn}")
            y = y * self.act_func_gate(g)

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
    
    def forward(self, x, external_kv=None, cache=None):
        y, cache = super().forward(x, external_kv, cache)
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
