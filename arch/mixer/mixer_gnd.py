# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import math
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from config import NanoConfig
from arch.utils import ScaledLinear

class MixerGatedDeltaNet(nn.Module):
    def __init__(
        self,
        config: NanoConfig,
        conv_bias=False,
        conv_init=None,
        norm_eps=1e-5
    ):
        super().__init__()
        self.config = config
        
        self.d_model = config.d_model
        self.expand_factor = config.expand_factor
        self.expand_v = config.expand_v
        self.use_gate = config.use_gate

        self.conv_size = config.d_conv
        self.conv_bias = conv_bias
        self.conv_init = conv_init

        self.n_heads = config.n_heads
        self.n_heads_local = self.n_heads//1
        self.d_head = int(self.d_model * (self.expand_factor/2)) // self.n_heads

        self.key_dim = self.n_heads * self.d_head
        self.value_dim = self.key_dim * self.expand_v
        self.head_k_dim = self.d_head
        self.head_v_dim = self.d_head * self.expand_v
        self.silu = nn.SiLU()

        self.dk = self.head_k_dim
        self.dv = self.head_v_dim
        self.per_head_proj = 2*self.dk + self.dv + 2 # [q k v b a] per head
        in_proj_dim_global = self.n_heads * self.per_head_proj
        self.in_proj = ScaledLinear(config, self.d_model, in_proj_dim_global, bias=False)

        # hard coded for now todo
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        A_init_range=(1, 16)

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.n_heads_local) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_bias = nn.Parameter(inv_dt)
        # Our initialization would set all Linear.bias to zero,
        # need to mark this one as _no_reinit
        self.dt_bias._no_reinit = True
        # Just to be explicit. Without this we already don't
        # put wd on dt_bias because of the check

        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(
            self.n_heads_local, dtype=torch.float32, device=torch.cuda.current_device()
        ).uniform_(*A_init_range)
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # ShortConvolution is a wrapper around nn.Conv1d (for definition) and causal_conv1d (for forward)
        self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=self.conv_size,
                activation='silu'
            )
        self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=self.conv_size,
                activation='silu'
            )
        self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=self.conv_size,
                activation='silu'
            )

        if self.conv_init is not None:
            nn.init.uniform_(self.q_conv1d.weight, -self.conv_init, self.conv_init)
            nn.init.uniform_(self.k_conv1d.weight, -self.conv_init, self.conv_init)
            nn.init.uniform_(self.v_conv1d.weight, -self.conv_init, self.conv_init)

        if self.use_gate:
            # gate projection
            if self.config.gate_type_gdn == "elementwise":
                self.g_proj = ScaledLinear(config, self.d_model, self.d_model*self.expand_factor, bias=False)
            elif self.config.gate_type_gdn == "headwise":
                self.g_proj = ScaledLinear(config, self.d_model, self.n_heads, bias=False)
            else:
                raise ValueError(f"Unknown gate type: {self.config.gate_type_gdn}")

            # activation function
            if self.config.gate_act_gdn == "silu":
                self.act_func_gate = F.silu
            elif self.config.gate_act_gdn == "srelu":
                self.act_func_gate = lambda g: F.relu(g).square()
            elif self.config.gate_act_gdn == "sigmoid":
                self.act_func_gate = F.sigmoid
            else:
                raise ValueError(f"Unknown gate activation: {self.config.gate_act_gdn}")
        
        # state passing
        self.register_buffer("prev_state", None, persistent=False) # (B, H, d_k, d_v)

    def _sample_init_state(self, B):
        if not self.training or self.prev_state is None:
            return None
        if self.prev_state.size(0) != B:
            return None
        p = self.config.p_state_passing
        if p == 0.0:
            return None
        keep_mask = (torch.rand(B, self.n_heads, device=self.prev_state.device) < p).to(self.prev_state.dtype)
        keep_mask = keep_mask[..., None, None] # (B, H, 1, 1)
        return (self.prev_state * keep_mask).detach()

    def _store_final_state(self, final_state):
        if self.training and final_state is not None:
            self.prev_state = final_state.detach()

    def forward(self, hidden_states, cache=None):
        """
        hidden_states: (b, l, d)
        Returns: same shape as hidden_states
        """

        _, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if q_len <= 64 else 'chunk'
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        # input projection (TP-aware)
        qkvba = self.in_proj(hidden_states) # (l, b, H_local * per_head_proj)
        # [L,B,(H*P)] -> [B,L,H,P]
        qkvba = rearrange(qkvba, "b l (h p) -> b l h p", h=self.n_heads_local).contiguous()
        # split per head: [B,L,H,dk/dk/dv/1/1]
        q_proj = qkvba[..., 0:self.dk]
        k_proj = qkvba[..., self.dk:2*self.dk]
        v_proj = qkvba[..., 2*self.dk:2*self.dk+self.dv]
        b_proj = qkvba[..., 2*self.dk+self.dv:2*self.dk+self.dv+1]
        a_proj = qkvba[..., 2*self.dk+self.dv+1:]  
        # concat for conv
        q_proj = rearrange(q_proj, "b l h d -> b l (h d)")
        k_proj = rearrange(k_proj, "b l h d -> b l (h d)")
        v_proj = rearrange(v_proj, "b l h d -> b l (h d)")
        b_proj = rearrange(b_proj, "b l h d -> b l (h d)") # d=1
        a_proj = rearrange(a_proj, "b l h d -> b l (h d)")

        h_cache, q_conv_cache, k_conv_cache, v_conv_cache = None, None, None, None
        if cache is not None:
            h_cache, q_conv_cache, k_conv_cache, v_conv_cache = cache

        q, q_conv_cache = self.q_conv1d(x=q_proj,
                             mask=None, 
                             cache=q_conv_cache,
                             output_final_state=(cache is not None),
                             seq_idx=None)
        k, k_conv_cache = self.k_conv1d(x=k_proj,
                             mask=None,
                             cache=k_conv_cache,
                             output_final_state=(cache is not None),
                             seq_idx=None)
        v, v_conv_cache = self.v_conv1d(x=v_proj,
                             mask=None,
                             cache=v_conv_cache,
                             output_final_state=(cache is not None),
                             seq_idx=None)

        # back to per-head for kernels
        q = rearrange(q, "b l (h d) -> b l h d", d=self.dk)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.dk)
        v = rearrange(v, "b l (h d) -> b l h d", d=self.dv)

        beta = b_proj.sigmoid()
        g = -self.A_log.float().exp() * self.config.uscaling_dt_mul * F.softplus(a_proj.float() + self.dt_bias)

        if mode == 'chunk':
            if cache is None: # training, state passing
                init_state = self._sample_init_state(hidden_states.size(0))
            else: # inference, kv cache
                init_state = h_cache

            o, final_state = chunk_gated_delta_rule(
                q=q.bfloat16(),
                k=k.bfloat16(),
                v=v.bfloat16(),
                g=g,
                beta=beta,
                scale=None if not self.config.use_uscaling else 1/self.head_k_dim,
                initial_state=init_state,
                output_final_state=self.training or (cache is not None),
                cu_seqlens=None, # for varlen training
                head_first=False,
                use_qk_l2norm_in_kernel=True
            ) # (b t h d) where d is head_v_dim

            self._store_final_state(final_state if self.training else None)
            h_cache = final_state if cache is not None else None
        elif mode == 'fused_recurrent':
            o, h_cache = fused_recurrent_gated_delta_rule(
                q=q.bfloat16(),
                k=k.bfloat16(),
                v=v.bfloat16(),
                g=g,
                beta=beta,
                scale=None if not self.config.use_uscaling else 1/self.head_k_dim,
                initial_state=h_cache,
                output_final_state=(cache is not None),
                cu_seqlens=None,
                head_first=False,
                use_qk_l2norm_in_kernel=True
            ) # (b t h d) where d is head_v_dim
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        
        if self.use_gate:
            # gate
            if self.config.gate_type_gdn == "elementwise":
                g = self.g_proj(hidden_states).view(o.size(0), o.size(1), o.size(2), o.size(3)) # (B, L, H, D)
            elif self.config.gate_type_gdn == "headwise":
                g = self.g_proj(hidden_states).view(o.size(0), o.size(1), o.size(2), 1) # (B, L, H, 1)
            else:
                raise ValueError(f"Unknown gate type: {self.config.gate_type_gdn}")
            o = o * self.act_func_gate(g)

        return o, (h_cache, q_conv_cache, k_conv_cache, v_conv_cache)
    
    def get_empty_cache(self):
        return (None, None, None, None) # (h_cache, q_conv_cache, k_conv_cache, v_conv_cache)
    
class GatedDeltaNet(MixerGatedDeltaNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_proj = ScaledLinear(None, self.value_dim, self.d_model, bias=False)
        #self.out_proj.weight.data.zero_()
    
    def forward(self, hidden_states, cache=None):
        out, cache = super().forward(hidden_states, cache=cache)
        out = self.out_proj(out)
        return out, cache
