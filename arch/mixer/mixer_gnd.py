# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Union
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from config import NanoConfig

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
        self.d_head = int(self.d_model * (self.expand_factor/2)) // self.n_heads

        self.key_dim = self.n_heads * self.d_head
        self.value_dim = self.key_dim * self.expand_v
        self.head_k_dim = self.d_head
        self.head_v_dim = self.d_head * self.expand_v
        self.silu = nn.SiLU()

        self.n_heads_local = self.n_heads // 1

        in_proj_dim = (
            self.key_dim +  # q_proj
            self.key_dim +  # k_proj
            self.value_dim +  # v_proj
            self.n_heads +  # b_proj
            self.n_heads  # a_proj
        )

        self.q_slice = slice(0, self.key_dim)
        self.k_slice = slice(self.key_dim, 2 * self.key_dim)
        self.v_slice = slice(2 * self.key_dim, 2 * self.key_dim + self.value_dim)
        self.b_slice = slice(
            2 * self.key_dim + self.value_dim,
            2 * self.key_dim + self.value_dim + self.n_heads,
        )
        self.a_slice = slice(
            2 * self.key_dim + self.value_dim + self.n_heads,
            2 * self.key_dim + self.value_dim + 2 * self.n_heads,
        )

        self.in_proj = nn.Linear(self.d_model, in_proj_dim, bias=False)

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
                self.g_proj = nn.Linear(self.d_model, self.d_model*self.expand_factor, bias=False)
            elif self.config.gate_type_gdn == "headwise":
                self.g_proj = nn.Linear(self.d_model, self.n_heads, bias=False)
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

        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, hidden_states, cache=None):
        """
        hidden_states: (b, l, d)
        Returns: same shape as hidden_states
        """

        _, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if q_len <= 64 else 'chunk'
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        qkvba = self.in_proj(hidden_states) # (b, l, D)
        
        # split proj into q, k, v, b, a
        q_proj = qkvba[:, :, self.q_slice]
        k_proj = qkvba[:, :, self.k_slice]
        v_proj = qkvba[:, :, self.v_slice]
        b_proj = qkvba[:, :, self.b_slice]
        a_proj = qkvba[:, :, self.a_slice]

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
        
        q, k = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)
        beta = b_proj.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a_proj.float() + self.dt_bias)

        if mode == 'chunk':
            o, h_cache = chunk_gated_delta_rule(
                q=q.bfloat16(),
                k=k.bfloat16(),
                v=v.bfloat16(),
                g=g,
                beta=beta,
                initial_state=h_cache,
                output_final_state=(cache is not None),
                cu_seqlens=None, # for varlen training
                head_first=False,
                use_qk_l2norm_in_kernel=True
            ) # (b t h d) where d is head_v_dim
        elif mode == 'fused_recurrent':
            o, h_cache = fused_recurrent_gated_delta_rule(
                q=q.bfloat16(),
                k=k.bfloat16(),
                v=v.bfloat16(),
                g=g,
                beta=beta,
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
        self.out_proj = nn.Linear(self.value_dim, self.d_model, bias=False)
        #self.out_proj.weight.data.zero_()
    
    def forward(self, hidden_states, cache=None):
        out, cache = super().forward(hidden_states, cache=cache)
        out = self.out_proj(out)
        return out, cache
