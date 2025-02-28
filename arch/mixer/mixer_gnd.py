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
from fla.ops.gated_delta_rule import (chunk_gated_delta_rule,
                                      fused_recurrent_gated_delta_rule)

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
        self.expand_v = config.expand_v
        self.use_gate = config.use_gate

        self.conv_size = config.d_conv
        self.conv_bias = conv_bias
        self.conv_init = conv_init

        self.n_heads = config.n_heads
        self.d_head = self.d_model // self.n_heads

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
            self.g_proj = nn.Linear(self.d_model, self.value_dim, bias=False)
            if config.rmsnorm:
                self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps) # norm(x) * f(z)
        else:
            if config.rmsnorm:
                self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        
        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, hidden_states):
        """
        hidden_states: (b, l, d)
        Returns: same shape as hidden_states
        """
        _, batch, dim = hidden_states.shape

        qkvba = self.in_proj(hidden_states) # (b, l, D)
        
        # split proj into q, k, v, b, a
        q_proj = qkvba[:, :, self.q_slice]
        k_proj = qkvba[:, :, self.k_slice]
        v_proj = qkvba[:, :, self.v_slice]
        b_proj = qkvba[:, :, self.b_slice]
        a_proj = qkvba[:, :, self.a_slice]

        q, _ = self.q_conv1d(x=q_proj,
                             mask=None, 
                             cache=None,
                             output_final_state=False,
                             seq_idx=None)
        k, _ = self.k_conv1d(x=k_proj,
                             mask=None,
                             cache=None,
                             output_final_state=False,
                             seq_idx=None)
        v, _ = self.v_conv1d(x=v_proj,
                             mask=None,
                             cache=None,
                             output_final_state=False,
                             seq_idx=None)
        
        q, k = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)
        beta = b_proj.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a_proj.float() + self.dt_bias)

        o, _ = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=None, # for varlen training
                head_first=False,
                use_qk_l2norm_in_kernel=True
            ) # (b t h d) where d is head_v_dim
        
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            if self.config.rmsnorm:
                o = self.o_norm(o, g)
            else:
                o = o * g * F.sigmoid(g)
        else:
            if self.config.rmsnorm:
                o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)').contiguous()
        return o
    
class GatedDeltaNet(MixerGatedDeltaNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_proj = nn.Linear(self.value_dim, self.d_model, bias=False)
        self.out_proj.weight.data.zero_()
    
    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        return self.out_proj(out)