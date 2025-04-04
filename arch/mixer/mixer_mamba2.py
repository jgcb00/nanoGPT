# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from config import NanoConfig

# warning: inference is not supported with only one starting token
# the prompt should be >d_conv tokens long

class MixerMamba2(nn.Module):
    def __init__(
        self,
        config: NanoConfig,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        A_init_range=(1, 16),
        D_has_hdim=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        conv_init=None,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.conv_init = conv_init
        self.expand = config.expand_factor
        self.sequence_parallel = sequence_parallel
        self.world_size = 1
        self.local_rank = 0
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = config.headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert config.ngroups % self.world_size == 0
        self.ngroups = config.ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = config.rmsnorm
        self.norm_before_gate = config.norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=config.d_conv,
            groups=conv_dim,
            padding=config.d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        # assert not(config.model == "dragon" and self.rmsnorm), "When using Dragon, no output norm should be used."
        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // config.ngroups, **factory_kwargs)
            
            # note about RMSNormGated (f is silu):
            # if norm_before_gate : norm(x) * f(z)
            # else                : norm(x * f(z))

    def forward(self, u, cache=None):
        """
        u: (batch, seqlen, hidden_dim)
        cache: TODO
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        return_cache = False
        if cache is not None and seqlen>1:
            # prefill
            cache = None
            return_cache = True
        
        if cache is not None:
            out, cache = self.step(u, cache)
            return out, cache

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if self.use_mem_eff_path:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=None,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                return_final_states=return_cache,
                **dt_limit_kwargs,
            )

            if return_cache:
                # get h_cache from out
                out, h_cache = out

                # get conv_cache with the last d_conv entries of xBC
                _, xBC, _ = torch.split(zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1)
                xBC_t = xBC.transpose(1, 2)
                conv_cache = F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)) # pad if sequence length is less than self.d_conv; otherwise, truncate

                cache = (h_cache, conv_cache)
        
        else:
            raise NotImplementedError("Not implemented")
        return out, cache
    
    def step(self, u, cache):
        """
        u: (B, 1, D)
        cache: (h_cache, conv_cache)
        """

        h_cache, conv_cache = cache
        
        zxbcdt = self.in_proj(u.squeeze(1))  # (B, 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_inner - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(zxbcdt, [d_mlp, d_mlp, self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1)

        # conv step
        xBC = causal_conv1d_update(xBC, conv_cache, rearrange(self.conv1d.weight, "d 1 w -> d w"), self.conv1d.bias, self.activation)

        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float()) # (n_heads)

        # SSM step
        A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
        dt = repeat(dt, "b h -> b h p", p=self.headdim)
        dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
        D = repeat(self.D, "h -> h p", p=self.headdim)
        B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
        C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
        x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        if not self.rmsnorm:
            z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            
        y = selective_state_update(h_cache, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None, dt_bias=dt_bias, dt_softplus=True)
        y = rearrange(y, "b h p -> b (h p)")

        if self.rmsnorm:
            y = self.norm(y, z)
        
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        
        return y.unsqueeze(1), (h_cache, conv_cache)

    def get_empty_cache(self):
        return (None, None) # (h_cache, conv_cache)

class Mamba2(MixerMamba2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        #self.out_proj.weight.data.zero_()
    
    def forward(self, u, cache=None):
        out, cache = super().forward(u, cache=cache)
        out = self.out_proj(out)
        return out, cache
