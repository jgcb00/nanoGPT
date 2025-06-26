import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.modules import RMSNorm

from config import NanoConfig

def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y

class MixerLaCT(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config

        self.d_model = config.d_model
        self.inter_multi = config.expand_factor*config.lact_expand_factor

        self.qkv = nn.Linear(config.d_model, 3*config.d_model, bias=False)
        
        self.q_norm = RMSNorm(config.d_model)
        self.k_norm = RMSNorm(config.d_model)

        #self.rope_theta = rope_theta
        #self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

        ### Fast Weight init
        self.use_muon = config.lact_use_muon
        self.use_momentum = config.lact_use_momentum
        assert not(self.use_muon and self.use_momentum), "Cannot use both Muon and Momentum at the same time."
        self.lact_chunk_size = config.lact_chunk_size
        self.n_heads = config.lact_n_heads
        self.d_head = config.d_model // self.n_heads
        
        d_in, d_out = self.d_head, self.d_head
        d_h = int(d_in * self.inter_multi)

        self.d_h = d_h
        self.d_in = d_in
        self.d_out = d_out
        self.w0_w2_low_rank = config.lact_w0_w2_low_rank
        self.fw_init_gain = config.lact_fw_init_gain

        # Low Rank parameterization of the fast weights.  
        # This is a compromise to keep the number of parameters low when comparing against baselines. 
        # Idealy, low-rank parameterization always hurts the performance. 
        if self.w0_w2_low_rank > 0:
            self.w0 = LowRankFastWeight(self.n_heads, d_h, d_in, self.w0_w2_low_rank, init_gain=self.fw_init_gain, add_identity=True)
            self.w2 = LowRankFastWeight(self.n_heads, d_h, d_in, self.w0_w2_low_rank, init_gain=self.fw_init_gain, add_identity=True)
        else:
            self.w0 = nn.Parameter(torch.randn(self.n_heads, int(d_h), d_in) / math.sqrt(d_in)) # [n_heads, d_h, d_in]
            self.w2 = nn.Parameter(torch.randn(self.n_heads, int(d_h), d_in) / math.sqrt(d_in)) # [n_heads, d_h, d_in]
        self.w1 = nn.Parameter(torch.randn(self.n_heads, int(d_out), d_h) / math.sqrt(d_h)) # [n_heads, d_out, d_h]
        
        #### Per-Token LR parameterization.
        self.lr_dim = int(config.lact_lr_dim * 3 * self.n_heads) # 1 LR for each of w0, w1, w2
        self.lr_proj = nn.Linear(config.d_model, self.lr_dim)
        base_lr = 0.001
        self.base_lr_inv = inv_softplus(base_lr)
        
        if self.use_momentum:
            self.momentum_proj = nn.Sequential(
                nn.Linear(config.d_model, self.n_heads),
                nn.Sigmoid(),
            )

    def forward(self, hidden_states, cache=None):
        """
        hidden_states: (b, l, d)
        Returns: same shape as hidden_states
        """

        batch_size, _, _ = hidden_states.size()

        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        q, k = self.q_norm(q), self.k_norm(k)

        q = rearrange(q, 'b s (n_h d) -> (b n_h) s d', n_h=self.n_heads)
        k = rearrange(k, 'b s (n_h d) -> (b n_h) s d', n_h=self.n_heads)
        v = rearrange(v, 'b s (n_h d) -> (b n_h) s d', n_h=self.n_heads)
        q, k, v = l2_norm(F.silu(q)), l2_norm(F.silu(k)), F.silu(v)

        """(lact rope) fast_q = rearrange(fast_q, '(b n_h) s d -> b s (n_h d)', n_h=self.n_heads)
        fast_k = rearrange(fast_k, '(b n_h) s d -> b s (n_h d)', n_h=self.n_heads)

        fast_q = rearrange(fast_q, 'b s (n_h d) -> b s n_h d', n_h=self.num_heads)
        fast_k = rearrange(fast_k, 'b s (n_h d) -> b s n_h d', n_h=self.num_heads)

        fast_q, fast_k = self.rotary(fast_q, fast_k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        fast_q = rearrange(fast_q, 'b s n_h d -> b s (n_h d)', n_h=self.num_heads)
        fast_k = rearrange(fast_k, 'b s n_h d -> b s (n_h d)', n_h=self.num_heads)

        fast_q = rearrange(fast_q, 'b s (n_h d) -> (b n_h) s d', n_h=self.n_heads)
        fast_k = rearrange(fast_k, 'b s (n_h d) -> (b n_h) s d', n_h=self.n_heads)"""
        
        # fast weights
        if self.w0_w2_low_rank > 0:
            fw_w0 = self.w0().repeat(batch_size, 1, 1)
            fw_w2 = self.w2().repeat(batch_size, 1, 1)
        else:
            fw_w0 = self.w0.repeat(batch_size, 1, 1) # [nh, d_h, d_in] -> [b*nh, d_h, d_in]
            fw_w2 = self.w2.repeat(batch_size, 1, 1) # [nh, d_h, d_in] -> [b*nh, d_h, d_in]
        fw_w1 = self.w1.repeat(batch_size, 1, 1) # [nh, d_out, d_h] -> [b*nh, d_out, d_h]

        # lr, momentum
        lr = self.lr_proj(hidden_states) # [b, s, num_heads * lr_dim_per_head]
        lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)

        fw_lr = rearrange(lr, 'b s (n_h lr_dim) -> (b n_h) s lr_dim', n_h=self.n_heads)
        fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

        if self.use_momentum:
            momentum = self.momentum_proj(hidden_states) # [b, s, nh]
            momentum = rearrange(momentum, 'b s (n_h d) -> (b n_h) s d', n_h=self.n_heads)
        else:
            momentum = None

        # [b * nh, s, d_ttt_head]
        o = block_causal_lact_swiglu(
            fw_w0, fw_w1, fw_w2, q, k, v,
            fw_lr1, fw_lr2, fw_lr3,
            chunk_size=self.lact_chunk_size,
            use_muon=self.use_muon,
            momentum=momentum
        )

        assert len(o.shape) == 4 #todo temp

        return o, None

    def get_empty_cache(self):
        return

class LaCT(MixerLaCT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        #self.out_proj.weight.data.zero_()
    
    def forward(self, hidden_states, cache=None):
        out, cache = super().forward(hidden_states, cache=cache)
        out = self.out_proj(out)
        return out, cache

class LowRankFastWeight(nn.Module):
    """
    Low rank fast weight. This is a compromise to keep the number of parameters low when comparing against baselines. 
    Idealy, low-rank parameterization always hurts the performance. 
    Args:
        num_heads: number of heads
        out_features: output features
        in_features: input features
        rank: rank of the low rank fast weight
        init_gain: initialization gain
        add_identity: whether to add identity matrix to the fast weight
    Returns:
        W: [num_heads, out_features, in_features]
    W = W_left @ W_right + I * 0.5
        where I is the identity matrix if add_identity is True.
    """
    def __init__(self, num_heads, out_features, in_features, rank=32, init_gain=0.5, add_identity=False):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.add_identity = add_identity
        
        self.w_left = nn.Parameter(torch.randn(num_heads, out_features, rank))
        self.w_right = nn.Parameter(torch.randn(num_heads, rank, in_features))
        self.init_gain = init_gain

        print("init low rank fast weight", num_heads, out_features, in_features, rank)

    def _init_weights(self):
        
        nn.init.normal_(self.w_left, std=1.0 / math.sqrt(self.rank) * self.init_gain)
        nn.init.normal_(self.w_right, std=1.0 / math.sqrt(self.in_features) * self.init_gain)

    def forward(self,):
        """
        Returns:
            W: [num_heads, out_features, in_features]
            W = W_left @ W_right + I * 0.5
            where I is the identity matrix if add_identity is True.
        """

        W = self.w_left @ self.w_right

        if self.add_identity:
            W += torch.eye(self.out_features, self.in_features, device=W.device, dtype=W.dtype).unsqueeze(0) * 0.5

        return W

@torch.compile()
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx

@torch.compile()
def l2_norm(x: torch.Tensor):
    """
    x: [b, l, d]
    """
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)  # norm will upcast to float32
    return ret.type(x_type)

@torch.compile()
def zeropower_via_newtonschulz5(G):
    """
    This is an updated version of the zeropower_via_newtonschulz5 function in here:
    https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L26
    The code is modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49, which contains the original muon implementation.
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Args:
        G: [b, d, d']
    Returns:
        X: [b, d, d']
    FLOPS:  When d=d', Total FLOPS=30 * b * d^3
    """
    assert len(G.shape) == 3
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X

@torch.compile()
def block_causal_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int=2048,  # test-time training chunk sizeq
    use_muon: bool = False,
    momentum: torch.Tensor = None, # [b, s, 1]
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))
    
    About precision:
        w0, w1, w2 are mostly likely fp32. 
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.
    
    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon, ignore last chunk)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, dv]
    """
    
    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dh, l]
        qi = q[:, :, s_index:e_index]
        # [b, l, d/1] fp32
        lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32

        # use previous w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
        
        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1.transpose(1, 2), vi)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
        # in bmm two mat is fp32, but the result is bf16.
        # it's better to cast the mat to bf16 before bmm.
        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(
            vi, (hidden.transpose(1, 2) * lr1i).type_as(vi)
        )  # [b, d, d]
        # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :] 
            m_i = m_i.mean(dim=1, keepdim=True)

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)
            # legacy code for different global lr for muon. Conclusion: 1.0 is good
            # if muon_w0_lr is not None:
            #     # lr is fp32 (after softplus)
            #     # in future version, we can cast it before input. TODO
            #     dw1 = (dw1 * muon_w1_lr).type_as(w1)
            #     dw0 = (dw0 * muon_w0_lr).type_as(w0)
            #     dw2 = (dw2 * muon_w2_lr).type_as(w2)

        w1 = w1 + dw1
        w0 = w0 + dw0
        w2 = w2 + dw2
    
        # Do channel-wise l2 norm.  conceptually like post-norm.
        w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    # use the last w0 and w1 to get the final output
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2)
