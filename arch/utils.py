import collections
import re
from typing import Dict
from functools import partial
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NanoConfig

class HeadWiseRMSNorm(nn.Module):
    def __init__(self, n_heads, d_head, eps=1e-5):
        super().__init__()
        self.rms = nn.RMSNorm(d_head, eps=eps, elementwise_affine=False)
        self.weight = nn.Parameter(torch.ones(n_heads, d_head))

    def forward(self, x):
        B, L, H, D = x.shape
        y = self.rms(x) * self.weight.view(1, 1, H, D)
        return y.view(B, L, H, D)

class _ScaledLinearFB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, alpha_fwd, alpha_bwd_x, alpha_bwd_w):
        ctx.save_for_backward(x, weight, bias)
        ctx.alpha_bwd_x = alpha_bwd_x
        ctx.alpha_bwd_w = alpha_bwd_w
        return F.linear(x, weight, bias) * alpha_fwd

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, bias = ctx.saved_tensors
        # -------- grads ----------
        grad_x = torch.matmul(grad_out * ctx.alpha_bwd_x, weight)

        go_flat = (grad_out * ctx.alpha_bwd_w).reshape(-1, grad_out.shape[-1])
        x_flat  = x.reshape(-1, x.shape[-1])
        grad_weight = go_flat.t() @ x_flat
        grad_bias   = go_flat.sum(0) if bias is not None else None

        return grad_x, grad_weight, grad_bias, None, None, None

class ScaledLinear(nn.Linear):
    """Linear layer with different forward/backward scalings."""
    def __init__(self, config: NanoConfig, in_features, out_features, bias=False, alpha_fwd=None, alpha_bwd_x=None, alpha_bwd_w=None):
        super().__init__(in_features, out_features, bias)

        if alpha_fwd is None:
            alpha_fwd = 1.0 / math.sqrt(in_features)

        if not config.use_uscaling:
            alpha_fwd, alpha_bwd_x, alpha_bwd_w = 1, 1, 1

        self.register_buffer("alpha_fwd", torch.tensor(float(alpha_fwd)))
        self.register_buffer("alpha_bwd_x", torch.tensor(float(alpha_bwd_x if alpha_bwd_x is not None else alpha_fwd)))
        self.register_buffer("alpha_bwd_w", torch.tensor(float(alpha_bwd_w if alpha_bwd_w is not None else alpha_fwd)))

    def forward(self, x):
        return _ScaledLinearFB.apply(x, self.weight, self.bias, self.alpha_fwd, self.alpha_bwd_x, self.alpha_bwd_w)

def get_model(nconfig):
    match nconfig.model:
        case 'gpt':
            from arch.gpt import GPT
            model = GPT(nconfig)
        case 'dragon':
            from arch.dragon import Dragon
            model = Dragon(nconfig)
            pass
        case 'gated-delta-net':
            from arch.gated_delta_net import GatedDeltaNetModel
            model = GatedDeltaNetModel(nconfig)
        case 'mamba2':
            from arch.mamba2 import Mamba2Model
            model = Mamba2Model(nconfig)
        case _:
            raise ValueError(f"Model {nconfig.model} not supported")
    return model

def param_groups_mup(model, base_lr_hidden, base_lr_scalar, base_lr_embed, base_lr_head, wd):
    groups, seen = [], set()
    id2name = {id(p): n for n, p in model.named_parameters()}

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            pname = id2name.get(id(mod.weight), "")
            fan_in = mod.weight.shape[1]
            scale = 1 / math.sqrt(fan_in)
            if "lm_head" in pname:
                lr_scaled = base_lr_head
            else:
                lr_scaled = base_lr_hidden * scale

            #print(f"{pname} | shape={tuple(mod.weight.shape)} | lr={lr_scaled:.3e}")
            groups.append({"params": [mod.weight], "lr": lr_scaled, "weight_decay": wd/lr_scaled})
            seen.add(mod.weight)

            if mod.bias is not None:
                groups.append({"params": [mod.bias], "lr": lr_scaled, "weight_decay": 0.0})
                seen.add(mod.bias)

    for p in model.parameters():
        if p in seen:
            continue
        pname = id2name.get(id(p), "<unnamed>")

        if pname == "module._orig_mod.transformer.wte.weight":
            fan_out = p.shape[1] # nn.Embedding is transposed
            #lr_scaled = base_lr / math.sqrt(fan_out) # u-muP
            lr_scaled = base_lr_embed
        else:
            lr_scaled = base_lr_scalar

        #print(f"  {pname} | shape={tuple(p.shape)} | lr={lr_scaled:.3e}")
        groups.append({"params": [p], "lr": lr_scaled, "weight_decay": 0.})

    return groups

def param_groups_mup_muon(model, base_lr_hidden, base_lr_scalar, base_lr_embed, base_lr_head, wd):
    groups_adamw, groups_muon, seen = [], [], set()
    id2name = {id(p): n for n, p in model.named_parameters()}

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            pname = id2name.get(id(mod.weight), "")
            fan_in = mod.weight.shape[1]
            scale = 1 / math.sqrt(fan_in)
            if "lm_head" in pname:
                lr_scaled = base_lr_head
            else:
                lr_scaled = base_lr_hidden * scale

            #print(f"{pname} | shape={tuple(mod.weight.shape)} | lr={lr_scaled:.3e}")
            if "lm_head" in pname:
                groups_adamw.append({"params": [mod.weight], "lr": lr_scaled, "weight_decay": 0.}) # wd/lr_scaled
            else:
                groups_muon.append({"params": [mod.weight], "lr": lr_scaled, "weight_decay": wd/lr_scaled})
            seen.add(mod.weight)

            if mod.bias is not None:
                groups_adamw.append({"params": [mod.bias], "lr": lr_scaled, "weight_decay": 0.})
                seen.add(mod.bias)

    for p in model.parameters():
        if p in seen:
            continue
        pname = id2name.get(id(p), "<unnamed>")

        if pname == "module._orig_mod.transformer.wte.weight":
            fan_out = p.shape[1] # nn.Embedding is transposed
            #lr_scaled = base_lr / math.sqrt(fan_out) # u-muP
            lr_scaled = base_lr_embed
        else:
            lr_scaled = base_lr_scalar

        #print(f"  {pname} | shape={tuple(p.shape)} | lr={lr_scaled:.3e}")
        groups_adamw.append({"params": [p], "lr": lr_scaled, "weight_decay": 0.})

    return (groups_adamw, groups_muon)

class StatsCollector:
    def __init__(self, config: NanoConfig):
        self.config = config
        self._buf = collections.defaultdict(lambda: dict(sum=0., sumsq=0., cnt=0, max=float('-inf')))

    def update(self, name: str, t: torch.Tensor):
        if not self.config.track_stats: return
        t = t.float().flatten()
        b = self._buf[name]
        b['sum']   += t.sum().item()
        b['sumsq'] += (t**2).sum().item()
        b['cnt']   += t.numel()
        b['max']    = max(b['max'], t.abs().max().item())
    
    def get(self, name=None, *, reset=False):
        src = self._buf if name is None else {name: self._buf[name]}
        out = {k: dict(
            mean = v['sum']/v['cnt'],
            std  = (v['sumsq']/v['cnt']-(v['sum']/v['cnt'])**2)**0.5,
            max  = v['max']
        ) for k,v in src.items() if v['cnt']}
        if reset:
            for v in src.values():
                v.update(sum=0., sumsq=0., cnt=0, max=float('-inf'))
        return out

    def is_enabled(self):
        return self.config.track_stats

_layer_pat = re.compile(r"\.h\.(\d+)\.")
_stat_pat = re.compile(r"(\.grad\.(?:std|mean|l1)|\.act\.(?:std|mean|l1)|\.(?:std|mean|l1))$")

def l1(x: torch.Tensor) -> float:
    return x.abs().mean().item()

@torch._dynamo.disable
def _capture(name: str, store: Dict[str, torch.Tensor], _m, _inp, out):
    """Save every tensor produced by a module so that we can measure activations."""
    def walk(x, suf=""):
        if torch.is_tensor(x):
            store[f"{name}{suf}"] = x.detach()
        elif isinstance(x, (list, tuple)):
            for i, xi in enumerate(x):
                walk(xi, suf + f"[{i}]")
    walk(out)

def _layer_idx(key: str) -> int:
    """Return integer index of the transformer block found in the key."""
    m = _layer_pat.search(key)
    return int(m.group(1)) if m else -1  # -1 for non‑layer params

def _base_key(key: str) -> str:
    """Return <parameter‑suffix>.<stat> (e.g. attn.k_norm.act.std) to serve as aggregation key."""
    # Slice off the transformer prefix and layer index
    pre_cut = _layer_pat.sub(".", key)  # remove ".h.<idx>." keeping one dot
    # Remove leading "transformer." (if present)
    pre_cut = pre_cut.split("transformer.")[-1]
    # Ensure we still include the stat suffix (.std / .grad.std / .act.std)
    stat_match = _stat_pat.search(pre_cut)
    assert stat_match, f"No stat suffix in key {key}"
    stat_suffix = stat_match.group(1)
    # Remove everything up to the stat suffix and rebuild
    base_no_stat = pre_cut[: -len(stat_suffix)]
    return f"{base_no_stat}{stat_suffix}"

class StatCapture:
    """Context-manager that grabs activations + grads for ONE forward/backward."""
    def __init__(self, model):
        self.model, self.n_layers = model, 20 # model.config.n_layers
        self.acts, self.hooks = {}, []

    def __enter__(self):
        for n, m in self.model.named_modules():
            if m is self.model:               # skip root
                continue
            self.hooks.append(
                m.register_forward_hook(partial(_capture, n, self.acts))
            )
        return self

    def collect(self):
        """After backward(), build the stats dict exactly like your script."""
        raw_stats = {}
        for n, p in self.model.named_parameters():
            raw_stats[f"{n}.std"]     = p.std().item()
            raw_stats[f"{n}.l1"]      = l1(p)
            if p.grad is not None:
                raw_stats[f"{n}.grad.std"] = p.grad.std().item()
                raw_stats[f"{n}.grad.l1"]  = l1(p.grad)
        for n, a in self.acts.items():
            raw_stats[f"{n}.act.std"] = a.std().item()
            raw_stats[f"{n}.act.l1"]  = l1(a)
        self.acts.clear()

        # --- aggregate across layers (unchanged from your script) ---
        agg = defaultdict(lambda: [None] * self.n_layers)
        flat = {}
        for k, v in raw_stats.items():
            layer = _layer_idx(k)
            if layer == -1:
                flat[k] = v
            else:
                agg[_base_key(k)][layer] = v
        merged = {**flat, **agg}

        stats = {}
        pad = len(str(self.n_layers - 1))
        for k, v in merged.items():
            if isinstance(v, list):
                for i, val in enumerate(v):
                    if val is not None:
                        stats[f"inspect/{k}.layer{i:0{pad}}"] = val
            else:
                stats[f"inspect/{k}"] = v
        return stats

    def __exit__(self, exc_type, exc, tb):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
