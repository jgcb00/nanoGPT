import collections
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NanoConfig

# old, kept for evaluating old runs
"""
class HeadWiseRMSNorm(nn.Module):
    def __init__(self, n_heads, d_head, eps=1e-5):
        super().__init__()
        self.eps = eps
        # poids distinct par tête
        self.weight = nn.Parameter(torch.ones(n_heads, d_head))

    def forward(self, x):
        # x: (B, T, H, D)
        var = x.pow(2).mean(dim=-1, keepdim=True)               # (B, T, H, 1)
        x_norm = x * torch.rsqrt(var + self.eps)               # normalisation RMS
        return x_norm * self.weight.unsqueeze(0).unsqueeze(0)  # (1,1,H,D) → broadcast
"""
class HeadWiseRMSNorm(nn.Module):
    def __init__(self, n_heads, d_head, eps=1e-5):
        super().__init__()
        self.rms = nn.RMSNorm(d_head, eps=eps, elementwise_affine=False)
        self.weight = nn.Parameter(torch.ones(n_heads, d_head))

    def forward(self, x):
        B, L, H, D = x.shape
        y = self.rms(x) * self.weight.view(1, 1, H, D)
        return y.view(B, L, H, D)
    
class ScaledLinear(nn.Linear):
    def __init__(self, config: NanoConfig, in_features, out_features, bias=False, alpha=None):
        super().__init__(in_features, out_features, bias)
        if config.use_uscaling:
            self.register_buffer("alpha", torch.tensor(1.0 / math.sqrt(in_features)) if alpha is None else torch.tensor(alpha))
        else:
            self.register_buffer("alpha", torch.tensor(1.0))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) * self.alpha

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

def param_groups_mup(model, base_lr, wd):
    groups, seen = [], set()
    id2name = {id(p): n for n, p in model.named_parameters()}

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            pname = id2name.get(id(mod.weight), "")
            fan_in = mod.weight.shape[1]
            scale  = 1 / math.sqrt(fan_in)
            if "lm_head" in pname:
                lr_scaled = base_lr
                print(f"[No Scale] Linear: {pname}  | shape={tuple(mod.weight.shape)}  | lr={lr_scaled:.3e}")
            else:
                lr_scaled = base_lr * scale
                print(f"[Scaled]   Linear: {pname}  | shape={tuple(mod.weight.shape)}  | lr={lr_scaled:.3e}")

            print(f"Linear: {id2name.get(id(mod.weight), '<unnamed>')}  | shape={tuple(mod.weight.shape)}  | lr={lr_scaled:.3e}")
            groups.append({
                "params": [mod.weight],
                "lr": lr_scaled,
                "weight_decay": wd
            })
            seen.add(mod.weight)

            if mod.bias is not None:
                print(f"Bias:   {id2name.get(id(mod.bias), '<unnamed>')}  | shape={tuple(mod.bias.shape)}  | lr={lr_scaled:.3e}")
                groups.append({
                    "params": [mod.bias],
                    "lr": lr_scaled,
                    "weight_decay": 0.0
                })
                seen.add(mod.bias)

    rest = [p for p in model.parameters() if p not in seen]
    if rest:
        print(f"Other params (no fan-in scaling): {len(rest)} tensors")
        for p in rest:
            print(f"  {id2name.get(id(p), '<unnamed>')}  | shape={tuple(p.shape)}  | lr={base_lr:.3e}")
        groups.append({"params": rest, "lr": base_lr, "weight_decay": wd})

    return groups

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