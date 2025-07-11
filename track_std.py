from typing import List, Dict
import json
import re
import torch
import torch.nn as nn
from functools import partial
from collections import defaultdict

from config import NanoConfig
from arch.utils import get_model

uscaling = True

B, L = 2, 2048
d_model, d_head, n_layers = 2048, 64, 36
nconfig = NanoConfig(model="dragon", uscaling_tau=0.2, d_model=d_model, n_heads=d_model//d_head, n_layers=n_layers, n_global_layers=3, global_attn_repart="middle", attn_type="diff", lin_attn_type="gdn", vocab_size=50000)

if uscaling:
    nconfig.use_uscaling = True
    nconfig.init_std = 1.0
else:
    nconfig.use_uscaling = False
    nconfig.init_std = 0.006

nconfig.groupnorm = True
nconfig.fused_loss_computation = False

model = get_model(nconfig)
model.to("cuda")

# ---------- helpers ---------- #

def _capture(name: str, store: Dict[str, torch.Tensor], _m, _inp, out):
    """Save every tensor produced by a module so that we can measure activations."""
    def walk(x, suf=""):
        if torch.is_tensor(x):
            store[f"{name}{suf}"] = x.detach()
        elif isinstance(x, (list, tuple)):
            for i, xi in enumerate(x):
                walk(xi, suf + f"[{i}]")
    walk(out)

_layer_pat = re.compile(r"\.h\.(\d+)\.")
_stat_pat = re.compile(r"(\.grad\.(?:std|mean)|\.act\.(?:std|mean)|\.(?:std|mean))$")

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

# ---------- main routine ---------- #

def show_layer_stats(model: nn.Module) -> str:
    """Run a forward/backward pass and return aggregated stats in JSON.

    The JSON schema is:
    {
        "attn.linear_qkv.weight.std": [layer0, layer1, ..., layerN],
        "attn.linear_qkv.grad.std"  : [...],
        "attn.linear_qkv.act.std"   : [...],
        ...
    }
    Layers that do not have a value for a given statistic are represented with null.
    Non‑layer parameters (e.g., embeddings) are kept flat as a single key‑value pair.
    """
    # ----- collect activations ----- #
    acts, hooks = {}, []
    for n, m in model.named_modules():
        if m is model:
            continue  # skip root
        hooks.append(m.register_forward_hook(partial(_capture, n, acts)))

    x = torch.randint(0, nconfig.vocab_size, (B, L), device="cuda")
    y = torch.randint(0, nconfig.vocab_size, (B, L), device="cuda")
    loss = model(x, targets=y)
    loss.backward()

    # ----- collect stats (weight / grad / act) ----- #
    raw_stats = {}
    for n, p in model.named_parameters():
        raw_stats[f"{n}.std"]      = p.std().item()
        raw_stats[f"{n}.grad.std"] = p.grad.std().item()
        #raw_stats[f"{n}.grad.mean"]  = p.grad.mean().item()
    for n, a in acts.items():
        raw_stats[f"{n}.act.std"]  = a.std().item()
        #raw_stats[f"{n}.act.mean"]   = a.mean().item()

    # ----- aggregate across layers ----- #
    agg: Dict[str, List] = defaultdict(lambda: [None] * n_layers)
    flat: Dict[str, float] = {}

    for key, val in raw_stats.items():
        layer = _layer_idx(key)
        if layer == -1:
            # params without layer index stay flat
            flat[key] = val
            continue
        base = _base_key(key)
        if layer < n_layers:
            agg[base][layer] = val
        else:
            # unexpected layer index; fall back to flat
            flat[key] = val

    # ----- merge flat & aggregated ----- #
    merged = {**flat, **agg}

    # Sort keys alphabetically so related entries stay grouped
    ordered = {k: merged[k] for k in sorted(merged)}

    blob = json.dumps(ordered, indent=2)
    for h in hooks:
        h.remove()
    return blob

filename = "layer_stats.json" if uscaling else "layer_stats_baseline.json"

json_blob = show_layer_stats(model)
with open(filename, "w") as f:
    if json_blob:
        f.write(json_blob)
print(f"✅ Saved layer stats to {filename} ✅")
