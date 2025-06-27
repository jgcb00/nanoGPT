import os
import json
from dataclasses import fields
from pathlib import Path
import time
from filelock import FileLock

import numpy as np
import torch

from config import NanoConfig
from arch.dragon import Dragon
from arch.mixer.mixer_attention import apply_rotary_emb
from arch.data.distributed_data_loader import DistributedDataLoader

VOCAB_SIZE = 196736

def load_nanogpt_model(checkpoint_path):
    with open(checkpoint_path / "config.json", "r") as f:
        data = json.load(f)
    valid_keys = {f.name for f in fields(NanoConfig)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    if "eval_benchmarks_tasks" in filtered and isinstance(filtered["eval_benchmarks_tasks"], list):
        filtered["eval_benchmarks_tasks"] = ",".join(filtered["eval_benchmarks_tasks"])
    config = NanoConfig(**filtered)

    model_nanogpt = Dragon(config)
    model_nanogpt = model_nanogpt.cuda(torch.cuda.current_device())
    model_nanogpt.eval()

    return model_nanogpt, config

def log_json(payload: dict, step: int, file_path: Path):
    """Thread‑safe append/update of *payload* under *step* inside *file_path*."""
    lock = FileLock(str(file_path) + ".lock")
    with lock:
        existing = json.loads(file_path.read_text()) if file_path.exists() else {}
        existing.setdefault(str(step), {}).update(payload)
        tmp = file_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(existing, indent=2))
        os.replace(tmp, file_path)

checkpoint_path = Path("logs/exp14long_Dragon-L-GDN-adamw_37113992/")
model, config = load_nanogpt_model(checkpoint_path)

model_file = checkpoint_path / "state_step032990.pt"
assert model_file.exists(), f"Model file {model_file} does not exist."
print(f"Loading model from {model_file} ({config.num_iterations}). Using wsize {config.slw_window}.")
checkpoint = torch.load(model_file, weights_only=False, map_location="cuda")
state_dict = checkpoint['model']
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

print("Model loaded successfully.")

# WEIGHTS STATS ----------------------
stats = {}
for name, p in model.named_parameters():
    parts = name.split(".")
    if len(parts) > 3 and parts[0] == "transformer" and parts[1] == "h":
        layer = parts[2]
        tag   = f"weights/layer{layer}_{'_'.join(parts[3:])}"
    else:
        tag   = f"weights/global_{name.replace('.', '_')}"
    stats[tag] = p.detach().abs().max().item()

w_stats_path = checkpoint_path / "weight_stats.json"
log_json(stats, config.num_iterations, w_stats_path)
print(f"Logged {len(stats)} weight metrics to {w_stats_path}")

# ACTIVATIONS STATS ----------------------
B = 2
steps = 1 # 20
loader = DistributedDataLoader('../nanoGPT/data/fineweb100B/fineweb_train_*.bin', B=B, T=config.sequence_length, process_rank=0, num_processes=1)

config.track_stats = True

st = time.time()
device = next(model.parameters()).device

total_loss = 0.0
with torch.no_grad():
    x, y = loader.next_batch()
    total_loss += model(x, targets=y).item()

act_stats = {}
for k, v in model.tracker.get().items():
    act_stats[f"activations/{k}_max"] = v['max']
for i, blk in enumerate(model.transformer.h):
    for tracker in [blk.tracker.get(), blk.attn.tracker.get(), blk.lin_attn.tracker.get()]: 
        for k, v in tracker.items():
            act_stats[f"activations/layer{i}_{k}_max"] = v['max']

# json logging
json_stats_path = checkpoint_path / "activation_stats.json"
log_json(act_stats, config.num_iterations, json_stats_path)

# prints
print(f"Logged {len(act_stats)} activation metrics to {json_stats_path}")
mean_loss = total_loss / steps
print(f"Mean loss over {steps} steps: {mean_loss:.4f} | Elapsed: {time.time() - st:.1f}s")
