from pathlib import Path
from dataclasses import dataclass
import pickle
import tyro
import json
from dataclasses import fields

import torch

from config import NanoConfig
from arch.utils import get_model


@dataclass
class Args:
    run_dir: Path  # something like logs/... (the dir that contains the .pt model)

    def __post_init__(self):
        assert self.run_dir.exists(), f"Run directory {self.run_dir} does not exist."


args = tyro.cli(Args)

# read config
# with open(args.run_dir / 'config.pkl', 'rb') as f:
#    config: NanoConfig = pickle.load(f)
# config.rmsnorm = False
# config.disable_scalable_softmax_for_local = True # False for loading old runs, True for newer ones
# config.use_patch_level_training = False
# config.fused_loss_computation = False
with open(Path(args.run_dir) / "config.json", "r") as f:
    data = json.load(f)
valid_keys = {f.name for f in fields(NanoConfig)}
filtered = {k: v for k, v in data.items() if k in valid_keys}
if "eval_benchmarks_tasks" in filtered and isinstance(
    filtered["eval_benchmarks_tasks"], list
):
    filtered["eval_benchmarks_tasks"] = ",".join(filtered["eval_benchmarks_tasks"])
config = NanoConfig(**filtered)

# define and load model, tokenizer
model = get_model(config)
model = model.to(torch.bfloat16)
model = torch.compile(model, dynamic=False)
model = model.cuda()

model_file = sorted(args.run_dir.glob("state_step*.pt"))[-1]
assert model_file.exists(), f"Model file {model_file} does not exist."
print(f"Loading model from {model_file}.")

checkpoint = torch.load(model_file)

model.load_state_dict(checkpoint["model"])

print(model)
