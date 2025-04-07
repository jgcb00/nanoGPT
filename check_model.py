from pathlib import Path
from dataclasses import dataclass
import pickle
import tyro

import torch

from config import NanoConfig
from arch.utils import get_model

@dataclass
class Args:
    run_dir: Path # something like logs/... (the dir that contains the .pt model)

    def __post_init__(self):
        assert self.run_dir.exists(), f"Run directory {self.run_dir} does not exist."

args = tyro.cli(Args)

# read config
with open(args.run_dir / 'config.pkl', 'rb') as f:
    config: NanoConfig = pickle.load(f)
config.rmsnorm = False
config.disable_scalable_softmax_for_local = True # False for loading old runs, True for newer ones
config.use_patch_level_training = False
config.fused_loss_computation = False

# define and load model, tokenizer
model = get_model(config)
model = model.to(torch.bfloat16)
model = torch.compile(model, dynamic=False)
model = model.cuda()

model_file = sorted(args.run_dir.glob("state_step*.pt"))[-1]
assert model_file.exists(), f"Model file {model_file} does not exist."
print(f"Loading model from {model_file}.")

checkpoint = torch.load(model_file)

model.load_state_dict(checkpoint['model'])

print(model)