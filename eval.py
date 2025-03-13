import os
from dataclasses import dataclass
from typing import List
import tyro
import pickle
import json
import tiktoken
from pathlib import Path
from arch.utils import get_model
import torch
import lm_eval
from config import NanoConfig

from arch.lm import NanoLM

"""
USED FOR EVALUATING ALREADY, OLD, TRAINED MODELS
WILL BE DELETED, AS THIS CODE IS ALSO PRESENT AFTER THE TRAINING LOOP IN THE MAIN SCRIPT
"""

@dataclass
class Args:
    run_dir: Path  # something like logs/... (the dir that contains the .pt model)
    tasks: str  # list of tasks to evaluate on (hellaswag, winogrande, ...)

    def __post_init__(self):
        self.tasks = self.tasks.split(',')
        assert self.run_dir.exists(), f"Run directory {self.run_dir} does not exist."

args = tyro.cli(Args)

# read config
with open(args.run_dir / 'config.pkl', 'rb') as f:
    config: NanoConfig = pickle.load(f)
config.rmsnorm = False
config.disable_scalable_softmax_for_local = False # False for loading old runs, True for newer ones

# define and load model, tokenizer
model = get_model(config)
model.cuda()

model_file = sorted(args.run_dir.glob("state_step*.pt"))[-1]
assert model_file.exists(), f"Model file {model_file} does not exist."

checkpoint = torch.load(model_file)
state_dict = checkpoint['model']

new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

with open('data/enc.pkl', 'rb') as f:
    enc_pickled = pickle.load(f)
enc = tiktoken.core.Encoding(enc_pickled.pop('name'), **enc_pickled)

lm = NanoLM(
    model=model, 
    config=config, 
    enc=enc, 
)

print(f"Evaluating on tasks: {args.tasks} with 1GPUs")

# evaluate
results = lm_eval.simple_evaluate(lm, tasks=args.tasks, limit=0.1)

# export all the results to a json file (to see the completions)
result_file_path = args.run_dir / f"results_{'_'.join(args.tasks)}.json"
with open(result_file_path, 'w') as f:
    json.dump(results, f, indent=4)

# save the scores in a separate file
result_file_path = args.run_dir / f"scores_{'_'.join(args.tasks)}.json"
with open(result_file_path, 'w') as f:
    json.dump(results['results'], f, indent=4)
print("Done evaluating.")
