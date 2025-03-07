import os
from dataclasses import dataclass
from typing import List
import tyro
import pickle
import json
import tiktoken

import torch

import lm_eval

from arch.gpt import GPT
from arch.lm import GPT_LM
from config import NanoConfig

@dataclass
class Args:
    run_dir: str # something like logs/... (the dir that contains the .pt model)
    tasks: str # list of tasks to evaluate on (hellaswag, winogrande, ...)
    batch_size: int = 32

    def __post_init__(self):
        self.tasks = self.tasks.split(',')

args = tyro.cli(Args)

# read config
with open(os.path.join(args.run_dir, 'config.pkl'), 'rb') as f:
    config = pickle.load(f)

# define and load model, tokenizer and encapsulate in LM object
model = GPT(config)
model.cuda()
#model = torch.compile(model, dynamic=False)
model.load_state_dict(torch.load(os.path.join(args.run_dir + 'model.pt')))

with open('data/enc.pkl', 'rb') as f:
    enc_pickled = pickle.load(f)
enc = tiktoken.core.Encoding(enc_pickled.pop('name'), **enc_pickled)

lm = GPT_LM(model, enc, batch_size=args.batch_size)

# evaluate
results = lm_eval.simple_evaluate(lm, tasks=args.tasks)

# save results (with the names of the tasks in the file)
with open(os.path.join(args.run_dir, 'results_' + '_'.join(args.tasks) + '.json'), 'w') as f:
    json.dump(results, f)