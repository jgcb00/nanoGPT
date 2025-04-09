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

def eval_benchmarks(log_dir, model, tokenizer_path, limit=None, tasks=None):
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
      
    if tasks is None:
        tasks = model.config.eval_benchmarks_tasks

    # load tokenizer
    with open(tokenizer_path, 'rb') as f:
        enc_pickled = pickle.load(f)
    enc = tiktoken.core.Encoding(enc_pickled.pop('name'), **enc_pickled)
    
    # wrap model in a LM object
    lm = NanoLM(
        model=model,
        enc=enc,
    )

    # evaluate
    results = lm_eval.simple_evaluate(lm, tasks=tasks, limit=limit)

    # export all the results to a json file (to see the completions)
    result_file_path = log_dir / f"results_{'_'.join(tasks)}.json"
    with open(result_file_path, 'w') as f:
        json.dump(results, f, indent=4)

    # save the scores in a separate file, also tell in the file if limit is not None
    result_file_path = log_dir / f"scores_{'_'.join(tasks)}"
    if limit is not None:
        result_file_path = result_file_path.with_suffix(f".limit{limit}.json")
    else:
        result_file_path = result_file_path.with_suffix('.json')
    with open(result_file_path, 'w') as f:
        json.dump(results['results'], f, indent=4)

    return results

if __name__ == "__main__":
    @dataclass
    class Args:
        run_dir: Path # something like logs/... (the dir that contains the .pt model)
        tasks: str # list of tasks to evaluate on (hellaswag, winogrande, ...)

        def __post_init__(self):
            self.tasks = self.tasks.split(',')
            assert self.run_dir.exists(), f"Run directory {self.run_dir} does not exist."

    args = tyro.cli(Args)

    # read config
    with open(args.run_dir / 'config.pkl', 'rb') as f:
        config: NanoConfig = pickle.load(f)
    config.rmsnorm = False
    config.disable_scalable_softmax_for_local = True # False for loading old runs, True for newer ones
    config.use_patch_level_training = False

    # define and load model, tokenizer
    model = get_model(config)
    model.to(torch.bfloat16)
    model.cuda()

    model_file = sorted(args.run_dir.glob("state_step*.pt"))[-1]
    assert model_file.exists(), f"Model file {model_file} does not exist."
    print(f"Loading model from {model_file}.")

    checkpoint = torch.load(model_file)
    state_dict = checkpoint['model']

    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    _ = eval_benchmarks(args.run_dir, model, 'data/enc.pkl', tasks=args.tasks)

    print(f"Done evaluating {args.run_dir} on {args.tasks}.")
