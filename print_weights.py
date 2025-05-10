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
from config import NanoConfig

from arch.lm import NanoLM

if __name__ == "__main__":
    @dataclass
    class Args:
        run_dir: Path # something like logs/... (the dir that contains the .pt model)

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

    # BLOCK/MIXER NORMS
    print(" ---------- BLOCK NORMS ----------")
    for i in range(len(model.transformer.h)):
        print(model.transformer.h[i].input_norm.weight)
        print(model.transformer.h[i].postmixer_norm.weight)
        print(model.transformer.h[i].attn.group_norm.weight)
        print(model.transformer.h[i].lin_attn.group_norm.weight)

    # A_log, dt_bias
    print(" ---------- A_log, dt_bias ----------")
    for i in range(len(model.transformer.h)):
        print(model.transformer.h[i].lin_attn.A_log)
        print(model.transformer.h[i].lin_attn.dt_bias)
    
    print(" ---------- LAYER 3 ----------")
    print(model.transformer.h[3].attn.softmax_scaler)
    print(model.transformer.h[3].attn.lambda_q1)
    print(model.transformer.h[3].attn.lambda_k1)
    print(model.transformer.h[3].attn.lambda_q2)
    print(model.transformer.h[3].attn.lambda_k2)
    
    print(" ---------- LAYER 10 ----------")
    print(model.transformer.h[10].attn.softmax_scaler)
    print(model.transformer.h[10].attn.lambda_q1)
    print(model.transformer.h[10].attn.lambda_k1)
    print(model.transformer.h[10].attn.lambda_q2)
    print(model.transformer.h[10].attn.lambda_k2)

    print(" ---------- LAYER 17 ----------")
    print(model.transformer.h[17].attn.softmax_scaler)
    print(model.transformer.h[17].attn.lambda_q1)
    print(model.transformer.h[17].attn.lambda_k1)
    print(model.transformer.h[17].attn.lambda_q2)
    print(model.transformer.h[17].attn.lambda_k2)