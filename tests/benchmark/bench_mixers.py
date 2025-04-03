# -*- coding: utf-8 -*-

import os

import torch
import triton

import argparse

from config import NanoConfig
from arch.mixer.mixer_attention import MixerAttention, MixerDiffAttention
from arch.mixer.mixer_mamba2 import MixerMamba2
from arch.mixer.mixer_gnd import MixerGatedDeltaNet

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
torch.empty(1, device='cuda', requires_grad=True).backward()

@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(5, 6)],
        #x_vals=[8192],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        #line_vals=['mixer_attn', 'mixer_attn_swa', 'mixer_diff', 'mixer_diff_swa', 'mixer_mamba2', 'mixer_gdn'],
        #line_names=['mixer_attn', 'mixer_attn_swa', 'mixer_diff', 'mixer_diff_swa', 'mixer_mamba2', 'mixer_gdn'],
        #styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('orange', '-'), ('purple', '-'), ('black', '-')],
        line_vals=['mixer_mamba2', 'mixer_gdn'],
        line_names=['mixer_mamba2', 'mixer_gdn'],
        styles=[('green', '-'), ('blue', '-')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)

def benchmark(T, provider):
    print(f"Running benchmark with T={T}, provider={provider}")
    torch.compiler.reset()

    device = 'cuda'
    dtype = torch.bfloat16
    requires_grad = True
    B, H, D = 2, 20, 64
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    config = NanoConfig()
    config.d_model = D*H
    config.n_heads = H
    config.n_kv_heads = H//2
    config.expand_factor = 2
    config.use_gate = True

    if 'swa' in provider:
        swa = True
    else:
        swa = False

    x = torch.randn(B, T, D*H, device=device, requires_grad=requires_grad, dtype=dtype)
    do = torch.ones(B, T, D*H*config.expand_factor, device=device, requires_grad=requires_grad, dtype=dtype)

    mixer = None
    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    match provider:
        case 'mixer_attn' | 'mixer_attn_swa':
            mixer = MixerAttention(config, swa=swa).to(device).to(torch.bfloat16)
        case 'mixer_diff' | 'mixer_diff_swa':
            mixer = MixerDiffAttention(config, swa=swa).to(device).to(torch.bfloat16)
        case 'mixer_mamba2':
            mixer = MixerMamba2(config).to(device).to(torch.bfloat16)
        case 'mixer_gdn':
            mixer = MixerGatedDeltaNet(config).to(device).to(torch.bfloat16)
        case _:
            raise ValueError(f"Unknown provider: {provider}")

    mixer = torch.compile(mixer)

    num_params = sum(p.numel() for p in mixer.parameters())
    print(f"number of parameters: {num_params}")

    # a few steps of warmup:
    for _ in range(10):
        with ctx:
            mixer(x)[0].backward(do)

    with ctx:
        results = triton.testing.do_bench(lambda: mixer(x)[0].backward(do), quantiles=quantiles)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flash Attn Benchmark')
    parser.add_argument('--output_dir', type=str, default="mixers")
    args = parser.parse_args()
    
    benchmark.run(print_data=True, save_path=args.output_dir)
