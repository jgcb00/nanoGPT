# -*- coding: utf-8 -*-

import os

import torch
import triton

import argparse

from mamba_ssm.modules.mamba2 import Mamba2
from fla.layers import GatedDeltaNet

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
torch.empty(1, device='cuda', requires_grad=True).backward()

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[128 * 2 ** i for i in range(4, 6)],
        line_arg='provider',
        line_vals=['mamba2', 'gdn'],
        line_names=['mamba2', 'gdn'],
        styles=[('green', '-'), ('blue', '-')],
        ylabel="Execution Time (ms)",
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
    B, H, D, E = 2, 32, 64, 2
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    x = torch.randn(B, T, D*H, device=device, requires_grad=requires_grad, dtype=dtype)
    do = torch.ones(B, T, D*H, device=device, requires_grad=requires_grad, dtype=dtype)

    mixer = None
    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    match provider:
        case 'mamba2':
            mixer = Mamba2(d_model=D*H, expand=2).to(device)#.to(torch.bfloat16)
        case 'gdn':
            mixer = GatedDeltaNet(hidden_size=D*H, use_gate=True, expand_v=2, num_heads=H, head_dim=int(0.75*D)).to(device)#.to(torch.bfloat16)
        case _:
            raise ValueError(f"Unknown provider: {provider}")

    #mixer = torch.compile(mixer)

    num_params = sum(p.numel() for p in mixer.parameters())
    print(f"number of parameters: {num_params}")

    # a few steps of warmup:
    for _ in range(10):
        with ctx:
            match provider:
                case 'mamba2':
                    mixer(x).backward(do)
                case 'gdn':
                    mixer(x)[0].backward(do)

    with ctx:
        match provider:
            case 'mamba2':
                results = triton.testing.do_bench(lambda: mixer(x).backward(do), quantiles=quantiles)
            case 'gdn':
                results = triton.testing.do_bench(lambda: mixer(x)[0].backward(do), quantiles=quantiles)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flash Attn Benchmark')
    parser.add_argument('--output_dir', type=str, default="mixers")
    args = parser.parse_args()
    
    benchmark.run(print_data=True, save_path=args.output_dir)
