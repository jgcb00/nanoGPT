# -*- coding: utf-8 -*-

import os
import pandas as pd

import torch
import triton

from config import NanoConfig
from arch.gpt import Block

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
memory_table = []

@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(5, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['nsa', 'nsa_bwd', 'flash', 'flash_bwd'],
        # label name for the lines
        line_names=['nsa', 'nsa_bwd', 'flash', 'flash_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('green', 'dotted'),
                ('blue', 'dotted'), ('red', 'dotted'), ('cyan', '-'), ('cyan', 'dotted')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(T, provider):
    print(f"Running benchmark with T={T}, provider={provider}")
    torch.compiler.reset()

    device = "cuda"
    dtype = torch.bfloat16
    requires_grad = True
    B, HQ, H, D = 2, 16, 1, 80
    kernel_size, kernel_stride, block_size, topn, swa = 32, 16, 64, 16, 512

    config = NanoConfig()
    config.d_model = HQ*D
    config.n_heads = HQ
    config.n_kv_heads = H
    config.nsa_block_size = block_size
    config.nsa_kernel_size = kernel_size
    config.nsa_kernel_stride = kernel_stride
    config.nsa_topn = topn
    config.nsa_swa = swa

    if "nsa" in provider:
        config.attn_type = "nsa"

    mixer = Block(config).to(device).to(dtype)
    mixer = torch.compile(mixer, dynamic=False)
    for _ in range(10):
        x = torch.randn(B, T, HQ*D, requires_grad=requires_grad, dtype=dtype, device=device)
        do = torch.ones_like(x, dtype=dtype)
        mixer(x).backward(do)

    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(B, T, HQ*D, requires_grad=requires_grad, dtype=dtype, device=device)
    do = torch.ones_like(x, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0

    if not 'bwd' in provider:
        with ctx:
            results = triton.testing.do_bench(lambda: mixer(x), quantiles=quantiles)
    else:
        with ctx:
            results = triton.testing.do_bench(lambda: mixer(x).backward(do), quantiles=quantiles)

    memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    memory_table.append((T, provider, memory_usage))
    
    return results

if __name__ == '__main__':
    save_path = 'bench_nsa'

    benchmark.run(print_data=True, save_path=save_path)

    os.makedirs(save_path, exist_ok=True)
    memory_file = os.path.join(save_path, 'memory_usage.csv')
    with open(memory_file, 'w') as f:
        f.write('T,provider,peak_memory_MB\n')
        for T, provider, mem in memory_table:
            f.write(f"{T},{provider},{mem:.2f}\n")

    df = pd.DataFrame(memory_table, columns=['T', 'provider', 'peak_memory_MB'])
    df_pivot = df.pivot(index='T', columns='provider', values='peak_memory_MB').reset_index()
    desired_order = ['T', 'nsa', 'nsa_bwd', 'flash', 'flash_bwd']
    df_pivot = df_pivot[[col for col in desired_order if col in df_pivot.columns]]
    print("\nPerformance:")
    print(df_pivot)
