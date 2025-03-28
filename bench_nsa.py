# -*- coding: utf-8 -*-

import torch
import triton

from config import NanoConfig
from arch.mixer.mixer_attention import MixerAttention, MixerNativeSparseAttention

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
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
    device = "cuda"
    dtype = torch.bfloat16
    requires_grad = True
    B, HQ, H, D = 4, 16, 1, 48
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

    x = torch.randn(B, T, HQ*D, requires_grad=True, dtype=dtype, device=device)
    do = torch.ones_like(x, dtype=dtype)

    # todo:
    # compile
    # warmup

    if 'nsa' in provider:
        mixer = MixerNativeSparseAttention(config).to(device).to(dtype)
    elif 'flash' in provider:
        mixer = MixerAttention(config).to(device).to(dtype)
    else:
        raise NotImplementedError

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0

    if not 'bwd' in provider:
        with ctx:
            results = triton.testing.do_bench(lambda: mixer(x), quantiles=quantiles)
    else:
        with ctx:
            results = triton.testing.do_bench(lambda: mixer(x)[0].backward(do), quantiles=quantiles)
    return results

if __name__ == '__main__':
    benchmark.run(print_data=True, save_path='bench_nsa')
