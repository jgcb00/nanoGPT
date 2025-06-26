# -*- coding: utf-8 -*-

import os
import pandas as pd

import torch
import triton

from config import NanoConfig
from arch.mixer.mixer_attention import (
    MixerAttention,
    MixerDiffAttention,
    MixerNativeSparseAttention,
)
from arch.mixer.mixer_gnd import MixerGatedDeltaNet
from arch.mixer.mixer_mamba2 import MixerMamba2
from arch.mlp import MLP
from arch.gpt import Block as BlockGPT
from arch.dragon import Block as BlockDragon

ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
memory_table = []


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=["T"],
        # different possible values for `x_name`
        x_vals=[128 * 2**i for i in range(5, 6)],
        # argument name whose value corresponds to a different line in the plot
        line_arg="provider",
        # possible values for `line_arg``
        line_vals=["block", "mixer_attn", "mixer_lin_attn", "mlp"],
        # label name for the lines
        line_names=["block", "mixer_attn", "mixer_lin_attn", "mlp"],
        # line styles
        styles=[
            ("green", "-"),
            ("blue", "-"),
            ("red", "-"),
            ("green", "dotted"),
            ("blue", "dotted"),
            ("red", "dotted"),
            ("cyan", "-"),
            ("cyan", "dotted"),
        ],
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
    B = 2

    config = NanoConfig()
    config.model = "gpt"
    config.attn_type = "nsa"
    config.lin_attn_type = "gdn"
    config.d_model = 1280
    config.n_heads = 16
    config.n_kv_heads = 1
    config.expand_factor = 1
    config.nsa_topn = 16

    match provider:
        case "block":
            if config.model == "gpt":
                model = BlockGPT(config).to(device).to(dtype)
            elif config.model == "dragon":
                model = BlockDragon(config).to(device).to(dtype)
            else:
                raise NotImplementedError
        case "mixer_attn":
            if config.attn_type == "normal":
                model = MixerAttention(config).to(device).to(dtype)
            elif config.attn_type == "diff":
                model = MixerDiffAttention(config).to(device).to(dtype)
            elif config.attn_type == "nsa":
                model = MixerNativeSparseAttention(config).to(device).to(dtype)
            else:
                raise NotImplementedError
        case "mixer_lin_attn":
            if config.lin_attn_type == "mamba2":
                model = MixerMamba2(config).to(device).to(dtype)
            elif config.lin_attn_type == "gdn":
                model = MixerGatedDeltaNet(config).to(device).to(dtype)
            else:
                raise NotImplementedError
        case "mlp":
            model = MLP(config).to(device).to(dtype)

    model = torch.compile(model, dynamic=False)

    def f(x):
        # print(provider)
        # print(type(model(x)))

        if "mixer" in provider:
            return model(x)[0]
        else:
            return model(x)

    for _ in range(10):
        x = torch.randn(
            B,
            T,
            config.d_model,
            requires_grad=requires_grad,
            dtype=dtype,
            device=device,
        )
        do = torch.ones_like(x, dtype=dtype)
        f(x).backward(do)

    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(
        B, T, config.d_model, requires_grad=requires_grad, dtype=dtype, device=device
    )
    do = torch.ones_like(x, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0

    if not "bwd" in provider:
        with ctx:
            results = triton.testing.do_bench(lambda: f(x), quantiles=quantiles)
    else:
        with ctx:
            results = triton.testing.do_bench(
                lambda: f(x).backward(do), quantiles=quantiles
            )

    memory_usage = torch.cuda.max_memory_allocated(device) / (1024**2)
    memory_table.append((T, provider, memory_usage))
    return results


if __name__ == "__main__":
    save_path = "bench_whole"

    benchmark.run(print_data=True, save_path=save_path)

    os.makedirs(save_path, exist_ok=True)
    memory_file = os.path.join(save_path, "memory_usage.csv")
    with open(memory_file, "w") as f:
        f.write("T,provider,peak_memory_MB\n")
        for T, provider, mem in memory_table:
            f.write(f"{T},{provider},{mem:.2f}\n")

    df = pd.DataFrame(memory_table, columns=["T", "provider", "peak_memory_MB"])
    df_pivot = df.pivot(
        index="T", columns="provider", values="peak_memory_MB"
    ).reset_index()
    desired_order = ["T", "block", "mixer_attn", "mixer_lin_attn", "mlp"]
    df_pivot = df_pivot[[col for col in desired_order if col in df_pivot.columns]]
    print("\nPerformance:")
    print(df_pivot)
