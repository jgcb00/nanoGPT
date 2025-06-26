from functools import partial
import numpy as np
import torch
from config import NanoConfig


def get_lr_wsd(num_iterations, warmup_iters, warmdown_iters, offset, it):
    it = it - offset
    assert (
        it <= num_iterations
    ), f"it : {it}, offset : {offset}, num_iterations : {num_iterations}"
    # 1) linear warmup for warmup_iters steps
    if warmup_iters > 0 and it < warmup_iters:
        return (it + 1) / warmup_iters
    # 2) constant lr for a while
    elif it < num_iterations - warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (num_iterations - it) / warmdown_iters
        return decay_ratio


def get_lr_moonlight(num_iterations, warmup_iters, warmdown_iters, offset, it):
    it = it - offset
    assert it <= num_iterations
    # 1) linear warmup for warmup_iters steps
    if warmup_iters > 0 and it < warmup_iters:
        return (it + 1) / warmup_iters
    # 2) cosine decay for a while:
    elif it < num_iterations - warmdown_iters:
        return 0.1 + 0.45 * (
            1
            + np.cos(
                (it - warmup_iters)
                / (num_iterations - warmup_iters - warmdown_iters)
                * np.pi
            )
        )
    # 3) linear warmdown
    else:
        # warmup to 2 times, the min of cosine decay, and then linearly decay to 0
        if it < num_iterations - warmdown_iters + 50:
            warm_up_it = it - (num_iterations - warmdown_iters)
            return 0.2 * (warm_up_it + 1) / 50
        decay_ratio = 0.2 * (num_iterations - it) / (warmdown_iters - 50)
        return decay_ratio


def get_linear_slow(num_iterations, warmup_iters, warmdown_iters, offset, it):
    it = it - offset
    assert it <= num_iterations
    # 1) linear warmup for warmup_iters steps
    if warmup_iters > 0 and it < warmup_iters:
        return (it + 1) / warmup_iters
    # 2) linear decay for a while from 1 to 0.5 :
    elif it < num_iterations - warmdown_iters:
        return 1 - (1 / 5 * it / (num_iterations - warmup_iters - warmdown_iters))
    # 3) linear warmdown
    else:
        # warmup to 2 times, the min of cosine decay, and then linearly decay to 0
        # if it < nconfig.num_iterations - nconfig.warmdown_iters + 50:
        #    warm_up_it = it - (nconfig.num_iterations - nconfig.warmdown_iters)
        #    return 0.2 * (warm_up_it + 1) / 50
        decay_ratio = 0.8 * (num_iterations - it) / (warmdown_iters)
        return decay_ratio


def get_schedulers(optimizers, nconfig: NanoConfig, out_of_patch_level=False):
    warmup_iters = int(nconfig.warmup_iters * nconfig.num_iterations)
    warmdown_iters = int(nconfig.warmdown_iters * nconfig.num_iterations)
    total_iters = nconfig.num_iterations
    offset = 0
    if nconfig.use_patch_level_training:
        warmdown_iters = int(warmdown_iters * nconfig.patch_training_fraction)
        total_iters = int(total_iters * nconfig.patch_training_fraction) + 2
    if out_of_patch_level:
        warmdown_iters = int(warmdown_iters * (1 - nconfig.patch_training_fraction))
        total_iters = int(total_iters * (1 - nconfig.patch_training_fraction)) + 2
        # offset = int(nconfig.num_iterations * nconfig.patch_training_fraction)
    match nconfig.scheduler:
        case "moonlight":
            func = get_lr_moonlight
        case "wsd":
            func = get_lr_wsd
        case "linear-slow":
            func = get_linear_slow
        case _:
            raise ValueError(f"Scheduler {nconfig.scheduler} not supported")
    func = partial(func, total_iters, warmup_iters, warmdown_iters, offset)
    return [torch.optim.lr_scheduler.LambdaLR(opt, func) for opt in optimizers]
