from functools import partial
import numpy as np
import torch
from config import NanoConfig
def get_lr_wsd(nconfig : NanoConfig, it):
    assert it <= nconfig.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < nconfig.warmup_iters:
        return (it+1) / nconfig.warmup_iters
    # 2) constant lr for a while
    elif it < nconfig.num_iterations - nconfig.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (nconfig.num_iterations - it) / nconfig.warmdown_iters
        return decay_ratio
    
def get_lr_moonlight(nconfig : NanoConfig, it):
    assert it <= nconfig.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < nconfig.warmup_iters:
        return (it+1) / nconfig.warmup_iters
    # 2) cosine decay for a while:
    elif it < nconfig.num_iterations - nconfig.warmdown_iters:
        return 0.1+0.45 * (1 + np.cos((it - nconfig.warmup_iters) / (nconfig.num_iterations - nconfig.warmup_iters - nconfig.warmdown_iters) * np.pi))
    # 3) linear warmdown
    else:
        # warmup to 2 times, the min of cosine decay, and then linearly decay to 0
        if it < nconfig.num_iterations - nconfig.warmdown_iters + 50:
            warm_up_it = it - (nconfig.num_iterations - nconfig.warmdown_iters)
            return 0.2 * (warm_up_it + 1) / 50  
        decay_ratio = 0.2 * (nconfig.num_iterations - it) / (nconfig.warmdown_iters - 50)
        return decay_ratio
    
def get_linear_slow(nconfig : NanoConfig, it):
    assert it <= nconfig.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < nconfig.warmup_iters:
        return (it+1) / nconfig.warmup_iters
    # 2) linear decay for a while from 1 to 0.5 :
    elif it < nconfig.num_iterations - nconfig.warmdown_iters:
        return 1 - (0.5 * it / (nconfig.num_iterations - nconfig.warmup_iters - nconfig.warmdown_iters))
    # 3) linear warmdown
    else:
        # warmup to 2 times, the min of cosine decay, and then linearly decay to 0
        #if it < nconfig.num_iterations - nconfig.warmdown_iters + 50:
        #    warm_up_it = it - (nconfig.num_iterations - nconfig.warmdown_iters)
        #    return 0.2 * (warm_up_it + 1) / 50  
        decay_ratio = 0.5 * (nconfig.num_iterations - it) / (nconfig.warmdown_iters)
        return decay_ratio


def get_schedulers(optimizers, nconfig : NanoConfig):
    match nconfig.optim:
        case 'moonlight':
            func = get_lr_moonlight
        case 'wsd':
            func = get_lr_wsd
        case 'linear-slow':
            func = get_linear_slow
        case _: 
            raise ValueError(f"Scheduler {nconfig.scheduler} not supported")        
    func = partial(func, nconfig)
    return [torch.optim.lr_scheduler.LambdaLR(opt, func) for opt in optimizers]
