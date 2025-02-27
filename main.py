import os
import sys
import argparse
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import NanoConfig
from arch.distributed_data_loader import DistributedDataLoader
from arch.muon import Muon
from arch.model import GPT

# TODO:
# wandb/tensorboard???
# check correspondance with megatron : do they have extra hparams ? do we have extra hparams? +SPAM
# implement mamba2
# implement gdn
# for hybrid model, add a way to tell which layers are attn
# add hymba features : local/global attn (why not flex attention?), head regrouping
# next step also will have to do a proper calibration with megatron, ie ensure that results are approx. the same (so need same data)

def parse_args():
    parser = argparse.ArgumentParser()
    parser = NanoConfig.add_args(parser)
    return parser.parse_args()

args = parse_args()
nconfig = NanoConfig.from_args(args)

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
torch._dynamo.config.optimize_ddp=False

# convenience variables
B, T = nconfig.device_batch_size, nconfig.sequence_length
# calculate the number of steps to take in the val loop.
assert nconfig.val_tokens % (B * T * ddp_world_size) == 0
val_steps = nconfig.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert nconfig.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = nconfig.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(nconfig.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(nconfig.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(nconfig)
model = model.cuda()
model = torch.compile(model, dynamic=False)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
if nconfig.optim == 'adamw':
    optimizer = torch.optim.Adam(model.parameters(), lr=nconfig.learning_rate, betas=(0.9, 0.95), weight_decay=nconfig.weight_decay)
    optimizers = [optimizer]
elif nconfig.optim == 'muon':
    optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=0.3, betas=(0.9, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=0.002, betas=(0.9, 0.95), fused=True)
    optimizer3 = Muon(raw_model.transformer.h.parameters(), lr=nconfig.learning_rate, momentum=0.95)
    optimizers = [optimizer1, optimizer2, optimizer3]
else:
    raise ValueError(f"Optimizer {nconfig.optim} not supported")

# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
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
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()
for step in range(nconfig.num_iterations + 1):
    last_step = (step == nconfig.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (nconfig.val_loss_every > 0 and step % nconfig.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{nconfig.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{nconfig.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (nconfig.save_every > 0 and step % nconfig.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{nconfig.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{nconfig.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
