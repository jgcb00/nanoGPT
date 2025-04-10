import sys
import time
st = time.time()

import random
random_sleep = random.random() * 5
time.sleep(random_sleep)

import os
import tyro
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
"""
from arch.utils import get_model
from config import NanoConfig
from arch.data.distributed_data_loader import DistributedDataLoader
from arch.optim.get_optimizer import get_optimizers
from arch.schedulers import get_schedulers


nconfig = tyro.cli(NanoConfig)
assert nconfig.run_name != "", "Please provide a run name for this training run."
"""

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(
    backend='nccl',
    init_method='env://',
    world_size=int(os.environ['WORLD_SIZE']),
    rank=int(os.environ['RANK']),
)
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
torch._dynamo.config.optimize_ddp=False

et = time.time()
runtime = et - st
print(f"Runtime: {runtime}")
sys.exit()

def print0(s, console=True):
    if master_process:    
        if console:
            print(s)
# log current code + versions + config
print0('='*100, console=False)
print0(f"Running Python {sys.version}")
print0(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi(), console=False)
print0("="*100)
print0(nconfig)
print0("="*100)

# convenience variables
B, T = nconfig.device_batch_size, nconfig.sequence_length
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert nconfig.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = nconfig.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(nconfig.input_bin, B, T, ddp_rank, ddp_world_size, score_pattern=nconfig.scoring_bin)
print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
x, y, scores = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = get_model(nconfig)
#count parameters
num_params = sum(p.numel() for p in model.parameters())
print0(f"number of parameters: {num_params}")
nconfig.num_params = num_params
model = model.to(torch.bfloat16)
model = torch.compile(model, dynamic=False)
model = model.cuda()

print0(model)

# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
# init the optimizer(s)
optimizers = get_optimizers(model, nconfig, raw_model)
schedulers = get_schedulers(optimizers, nconfig)

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
reset_step = 10 # step at which we reset the timer
# begin training
if nconfig.is_scorer:
    tokens_per_shard = 100e6
    tokens_to_skip = 10e9
    train_loader.reset(shard=math.ceil(tokens_to_skip/tokens_per_shard)+1)
else:
    train_loader.reset()

if nconfig.setup_only:
    et = time.time()
    runtime = et - st
    print0(f"Runtime: {runtime}")
    dist.destroy_process_group()
    sys.exit()
else:
    st = time.time()

for step in range(nconfig.num_iterations + 1):
    last_step = (step == nconfig.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == reset_step:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step-reset_step <= 11 else (step - reset_step) + 1 # <= to avoid bug in val

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            loss = model(x, targets=y, scores=scores)
            train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y, scores = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # clip those gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=nconfig.grad_norm_clip, foreach=True)
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    approx_time = training_time_ms + 1000 * (time.time() - t0)
    avg_step_time = approx_time / timed_steps
    print0(f"step:{step+1}/{nconfig.num_iterations} train_loss:{train_loss.item():.4f} lr: {schedulers[0].get_last_lr()[0]:.4f} slw_window: {nconfig.slw_window} current_step_time:{approx_time:.0f}ms step_avg:{avg_step_time:.2f}ms")

print0(f"peak memory consumption during training: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
print0("Training complete.")

et = time.time()
runtime = et - st
print0(f"Runtime: {runtime}")

dist.destroy_process_group()
