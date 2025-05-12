import itertools
import os
import sys
import tyro
import argparse
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
import json
import pickle
import wandb

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.attention.flex_attention import create_block_mask
from arch.utils import get_model
from config import NanoConfig
from arch.data.distributed_data_loader import DistributedDataLoader
from arch.optim.get_optimizer import get_optimizers
from arch.schedulers import get_schedulers
# TODO:
# check correspondance with megatron : do they have extra hparams ? do we have extra hparams?
# next step also will have to do a proper calibration with megatron, ie ensure that results are approx. the same (so need same data)
# learning rate decay scheduler (linear warmup and warmdown)

nconfig = tyro.cli(NanoConfig)
assert nconfig.run_name != "", "Please provide a run name for this training run."

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

logfile = None
if master_process:
    run_id = nconfig.run_name + '_' + str(uuid.uuid4().hex[:8])
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    with open(f'{logdir}/config.json', 'w') as f:
        json.dump(vars(nconfig), f)
    with open(f'{logdir}/config.pkl', 'wb') as f:
        pickle.dump(nconfig, f)
    logfile = 'logs/%s.txt' % run_id
    print(logfile)
def print0(s, console=True):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)
# log current code + versions + config
print0(code, console=False)
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

if nconfig.use_patch_level_training:
    prev_device_batch_size = nconfig.device_batch_size
    prev_train_accumulation_steps = nconfig.batch_size // (prev_device_batch_size * ddp_world_size)
    nconfig.device_batch_size = min(nconfig.patch_size, prev_train_accumulation_steps) * prev_device_batch_size
    print0(f"Using patch-level training. Modifying the device batch size to account for the patch size, from {prev_device_batch_size} to {nconfig.device_batch_size}.")

# convenience variables
B, T = nconfig.device_batch_size, nconfig.sequence_length
# calculate the number of steps to take in the val loop.
assert nconfig.val_tokens % (B * T * ddp_world_size) == 0
val_steps = nconfig.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert nconfig.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = nconfig.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(nconfig.input_bin, B, T, ddp_rank, ddp_world_size, score_pattern=nconfig.scoring_bin)
val_loader = DistributedDataLoader(nconfig.input_val_bin, B, T, ddp_rank, ddp_world_size)
print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y, scores = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = get_model(nconfig)
#count parameters
num_params = sum(p.numel() for p in model.parameters())
print0(f"number of parameters: {num_params}")
nconfig.num_params = num_params
#model = model.to(torch.bfloat16)
# only cast specific weights to bfloat16
with torch.no_grad():
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight.data = module.weight.data.to(torch.bfloat16)
        if isinstance(module, torch.nn.Linear):
            module.weight.data = module.weight.data.to(torch.bfloat16)
model = torch.compile(model, dynamic=nconfig.slw_warmup_iters > 0)
model = model.cuda()

print0(model)
for name, p in model.named_parameters():
    print0(f"{name}: {p.shape}")

# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
# init the optimizer(s)
optimizers = get_optimizers(model, nconfig, raw_model)
schedulers = get_schedulers(optimizers, nconfig)

# begin wandb logging
if master_process:
    if "longcrawl64" in nconfig.input_bin:
        project = 'nanoGPT-longcrawl64'
    else:
        project = 'nanoGPT'
    wandb.init(project=project, name=nconfig.run_name, config={**vars(nconfig)}, mode=None if nconfig.log_wandb else 'disabled')

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

wsize = 0

for step in range(nconfig.num_iterations + 1):
    last_step = (step == nconfig.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == reset_step:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step-reset_step <= 11 else (step - reset_step) + 1 # <= to avoid bug in val

    # update the window size, following SkyLadder (https://arxiv.org/abs/2503.15450)
    if nconfig.model in ["gpt", "dragon"] and nconfig.slw_warmup_iters > 0:
        wsize_old = wsize

        slw_warmup_iters = int(nconfig.slw_warmup_iters * nconfig.num_iterations)

        progress_ratio = step / slw_warmup_iters
        window = nconfig.slw_start + progress_ratio * (nconfig.sequence_length - nconfig.slw_start)
        window = nconfig.slw_increment * math.ceil(window / nconfig.slw_increment) # quantize
        window = int(min(window, nconfig.sequence_length)) # cap
        nconfig.slw_window = window

    # --------------- VALIDATION SECTION -----------------
    if (last_step or (nconfig.val_loss_every > 0 and step % nconfig.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        avg_step_time = training_time_ms / (timed_steps-1)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val, _ = val_loader.next_batch()
            with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                loss = model(x_val, targets=y_val)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss
        print0(f'step:{step}/{nconfig.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{avg_step_time:.2f}ms')
        if master_process:
            wandb.log({'val_loss': val_loss}, step=step)
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

    RMS_LOG_EVERY = 250
    if master_process and (step % RMS_LOG_EVERY == 0 or last_step):
        with torch.no_grad():
            rms_dict = {}
            for name, param in raw_model.named_parameters():
                if param.requires_grad:
                    rms = torch.sqrt(torch.mean(param.data.float() ** 2)).item()
                    rms_dict[f"weights_rms/{name}"] = rms
        if rms_dict:
            wandb.log(rms_dict, step=step)

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    approx_time = training_time_ms + 1000 * (time.time() - t0)
    avg_step_time = approx_time / timed_steps
    print0(f"step:{step+1}/{nconfig.num_iterations} train_loss:{train_loss.item():.4f} lr: {schedulers[0].get_last_lr()[0]:.4f} slw_window: {nconfig.slw_window} current_step_time:{approx_time:.0f}ms step_avg:{avg_step_time:.2f}ms")
    if master_process:
        wandb.log({'train_loss': train_loss.item(), 'step_avg_time': avg_step_time, **{f'lr_{i}': sched.get_last_lr()[0] for i, sched in enumerate(schedulers)}, 'slw_window': nconfig.slw_window, 'grad_norm': grad_norm.item()}, step=step)

    # monitor patch/token-level training
    if nconfig.use_patch_level_training and step > nconfig.patch_training_fraction*nconfig.num_iterations:
        print0("Switching to token-level training.")
        nconfig.use_patch_level_training = False

        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)

        # reset any stateful layers
        model.eval()
        model.train()

        # fallback to the original device batch size
        nconfig.device_batch_size = prev_device_batch_size
        B, T = nconfig.device_batch_size, nconfig.sequence_length
        assert nconfig.val_tokens % (B * T * ddp_world_size) == 0
        val_steps = nconfig.val_tokens // (B * T * ddp_world_size)
        assert nconfig.batch_size % (B * ddp_world_size) == 0
        train_accumulation_steps = nconfig.batch_size // (B * ddp_world_size)

        # recompute current_position in the data loaders (we dont interrupt the stream of tokens this way)
        current_pos = train_loader.current_position - train_loader.process_rank * train_loader.B * T # same on each rank
        train_loader.B = B
        train_loader.current_position = current_pos + train_loader.process_rank * train_loader.B * T
        current_pos = val_loader.current_position - val_loader.process_rank * val_loader.B * T # same on each rank
        val_loader.B = B
        val_loader.current_position = current_pos + val_loader.process_rank * val_loader.B * T

        # get the next batch (erase the prev one that used the older B)
        x, y, scores = train_loader.next_batch()

        # reset optimizer state  #todo: clean and re-use the optimizer creation code
        del optimizers
        optimizers = get_optimizers(model, nconfig, raw_model)

        # reset the learning rate scheduler (update the step count for each scheduler to match the current training progress)
        del schedulers
        schedulers = get_schedulers(optimizers, nconfig, out_of_patch_level=True)

        # start the clock again (we will have a new step_avg)
        training_time_ms = 0
        torch.cuda.synchronize()
        t0 = time.time()
        reset_step = step+1

        torch.cuda.empty_cache()

print0(f"peak memory consumption during training: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
print0("Training complete.")
dist.destroy_process_group()

# ====================================== SCORER ======================================
if nconfig.is_scorer and master_process:
    train_loader.reset(shard=0)
    run_id = nconfig.run_name + '_' + str(uuid.uuid4().hex[:8])
    data_dir = f"data/{run_id}/"
    os.makedirs(data_dir, exist_ok=True)

    shard_losses = []
    last_shard = train_loader.current_shard
    while train_loader.current_shard < int(tokens_to_skip/tokens_per_shard):
        x, y, _ = train_loader.next_batch()
        with ctx:
            loss = raw_model(x, targets=y)
            shard_losses.append(loss.detach().item()) # (B, T)
        
        if train_loader.current_shard != last_shard:
            shard_file = f"{data_dir}/shard_{last_shard:03d}.bin"
            with open(shard_file, "wb") as f:
                np.array(shard_losses, dtype=np.float32).tofile(f)
            print0(f"Saved shard losses to {shard_file}")
            shard_losses = []
            last_shard = train_loader.current_shard

    if shard_losses:
        shard_file = f"{data_dir}/shard_{last_shard:03d}.bin"
        with open(shard_file, "wb") as f:
            np.array(shard_losses, dtype=np.float32).tofile(f)
        print0(f"Saved shard losses to {shard_file}")

# ====================================== EVAL - BENCHMARKS ======================================
if nconfig.eval_benchmarks and master_process:
    from eval import eval_benchmarks

    print0(f"Evaluating on tasks: {nconfig.eval_benchmarks_tasks}.")
    results = eval_benchmarks(logdir, raw_model, nconfig.eval_tokenizer_path)
    print0("Done evaluating benchmarks.")

    # log to wandb
    for task, result in results['results'].items():
        task_name = result.get('alias')
        acc = result.get('acc,none')
        acc_norm = result.get('acc,norm')
        # WARNING: could cause problem with tasks that have an accuracy attribute in their results
        
        wandb.log({"eval/"+task_name+"_acc": acc, "eval/"+task_name+"_acc_norm": acc_norm}, step=nconfig.num_iterations)

# ====================================== EVAL - LONG-CONTEXT PG19 ======================================
if nconfig.evalpg19 and master_process:
    from eval_pg19 import eval_pg19
    
    eval_pg19(logdir, raw_model, nconfig.evalpg19_num_samples, nconfig.evalpg19_ctx_len, nconfig.evalpg19_batch_size, log_wandb=True)
    print0("Done evaluating PG19.")
