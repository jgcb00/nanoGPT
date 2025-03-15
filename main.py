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

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from arch.utils import get_model
from config import NanoConfig
from arch.data.distributed_data_loader import DistributedDataLoader
from arch.optim.get_optimizer import get_optimizers
from arch.schedulers import get_schedulers
from eval_pg19 import eval_pg19
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
train_loader = DistributedDataLoader(nconfig.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(nconfig.input_val_bin, B, T, ddp_rank, ddp_world_size)
print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

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

# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
# init the optimizer(s)
optimizers = get_optimizers(model, nconfig, raw_model)
schedulers = get_schedulers(optimizers, nconfig)

# begin wandb logging
if master_process:
    wandb.init(project='nanoGPT', name=nconfig.run_name, config={**vars(nconfig)}, mode=None if nconfig.log_wandb else 'disabled')

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

    # update the local/swa window size (start at 64, and increase by 64 gradually over swa_warmup_iters)
    if nconfig.model in ["gpt", "dragon"] and nconfig.use_swa and nconfig.swa_warmup_iters > 0:
        swa_window_size = int(min(64*((step/nconfig.swa_warmup_iters * (nconfig.swa_window_size - 64) + 64)//64), nconfig.swa_window_size))
        for block in raw_model.transformer.h:
            block.attn.window_size = swa_window_size

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
                loss = model(x_val, targets=y_val)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss
        print0(f'step:{step}/{nconfig.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
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

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            loss = model(x, targets=y)
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
    print0(f"step:{step+1}/{nconfig.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
    if master_process:
        wandb.log({'train_loss': train_loss.item(), 'step_avg_time': approx_time/timed_steps, **{f'lr_{i}': sched.get_last_lr()[0] for i, sched in enumerate(schedulers)}, 'grad_norm': grad_norm.item()}, step=step)

    # monitor patch/token-level training
    if nconfig.use_patch_level_training and step > nconfig.patch_training_fraction*nconfig.num_iterations:
        print0("Switching to token-level training.")
        nconfig.use_patch_level_training = False

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
        x, y = train_loader.next_batch()

        # reset optimizer state  #todo: clean and re-use the optimizer creation code
        del optimizers
        optimizers = get_optimizers(model, nconfig, raw_model)

        # reset the learning rate scheduler (update the step count for each scheduler to match the current training progress)
        del schedulers
        schedulers = get_schedulers(optimizers, nconfig, out_of_patch_level=True)
        current_step_fraction = step / nconfig.num_iterations
        for scheduler in schedulers:
            scheduler.last_epoch = int(current_step_fraction*nconfig.num_iterations) - 1

        torch.cuda.empty_cache()

print0(f"peak memory consumption during training: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
print0("Training complete.")
dist.destroy_process_group()

# ====================================== EVAL - BENCHMARKS ======================================
if nconfig.eval_benchmarks and master_process:
    import lm_eval
    import tiktoken
    from arch.lm import NanoLM
    
    # load tokenizer
    with open(nconfig.eval_tokenizer_path, 'rb') as f:
        enc_pickled = pickle.load(f)
    enc = tiktoken.core.Encoding(enc_pickled.pop('name'), **enc_pickled)
    
    print0(f"Evaluating on tasks: {nconfig.eval_benchmarks_tasks} with 1GPUs")

    lm = NanoLM(
        model=raw_model, 
        config=nconfig, 
        enc=enc, 
    )

    # evaluate
    results = lm_eval.simple_evaluate(lm, tasks=nconfig.eval_benchmarks_tasks, limit=1.)

    # save results (with the names of the tasks in the file)
    result_file_path = os.path.join(logdir, f"results_{'_'.join(nconfig.eval_benchmarks_tasks)}.json")
    with open(result_file_path, 'w') as f:
        json.dump(results['results'], f)

    # log to wandb
    for task, result in results['results'].items():
        task_name = result.get('alias')
        acc = result.get('acc,none')
        acc_norm = result.get('acc,norm')
        # could cause problem with tasks that have an accuracy attribute in their results
        
        wandb.log({"eval/"+task_name+"_acc": acc, "eval/"+task_name+"_acc_norm": acc_norm}, step=nconfig.num_iterations)

    print0("Done evaluating benchmarks.")

# ====================================== EVAL - LONG-CONTEXT PG19 ======================================
if nconfig.evalpg19 and master_process:
    eval_pg19(logdir, model, nconfig.evalpg19_num_samples, nconfig.evalpg19_ctx_len, nconfig.evalpg19_batch_size, log_wandb=True)
    print0("Done evaluating PG19.")
