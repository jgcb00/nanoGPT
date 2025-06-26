from pathlib import Path
from dataclasses import dataclass
import pickle
import tyro
import os
import uuid
from tqdm import tqdm
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import NanoConfig
from arch.utils import get_model
from arch.data.distributed_data_loader import DistributedDataLoader


@dataclass
class Args:
    run_dir: Path  # something like logs/... (the dir that contains the .pt model)
    batch_size: int = 8

    def __post_init__(self):
        assert self.run_dir.exists(), f"Run directory {self.run_dir} does not exist."


args = tyro.cli(Args)

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    world_size=int(os.environ["WORLD_SIZE"]),
    rank=int(os.environ["RANK"]),
)
ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
ddp_world_size = int(os.environ["WORLD_SIZE"])
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
torch._dynamo.config.optimize_ddp = False


def print0(s, console=True):
    if master_process:
        print(s)


# read config
with open(args.run_dir / "config.pkl", "rb") as f:
    config: NanoConfig = pickle.load(f)
config.rmsnorm = False
config.disable_scalable_softmax_for_local = (
    True  # False for loading old runs, True for newer ones
)
config.use_patch_level_training = False
config.fused_loss_computation = False
config.scoring = True

# define and load model, tokenizer
model = get_model(config)
model = model.to(torch.bfloat16)
model = torch.compile(model, dynamic=False)
model = model.cuda()

model_file = sorted(args.run_dir.glob("state_step*.pt"))[-1]
assert model_file.exists(), f"Model file {model_file} does not exist."
print(f"Loading model from {model_file}.")

checkpoint = torch.load(model_file)
model.load_state_dict(checkpoint["model"])

model = DDP(model, device_ids=[ddp_local_rank])
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

train_loader = DistributedDataLoader(
    config.input_bin, args.batch_size, config.sequence_length, ddp_rank, ddp_world_size
)
print0(
    f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files"
)
train_loader.reset()

if master_process:
    run_id = config.run_name + "_" + str(uuid.uuid4().hex[:8])
    data_dir = f"data/{run_id}/"
    os.makedirs(data_dir, exist_ok=True)

tokens_per_shard = 100e6
tokens_to_score = 100 * 100e6  # 10e9

shard_losses = []
last_shard = train_loader.current_shard
total_shards = int(tokens_to_score / tokens_per_shard)

if master_process:
    pbar = tqdm(
        total=int(
            total_shards
            * (tokens_per_shard / (args.batch_size * config.sequence_length))
        )
    )

while train_loader.current_shard < total_shards:
    if master_process:
        pbar.set_description(f"Shard {train_loader.current_shard}/{total_shards}")

    x, y, _ = train_loader.next_batch()
    with ctx, torch.no_grad():
        loss = model(x, targets=y)  # (B*L)

    # gather loss from all processes
    loss_tensor = loss
    gathered_losses = [torch.empty_like(loss_tensor) for _ in range(ddp_world_size)]
    dist.all_gather(gathered_losses, loss_tensor)

    if master_process:
        for l in gathered_losses:
            shard_losses.extend(l.cpu().tolist())

        pbar.update(1)

        if train_loader.current_shard != last_shard:
            shard_file = os.path.join(data_dir, f"scores_{last_shard:03d}.bin")
            with open(shard_file, "wb") as f:
                np.array(shard_losses, dtype=np.float16).tofile(f)
            print0(f"Saved shard losses to {shard_file}")
            shard_losses = []
            last_shard = train_loader.current_shard

# todo: virer
if master_process and shard_losses:
    shard_file = os.path.join(data_dir, f"shard_{last_shard:03d}.bin")
    with open(shard_file, "wb") as f:
        np.array(shard_losses, dtype=np.float16).tofile(f)
    print(f"Saved shard losses to {shard_file}")

print0("Scoring complete.")
dist.destroy_process_group()
