import os

import torch
import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP

from config import NanoConfig
from arch.mlp import MLP
from arch.mixer.mixer_attention import MixerAttention, MixerDiffAttention

# CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 test_tp.py

world_size = int(os.environ['WORLD_SIZE'])
rank       = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
dist.init_process_group(
    backend='nccl',
    init_method='env://',
    world_size=world_size,
    rank=rank,
)
device = f'cuda:{local_rank}'
torch.cuda.set_device(device)
master_process = (rank == 0)

def print0(msg):
    if master_process:
        print(msg)

torch.manual_seed(123456789)
torch.cuda.manual_seed(123456789)

# 2. Split that into DP vs. TP groups however you like.
#    Here’s the simplest: use all ranks as one TP group:
tp_ranks = list(range(dist.get_world_size()))
tp_group = dist.new_group(tp_ranks, backend="nccl")

B, L, d_model = 2, 1024, 1024
n_layers = 12
n_heads = 16

config = NanoConfig(
    model="dragon",
    d_model=d_model,
    n_heads=n_heads,
    mlp_expand=2,
    expand_factor=2,
    scalable_softmax=True,
)

x = torch.randn(B, L, d_model).to(device)

"""
# MLP
model = MLP(config=config, tp_group=tp_group, tp_size=world_size, device=device)
model = torch.compile(model)
model = model.to(device)
y = model(x)
print(y.shape)
"""

"""
# MixerAttention
model = MixerAttention(config=config, tp_group=tp_group, tp_size=world_size, device=device, swa=True, kv_share=False)
#model = torch.compile(model)
model = model.to(device)
y, _ = model(x)
print(y.shape)
"""

"""
# MixerDiffAttention
model = MixerDiffAttention(config=config, tp_group=tp_group, tp_size=world_size, device=device, swa=False, layer_depth=1)
#model = torch.compile(model)
model = model.to(device)
y, _ = model(x)
print(y.shape)
"""

# Dragon (without GDN)
model = 0

# loop and print the params of the model
for name, param in model.named_parameters():
    print0(f"  {name}: {param.data.shape}")

dist.destroy_process_group(tp_group)
dist.destroy_process_group()
