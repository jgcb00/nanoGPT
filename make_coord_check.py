
import math
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from config import NanoConfig
from arch.utils import get_model, param_groups_mup

from coord_check import get_coord_data, plot_coord_data

# TRITON_HOME="/p/project1/jureap140/temp" python make_coord_check.py

use_mup = True
#lr = 2e-3
lr = 1e-2

batch_size = 8
batch_len = 1024
max_value = 100

widths = [512, 1024, 2048]
d_head = 32
n_layers = 8

class RandomDataset(Dataset):
    def __len__(self):
        return 9999999

    def __getitem__(self, idx):
        data = torch.randint(low=0, high=max_value, size=(batch_size, batch_len))
        x = data[:, :-1].int()
        y = data[:, 1:].long()
        return x.cuda(), y.cuda()

def lazy_model(width):
    nconfig = NanoConfig(model="dragon", uscaling_tau=0.2, d_model=width, n_heads=width//d_head, n_layers=n_layers, n_global_layers=3, global_attn_repart="middle", attn_type="diff", lin_attn_type="gdn", vocab_size=max_value)

    if use_mup:
        nconfig.use_uscaling = True
        nconfig.init_std = 1.0
    else:
        nconfig.use_uscaling = False
        nconfig.init_std = 0.006

    return lambda: get_model(nconfig).to("cuda")

models = {width: lazy_model(width) for width in widths}

dataset = RandomDataset()
loader = DataLoader(dataset, batch_size=None, shuffle=True)
iter_ = iter(loader)

def get_optim(model):
    global lr
    if use_mup:
        param_list = param_groups_mup(model, base_lr=lr, wd=0)
        optimizer = AdamW(param_list, betas=(0.9, 0.95))
    else:
        optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.)
    return optimizer

optcls = lambda model: get_optim(model)

df = get_coord_data(models, iter_, optcls, nsteps=10)

if use_mup:
    name = f"mup_lr1oversqrt_GDN1overd_attn1overd_{lr}.png"
else:
    name = f"no_mup_{lr}.png"

plot_coord_data(df, legend="full", save_to=name)
