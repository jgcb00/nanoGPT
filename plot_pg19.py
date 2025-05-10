import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
import tyro
from pathlib import Path
from typing import Optional, Sequence

import torch

@dataclass
class Args:
    run_dirs: Sequence[Path]               # list of run directories
    names: Optional[Sequence[str]] = None  # optional labels for each run

    def __post_init__(self):
        for rd in self.run_dirs:
            print(rd)
            assert rd.exists(), f"Run directory {rd} does not exist."
        if self.names is not None:
            assert len(self.names) == len(self.run_dirs), \
                "Number of names must match number of run_dirs."

args = tyro.cli(Args)

plt.figure(figsize=(10, 6))
for idx, rd in enumerate(args.run_dirs):
    per_token_loss = torch.load(rd / 'per_token_loss.pt', map_location='cpu')
    window_size = 500
    weights = np.ones(window_size) / window_size
    smoothed_loss = np.convolve(per_token_loss.numpy(), weights, mode='valid')
    x_values = np.arange(len(smoothed_loss))
    label = args.names[idx] if args.names else rd.name
    plt.plot(x_values, smoothed_loss, linewidth=2, label=label)

y_min = 2.8
y_max = 3.1
plt.ylim(y_min, y_max)

# Add other plot elements
plt.title('Per-token Loss L(i) Over Position', fontsize=14)
plt.xlabel('Token Position i', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('per_token_loss.png', dpi=600)
plt.show()