import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
import tyro
from pathlib import Path

import torch

@dataclass
class Args:
    run_dir: Path  # something like logs/... (the dir that contains the .pt model)

    def __post_init__(self):
        print(self.run_dir)
        assert self.run_dir.exists(), f"Run directory {self.run_dir} does not exist."

args = tyro.cli(Args)

per_token_loss = torch.load(args.run_dir / 'per_token_loss.pt', map_location='cpu')

plt.figure(figsize=(10, 6))

# Method 1: Simple moving average smoothing
window_size = 100  # Adjust this based on your desired smoothness level
weights = np.ones(window_size) / window_size
smoothed_loss = np.convolve(per_token_loss.numpy(), weights, mode='valid')

# Fix: Ensure x and y have the same dimensions
x_values = np.arange(len(smoothed_loss))
plt.plot(x_values, smoothed_loss, linewidth=2, color='red', label='Moving Avg (window={})'.format(window_size))

y_min = 3.0
y_max = 5.0
plt.ylim(y_min, y_max)

# Add other plot elements
plt.title('Per-token Loss L(i) Over Position', fontsize=14)
plt.xlabel('Token Position i', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('per_token_loss.png', dpi=300)
plt.show()