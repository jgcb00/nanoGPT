import torch

import matplotlib.pyplot as plt
import numpy as np

from arch.data.distributed_data_loader import DistributedDataLoader
from config import NanoConfig

config = NanoConfig()
config.sequence_length = 4736
config.input_bin = '../nanoGPT/data/fineweb100B/fineweb_train_*.bin'
score_pattern = "data/exp0_GPT2-xs-scorer_25f0c569/scores_*.bin"

B = 16
rank, processes = 0, 1

train_loader = DistributedDataLoader(config.input_bin, B, config.sequence_length, rank, processes, score_pattern=score_pattern)
print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files.")

num_batches = 1000  # or any number you want
all_scores = []

for _ in range(num_batches):
    x, y, scores = train_loader.next_batch()
    scores_np = scores.cpu().numpy() if hasattr(scores, 'cpu') else np.array(scores)
    all_scores.append(scores_np)

all_scores_np = np.concatenate(all_scores, axis=0)  # shape: (num_batches * B, L)

scores_np = all_scores_np

print(scores_np.shape)
print(scores_np)

print(f"Max: {scores_np.max()}, Min: {scores_np.min()}, Mean: {scores_np.mean()}, Std: {scores_np.std()}")

# Histogram of all scores
plt.figure(figsize=(8, 6))
plt.hist(scores_np.flatten(), bins=50, alpha=0.7, range=(0, 10))
plt.title("Histogram of Scores")
plt.xlabel("Score Value")
plt.ylabel("Frequency")
plt.savefig("scores_histogram.png")

# Mean score per batch element
mean_scores = scores_np.mean(axis=1)  # shape: (B,)
plt.figure(figsize=(8, 6))
plt.plot(mean_scores, 'o-', label='Mean Score')
plt.title("Mean Score per Batch Element")
plt.xlabel("Batch Index")
plt.ylabel("Mean Score")
plt.legend()
plt.savefig("scores_mean.png")

# Heatmap of scores for the first batch element (if L is manageable)
plt.figure(figsize=(12, 2))
plt.imshow(scores_np[0:1, :], aspect='auto', cmap='viridis')
plt.title("Heatmap of Scores (First Batch Element)")
plt.xlabel("Token Position")
plt.colorbar()
plt.savefig("scores_heatmap.png")

# plot the evolution of the scores throughout the sequence
plt.figure(figsize=(12, 6))
plt.plot(scores_np[0, :], marker='o', linestyle='-', label='Score Evolution')
plt.title("Score Evolution for First Batch Element")
plt.xlabel("Token Position")
plt.ylabel("Score Value")
plt.legend()
plt.savefig("scores_evolution.png")

# ----- Sigmoid Smoothing Weighting -----
# Here we compute a smooth weight for each token based on the reference score using a sigmoid.
alpha = 1 # 10.0  # Controls the steepness of the transition (larger alpha => steeper transition)
excess = scores - 1.0  # Compute excess loss over a baseline (1.0 in this case)
smooth_weights = torch.sigmoid(alpha * excess)
#smooth_weights[:, :32] = 1.0  # Ensure first 32 tokens are fully weighted

smooth_weights_np = smooth_weights.cpu().numpy() if hasattr(smooth_weights, 'cpu') else np.array(smooth_weights)

# Histogram of smooth weights
plt.figure(figsize=(8, 6))
plt.hist(smooth_weights_np.flatten(), bins=50, alpha=0.7)
plt.title("Histogram of Smooth Weights")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.savefig("smooth_weights_histogram.png")

# Heatmap of smooth weights for the first batch element
plt.figure(figsize=(12, 2))
plt.imshow(smooth_weights_np[0:1, :], aspect='auto', cmap='viridis')
plt.title("Heatmap of Smooth Weights (First Batch Element)")
plt.xlabel("Token Position")
plt.colorbar()
plt.savefig("smooth_weights_heatmap.png")

# Plot the evolution of the smooth weights throughout the sequence
plt.figure(figsize=(12, 6))
plt.plot(smooth_weights_np[0, :], marker='o', linestyle='-', label='Smooth Weight Evolution')
plt.title("Smooth Weight Evolution for First Batch Element")
plt.xlabel("Token Position")
plt.ylabel("Weight Value")
plt.legend()
plt.savefig("smooth_weights_evolution.png")