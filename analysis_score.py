from arch.data.distributed_data_loader import DistributedDataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

B, T = 64, 4736
ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
ddp_world_size = int(os.environ["WORLD_SIZE"])
skip_tokens = 16

# load tokens
train_loader = DistributedDataLoader(
    "../nanoGPT/data/fineweb100B/fineweb_train_*.bin",
    B,
    T,
    ddp_rank,
    ddp_world_size,
    score_pattern="data/exp0_GPT2-xs-scorer_25f0c569/scores_*.bin",
)
val_loader = DistributedDataLoader(
    "../nanoGPT/data/fineweb100B/fineweb_val_*.bin", B, T, ddp_rank, ddp_world_size
)
print(
    f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files"
)
print(
    f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files"
)
x, y, scores = train_loader.next_batch()

# Define histogram bin edges
num_bins = 10000
hist_min, hist_max = 0, 10  # adjust based on expected score range
bin_edges = np.linspace(hist_min, hist_max, num_bins + 1)
hist_counts = np.zeros(num_bins, dtype=np.int64)
nb_scores = 0
accumulated_scores = torch.zeros(T, device="cuda")


for step in range(32991):
    try:
        x, y, scores = train_loader.next_batch()
    except IndexError:
        print("End of training data.")
        break
    accumulated_scores += scores.sum(dim=0)
    nb_scores += scores.shape[0]
    trimmed_scores = scores[:, skip_tokens:].flatten().cpu()

    # Compute histogram just for this batch
    batch_hist, _ = np.histogram(trimmed_scores, bins=bin_edges)
    hist_counts += batch_hist

# Plot the final histogram
plt.figure(figsize=(10, 6))
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
plt.bar(bin_centers, hist_counts, width=bin_edges[1] - bin_edges[0], color="skyblue")
plt.title("Score Distribution (incremental, excluding first 16 tokens)")
plt.xlabel("Score")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("score_distribution.png")
plt.show()


# Plot loss distribution by tokens
per_token_loss = accumulated_scores / nb_scores  # L(i)
per_token_loss_cpu = per_token_loss.cpu().numpy()
print(f"Per-token loss shape: {per_token_loss_cpu[:256]}")
plt.figure(figsize=(10, 6))
plt.plot(per_token_loss_cpu)
plt.title("Scores per-token Loss L(i) Over Position", fontsize=14)
plt.xlabel("Token Position i", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("scores_per_token_loss.png", dpi=300)
