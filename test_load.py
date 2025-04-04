import torch

from arch.data.distributed_data_loader import DistributedDataLoader
from config import NanoConfig

config = NanoConfig()
config.sequence_length = 4736
config.input_bin = '../nanoGPT/data/fineweb100B/fineweb_train_*.bin'
score_pattern = "data/exp0_GPT2-xs-scorer_7268ddf8/scores_*.bin"

B = 48
rank, processes = 0, 1

train_loader = DistributedDataLoader(config.input_bin, B, config.sequence_length, rank, processes, score_pattern=score_pattern)
print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files.")

x, y, scores = train_loader.next_batch()

print("Inputs shape:", x.shape)
print("Targets shape:", y.shape)
print("Scores shape:", scores.shape)
