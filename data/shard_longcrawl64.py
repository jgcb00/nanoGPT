import numpy as np
import os

# shard + put headers (train.bin => train_000.bin, train_001.bin, ...)

INPUT_FILE = "../longcrawl64/train.bin"
OUT_PATTERN = "../longcrawl64/train_{:03d}.bin"
TOKENS_PER_SHARD = 100_000_000  # tune as you like
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4

# 1) figure out total tokens (2 bytes each)
file_size = os.path.getsize(INPUT_FILE)
total_tokens = file_size // 2

"""
# 1.1) little sanity check (memory-map and decode the first N tokens)
import pickle
import tiktoken

# load tokenizer
with open("data/enc.pkl", 'rb') as f:
    enc_pickled = pickle.load(f)
enc = tiktoken.core.Encoding(enc_pickled.pop('name'), **enc_pickled)

N = 2000  # or whatever subset size you like
tokens = np.memmap(INPUT_FILE, dtype=np.uint16, mode="r", shape=(total_tokens,))
sample = tokens[:N].tolist()

# decode and print
text = enc.decode(sample)
print(text)
"""

# 2) memmap all tokens
tokens = np.memmap(INPUT_FILE, dtype=np.uint16, mode="r", shape=(total_tokens,))

n_shards = int(np.ceil(total_tokens / TOKENS_PER_SHARD))

for i in range(n_shards):
    start = i * TOKENS_PER_SHARD
    end = min((i + 1) * TOKENS_PER_SHARD, total_tokens)
    shard = tokens[start:end]

    # build a fresh header
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1  # version
    header[2] = end - start  # this shard’s token count

    out_file = OUT_PATTERN.format(i)
    with open(out_file, "wb") as f:
        f.write(header.tobytes())
        f.write(shard.tobytes())

    print(f"wrote {out_file}: {end-start} tokens")
