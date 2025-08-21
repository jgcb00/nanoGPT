import os
import glob
import torch
import numpy as np

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename, source_type: str):
    if source_type == "fineweb":
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        if header[0] != 20240520:
            print("ERROR: magic number mismatch in the data .bin file!")
            print("---> HINT: Are you passing in a correct file with --input_bin?")
            print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
            exit(1)
        assert header[1] == 1, "unsupported version"
        ntok = int(header[2])
        return ntok
    elif source_type == "MG":
        return os.path.getsize(filename) // 4
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

def _load_data_shard(filename, source_type: str):
    if source_type == "fineweb":
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
            assert header[0] == 20240520, "magic number mismatch in the data .bin file"
            assert header[1] == 1, "unsupported version"
            ntok = int(header[2])
        # memmap the token payload directly (uint16) after the 256*4B header
        tokens = np.memmap(filename, dtype=np.uint16, mode="r", offset=256 * 4, shape=(ntok,))
        assert tokens.size == ntok, "number of tokens read does not match header?"
        return tokens
    elif source_type == "MG":
        # Entire file is uint32 tokens, no header
        tokens = np.memmap(filename, dtype=np.uint32, mode="r")
        return tokens
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes, source_type: str = "fineweb"):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B # micro batch size
        self.T = T
        self.source_type = source_type

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        self.shard_ntoks = []
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname, self.source_type)
            print(f"shard {fname} has {shard_ntok} tokens")
            assert shard_ntok >= num_processes * B * T + 1
            self.shard_ntoks.append(shard_ntok)
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self, shard=0):
        self.current_shard = shard
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard], self.source_type)

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard], self.source_type)
        
        if self.process_rank == 0:
            shard_tokens = self.shard_ntoks[self.current_shard]
            cum_tokens = sum(self.shard_ntoks[: self.current_shard + 1])

            def _fmt(n):
                return f"{n/1e9:.2f}B" if n >= 1_000_000_000 else (
                    f"{n/1e6:.2f}M" if n >= 1_000_000 else str(n))

            print(
                f"Advancing to shard {self.current_shard}/{len(self.files)-1} "
                f"(this={_fmt(shard_tokens)} tok, cum={_fmt(cum_tokens)}/{_fmt(self.ntok_total)})"
            )

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = np.asarray(buf, dtype=np.int64)
        x = torch.from_numpy(buf[:-1].reshape(B, T)) # inputs
        y = torch.from_numpy(buf[1: ].reshape(B, T)) # targets

        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()

        return x.cuda(), y.cuda()
