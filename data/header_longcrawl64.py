import numpy as np
import os

# put header (val.bin => val_000.bin)

INPUT_FILE  = "../longcrawl64/val.bin"
OUT_FILE    = "../longcrawl64/val_000.bin"
HEADER_INTS = 256

num_tokens = os.path.getsize(INPUT_FILE) // 2

# build header: magic, version, token-count, rest zero
header = np.zeros(HEADER_INTS, dtype=np.int32)
header[0] = 20240520
header[1] = 1
header[2] = num_tokens

# write header + original data into a new file
with open(OUT_FILE, "wb") as out, open(INPUT_FILE, "rb") as inp:
    out.write(header.tobytes())
    out.write(inp.read())

print(f"wrote {OUT_FILE} with {num_tokens} tokens")