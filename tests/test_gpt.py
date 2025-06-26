import torch
import torch.nn as nn
from config import NanoConfig
from arch.gpt import GPT

# Set up configuration
config = NanoConfig()
config.d_model = 128
config.n_heads = 4
config.n_kv_heads = 2  # For GQA
config.n_layers = 1
config.vocab_size = 1000
config.expand_factor = 1
config.qk_norm = True
config.scalable_softmax = True
config.attn_type = "diff"
config.use_kv_sharing = False
config.use_swa = True
config.layer_norm_scaling = False
config.d_conv = 4  # Needed for proper cache initialization

# Set seed for reproducibility
torch.manual_seed(42)
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# Create model
model = GPT(config).to("cuda")
model.eval()

# Create a test prompt
batch_size = 2
seq_len = 8
prompts = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")

# Number of tokens to generate
n_tokens = 64

# Generation parameters
sample = False  # Whether to sample or take argmax
top_k = 50  # Top-k sampling
temperature = 0.8  # Temperature for sampling

# Test 1: Generate using the naive method (full forward pass)
print("Running generate_naive...")
with torch.no_grad(), ctx:
    output_naive = model.generate_naive(
        prompts, n_tokens=n_tokens, sample=sample, top_k=top_k, temperature=temperature
    )

# Test 2: Generate using the optimized method (with caching)
print("Running generate...")
with torch.no_grad(), ctx:
    # For generate function, we need to prepare list of 1D tensors
    prompt_list = [prompts[i, :] for i in range(batch_size)]
    n_tokens_list = [n_tokens] * batch_size

    output_cached = model.generate(
        prompt_list,
        n_tokens=n_tokens_list,
        sample=sample,
        top_k=top_k,
        temperature=temperature,
    )

    # Stack the results to match output_naive format (from list of responses to tensor)
    output_cached = torch.stack(output_cached)

print(f"Naive output shape: {output_naive.shape}")
print(f"Cached output shape: {output_cached.shape}")

print(torch.allclose(output_naive, output_cached))
