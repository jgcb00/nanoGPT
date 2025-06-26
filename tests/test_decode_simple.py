import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from config import NanoConfig
from arch.gpt import GPT
from arch.lm import NanoLM

ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

config = NanoConfig()
config.model = "dragon"
config.rmsnorm = False
config.expand_factor = 2
config.n_layers = 7
config.attn_type = "diff"
config.lin_attn_type = "mamba2"
config.d_model = 768
config.n_heads = 12
config.vocab_size = 50000
config.vocab_size_real = 50000

model = GPT(config)
model.cuda()
lm = NanoLM(model)

bsz = 100
n_tokens = 100
prompt = torch.randint(0, config.vocab_size, (bsz, 10)).cuda()

with ctx:
    generated1 = lm.generate_naive(prompt, n_tokens=n_tokens, sample=False)
    generated2 = lm.generate(
        list(torch.unbind(prompt, dim=0)), n_tokens=[n_tokens] * bsz, samples=False
    )

# print(prompt)
print(torch.allclose(torch.stack(generated2, 0), generated1))
diff_count = torch.sum(generated1 != torch.stack(generated2, 0)).item()
print(diff_count / (bsz * n_tokens))
# print(generated1)
# print(torch.stack(generated2, 0))
