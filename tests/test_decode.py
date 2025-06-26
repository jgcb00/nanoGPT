import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from config import NanoConfig
from arch.gpt import GPT

ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

config = NanoConfig()
config.d_model = 768
config.n_heads = 12
config.n_layers = 12
config.vocab_size = 20

model = GPT(config)
model.cuda()

bsz = 5
prompt = torch.randint(0, config.vocab_size, (bsz, 10)).cuda()

# stop_tokens = [None, None, None, [0, 1, 2, 3, 4, 5], None]
# stop_tokens = [[0, 1, 2, 3, 4, 5]]*bsz
# stop_tokens = [None]*bsz
# stop_tokens = [[0, 1, 2], [3, 4], [7, 8], [10, 12], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

with ctx:
    generated1 = model.generate_naive(prompt, n_tokens=10, sample=False)
    generated2 = model.generate(
        list(torch.unbind(prompt, dim=0)), n_tokens=[10] * bsz, sample=False
    )
    generated3 = model.generate_new(
        list(torch.unbind(prompt, dim=0)),
        n_tokens=[10] * bsz,
        sample=False,
        stop_tokens=stop_tokens,
    )

print(prompt)
print(torch.allclose(torch.stack(generated2, 0), generated1))
print(generated1)
print(torch.stack(generated2, 0))
padded = pad_sequence(generated3, batch_first=True, padding_value=-1)
if padded.size(1) < 10:
    padded = F.pad(padded, (0, 10 - padded.size(1)), value=-1)
print(padded)
