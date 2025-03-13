import pickle
import tiktoken
import tyro
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from config import NanoConfig
from arch.utils import get_model

from datasets import load_dataset

"""
USED FOR EVALUATING ALREADY, OLD, TRAINED MODELS
WILL BE DELETED, AS THIS CODE IS ALSO PRESENT AFTER THE TRAINING LOOP IN THE MAIN SCRIPT
"""

def eval_pg19(log_dir, model, nsamples, ctx_len, batch_size, log_wandb=True):
    log_dir = Path(log_dir)
    
    # load tokenizer
    with open('data/enc.pkl', 'rb') as f:
        enc_pickled = pickle.load(f)
    enc = tiktoken.core.Encoding(enc_pickled.pop('name'), **enc_pickled)

    # load PG19 dataset
    ds = load_dataset("emozilla/pg19")

    accumulated_losses = torch.zeros(ctx_len, device='cuda')
    example_count = 0
    batch_examples = []

    for example in tqdm(ds["train"], total=nsamples):
        if nsamples > 0 and example_count >= nsamples:
            break

        input_enc = enc.encode(example['text'])

        if len(input_enc) < ctx_len:
            continue

        batch_examples.append(input_enc[:ctx_len+1])

        if len(batch_examples) == batch_size:
            if nsamples > 0 and example_count + len(batch_examples) > nsamples:
                batch_examples = batch_examples[: nsamples - example_count]

            x = torch.tensor([ex[:-1] for ex in batch_examples], dtype=torch.long, device='cuda')
            y = torch.tensor([ex[1:] for ex in batch_examples], dtype=torch.long, device='cuda')

            with torch.no_grad():
                with ctx:
                    logits = model(x, just_logits=True)
                B, L, vocab_size = logits.size()
                token_losses = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), reduction='none')
                token_losses = token_losses.view(B, L)
            accumulated_losses += token_losses.sum(dim=0)
            example_count += len(batch_examples)
            batch_examples = []

    per_token_loss = accumulated_losses / example_count # L(i)
    per_token_loss_cpu = per_token_loss.cpu().numpy()

    # save tensor to file
    torch.save(per_token_loss, log_dir / f'per_token_loss.pt')

    plt.figure(figsize=(10, 6))
    plt.plot(per_token_loss_cpu)
    plt.title('Per-token Loss L(i) Over Position', fontsize=14)
    plt.xlabel('Token Position i', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(log_dir / 'per_token_loss.png', dpi=300)

    if log_wandb:
        import wandb
        # log to wandb
        data = [[i, loss] for i, loss in enumerate(per_token_loss_cpu)]
        table = wandb.Table(data=data, columns=["token_position", "loss"])
        wandb.log({"eval/per_token_loss_plot": wandb.plot.line(table, "token_position", "loss", title="Loss per Token")}, step=nconfig.num_iterations)


if __name__ == "__main__":
    ctx_len = 16384 # 4 * 4096, the training context length
    batch_size = 4 # for that fits on A100-65GB

    @dataclass
    class Args:
        run_dir: Path # something like logs/... (the dir that contains the .pt model)
        num_samples: int = 100 # -1 for all

        def __post_init__(self):
            assert self.run_dir.exists(), f"Run directory {self.run_dir} does not exist."

    args = tyro.cli(Args)
    # read config
    with open(args.run_dir / 'config.pkl', 'rb') as f:
        config: NanoConfig = pickle.load(f)
    config.rmsnorm = False
    config.disable_scalable_softmax_for_local = True # for loading old runs

    # define and load model
    model = get_model(config)
    model.cuda()
    model.eval()

    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    model_file = sorted(args.run_dir.glob("state_step*.pt"))[-1]
    assert model_file.exists(), f"Model file {model_file} does not exist."
    checkpoint = torch.load(model_file)
    state_dict = checkpoint['model']
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    eval_pg19(args.run_dir, model, args.num_samples, ctx_len, 4, log_wandb=False)