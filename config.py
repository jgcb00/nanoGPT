from dataclasses import dataclass, asdict, field
import argparse

@dataclass
class NanoConfig:
    # model
    model : str = "gpt" #gpt or dragon
    run_name : str = None
    
    # arch - general
    d_model : int = 768
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_layer : int = 12

    # optim
    optim : str = "muon" # adamw or muon
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device
    num_iterations : int = 4578 # number of iterations to run
    learning_rate : float = 1e-4
    warmup_iters : int = 0
    warmdown_iters : int = 1308 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0.
    grad_norm_clip : float = 1.0

    # data
    vocab_size : int = 50304
    sequence_length : int = 1024 # sequence length, in tokens
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    
    # evaluation and logging
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    log_wandb : bool = False # whether to log to wandb
    
    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group('NanoConfig')
        for key, value in NanoConfig().__dict__.items():
            arg_type = type(value)
            parser.add_argument(f'--{key}', type=arg_type, default=value)
        return parent_parser

    @classmethod
    def from_args(cls, args):
        """Create config from argparse namespace."""
        config_dict = {k: v for k, v in vars(args).items() if k in cls().__dict__}
        return cls(**config_dict)
