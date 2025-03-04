from dataclasses import dataclass

# todo: on pourra utiliser un typing union pour forcer certaines valeurs c'est plus propre

@dataclass
class NanoConfig:
    # model
    model : str = "gpt" #gpt or dragon
    run_name : str = ""
    
    # arch - general
    d_model : int = 768
    n_heads : int = 6 # head dim 128 suggested by @Grad62304977
    n_layers : int = 12
    expand_factor : int = 1 # expand factor for Mamba/Dragon
    attn_type : str = "normal" # normal, diff
    lin_attn_type: str = "mamba2" # mamba2, gdn
    layer_norm_scaling : bool = False # whether to scale layer norm by sqrt(layer_depth)

    # Attention related
    n_kv_heads : int = 0
    use_kv_sharing : bool = False # cross-layer KV sharing
    use_swa : bool = False # mix global and local attention (first, middle and last block) or use full global
    swa_window_size : int = 1024 # local attention window size
    swa_warmup_iters: int = 0 # on how many iteratons to warmup the local attention window size (0=no warmup)
    qk_norm: bool = True
    scalable_softmax: bool = False

    # Mamba and GatedDeltaNet related
    rmsnorm: bool = False # whether to use an output norm (before proj)

    # Mamba related
    d_state: int = 128
    d_conv: int = 4
    headdim: int = 64
    ngroups : int = 8
    norm_before_gate: bool = False # placement of the output norm relative to the gate: True is norm(x) * f(z) and False is norm(x * f(z))

    # GatedDeltaNet related
    use_gate: bool = True
    expand_v : int = 2

    # optim
    optim : str = "muon" # adamw, spam, or muon
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device
    num_iterations : int = 1000 # number of iterations to run
    learning_rate : float = 1e-4
    warmup_iters : int = 0 # WSD
    warmdown_iters : int = 150 # WSD
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

    # for logging
    num_params: int = 0
    
    def __post_init__(self):
        # check for valid model
        assert self.model in ["gpt", "dragon", "gated-delta-net"]
        # check for valid attention type
        assert self.attn_type in ["normal", "diff"]
        # check for valid lin attn type
        assert self.lin_attn_type in ["mamba2", "gdn"]
        # check for valid optim type
        assert self.optim in ["adamw", "spam", "muon", "stable-spam"]
        # check for valid n_kv_heads
        if self.n_kv_heads == 0:
            self.n_kv_heads = self.n_heads
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        if self.attn_type == "diff":
            assert self.n_heads % 2 == 0, "n_heads must be even when using diff attention"
            assert self.n_kv_heads % 2 == 0, "n_kv_heads must be even when using diff attention"