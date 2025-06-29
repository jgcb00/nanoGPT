from dataclasses import dataclass
from typing import List, Union, Optional

# todo: on pourra utiliser un typing union pour forcer certaines valeurs c'est plus propre
# gpt: n_layers=12, expand_factor=1; dragon: n_layers=7, expand_factor=2


@dataclass
class NanoConfig:
    # model
    model: str = "gpt"  # gpt or dragon or gated-delta-net or mamba2
    run_name: str = ""

    # arch - general
    d_model: int = 768
    n_heads: int = 6  # head dim 128 suggested by @Grad62304977
    n_layers: int = 12
    n_global_layers: int = 3
    expand_factor: int = 1  # expand factor for Mamba/Dragon
    attn_type: str = "normal"  # normal, diff
    local_attn_type: str = "normal"
    lin_attn_type: str = "mamba2"  # mamba2, gdn
    global_attn_repart: str = (
        "hymba"  # hymba (beginning,middle,end), middle (3 parts, global @middle of each part)
    )
    rope_theta_global: float = 10000.0
    rope_theta_local: float = 10000.0
    layer_norm_scaling: bool = False  # whether to scale layer norm by sqrt(layer_depth)
    layer_norm_scaling_type: str = "simple"  # simple, double
    rmsnorm_weights: bool = False
    groupnorm: bool = True
    groupnorm_weights: bool = True
    groupnorm_unique: bool = (
        False  # if False, groupnorm weights are shared among heads of same nature. if True, weights for each head.
    )
    groupnorm_unique_independent: bool = (
        False  # True=normalization is done indenpendently for each head, False=normalization is done jointly for all heads
    )
    eps_rmsnorm: Optional[float] = None
    mlp_expand: int = 4  # expand factor for MLP
    gate_act_attn: str = "silu"  # silu, srelu, sigmoid
    gate_type_attn: str = "elementwise"  # elementwise, headwise
    norm_before_gate_attn: bool = False
    gate_act_gdn: str = "silu"  # silu, srelu, sigmoid
    gate_type_gdn: str = "elementwise"  # elementwise, headwise
    input_norm: bool = False
    full_lambdas: bool = (
        False  # (n_heads, d_head) if full_lambdas=True, (n_heads,) if False (MG compatibility)
    )
    fused_loss_computation: bool = (
        True  # whether to use fused linear + cross entropy loss
    )

    # Attention related
    n_kv_heads: int = 0
    use_kv_sharing: bool = False  # cross-layer KV sharing
    use_swa: bool = (
        False  # mix global and local attention (first, middle and last block) or use full global
    )
    swa_window_size: int = 1024  # local attention window size
    slw_warmup_iters: float = (
        0  # on how many iteratons (%) to warmup the attention window size (0=no warmup)
    )
    slw_start: int = 8  # window size at the start of training
    slw_increment: int = 64  # window size increment at each step
    qk_norm: bool = True
    scalable_softmax: bool = False
    disable_scalable_softmax_for_local: bool = True
    rope_to_nope: bool = (
        False  # whether to use the rope-to-nope arch (2501.18795, ie disable RoPE in full attn layers) (only effective if use_swa=True)
    )
    use_gate_attn: bool = False  # applies to all attentions (normal and diff)

    # Mamba related
    d_state: int = 128
    d_conv: int = 4
    headdim: int = 64
    ngroups: int = 8
    norm_before_gate: bool = (
        False  # placement of the output norm relative to the gate: True is norm(x) * f(z) and False is norm(x * f(z))
    )

    # GatedDeltaNet related
    use_gate: bool = True
    expand_v: int = 2

    # LaCT related
    lact_chunk_size: int = 2048
    lact_n_heads: int = 4
    lact_expand_factor: int = 1
    lact_use_momentum: bool = False
    lact_use_muon: bool = False
    lact_w0_w2_low_rank: int = 0 # 0=no low rank, >0 for low rank
    lact_fw_init_gain: float = 0.5
    lact_lr_dim: int = 1

    # optim
    optim: str = "muon"  # adamw, spam, stable-spam, muon, muon_moonlight, splus
    batch_size: int = 8 * 64  # batch size, in sequences, across all devices
    device_batch_size: int = 64  # batch size, in sequences, per device
    num_iterations: int = 1000  # number of iterations to run
    learning_rate: float = 1e-4
    warmup_iters: float = 0.0  # WSD (%)
    warmdown_iters: float = 0.15  # WSD (%)
    weight_decay: float = 0.0
    grad_norm_clip: float = 1.0
    scheduler: str = "wsd"  # linear-slow or moonlight

    # data
    vocab_size: int = 50304
    sequence_length: int = 1024  # sequence length, in tokens
    use_patch_level_training: bool = False
    patch_size: int = 4
    patch_training_fraction: float = 0.67
    input_bin: str = "data/fineweb10B/fineweb_train_*.bin"  # input .bin to train on
    input_val_bin: str = (
        "data/fineweb10B/fineweb_val_*.bin"  # input .bin to eval validation loss on
    )

    # evaluation and logging
    val_loss_every: int = (
        125  # every how many steps to evaluate val loss? 0 for only at the end
    )
    val_tokens: int = (
        10485760  # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    )
    save_every: int = (
        0  # every how many steps to save the checkpoint? 0 for only at the end
    )
    log_wandb: bool = False  # whether to log to wandb

    # used during training
    slw_window: int = 0

    # for logging
    num_params: int = 0

    # inference related
    vocab_size_real: int = 50257

    # eval - benchmarks (at the end of training)
    eval_tokenizer_name: str = (
        "gpt2"  # should point to a HF tokenizer. 'alexandretl/dragon-tokenizer'
    )
    eval_benchmarks: bool = True
    eval_benchmarks_tasks: str = (
        "hellaswag,swde,fda,openbookqa,arc_easy,arc_challenge,piqa,winogrande,lambada,squadv2"
    )

    # eval - long-context PG19
    evalpg19: bool = True
    evalpg19_ctx_len: int = 16384
    evalpg19_num_samples: int = 2048
    evalpg19_batch_size: int = 4

    def __post_init__(self):
        # check for valid model
        assert self.model in ["gpt", "dragon", "gated-delta-net", "mamba2"]
        # check for valid attention type
        assert self.attn_type in ["normal", "diff", "gta", "gtda"]
        # check for valid linear attention type
        assert self.lin_attn_type in ["mamba2", "gdn"]
        # check for valid optimizer type
        assert self.optim in [
            "adamw",
            "spam",
            "muon",
            "stable-spam",
            "muon_moonlight",
            "swan",
            "splus",
        ]
        # check for valid n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = self.d_model // self.n_heads
        # check for valid n_kv_heads
        if self.n_kv_heads == 0:
            self.n_kv_heads = self.n_heads
        assert (
            self.n_heads % self.n_kv_heads == 0
        ), "n_heads must be divisible by n_kv_heads"
        if self.attn_type == "diff":
            assert (
                self.n_heads % 2 == 0
            ), "n_heads must be even when using diff attention"
            assert (
                self.n_kv_heads % 2 == 0
            ), "n_kv_heads must be even when using diff attention"

        self.eval_benchmarks_tasks = self.eval_benchmarks_tasks.split(",")
