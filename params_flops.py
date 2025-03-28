""" Computes the number of parameters and FLOPs for a given model. """

from typing import List

from config import NanoConfig


def flops_fwd_mlp(config: NanoConfig, B, L):
    return 16 * B * L * config.d_model ** 2

def flops_fwd_gdn(config: NanoConfig, B, L):
    # 64 = chunk size
    return 3 * B * L * config.d_model ** 2 + 5 * B * L * 64 * config.d_model

def flops_fwd_attn(config: NanoConfig, B, L):
    

def flops_fwd_mixer_proj(config: NanoConfig, B, L):
    return 2 * B * L * config.d_model ** 2

def get_flops(config: NanoConfig, B, L):
    flops = 0

    D = config.d_model

    # embedding layer
    flops += 0 # considered as a lookup table

    # block
    if config.model == "gpt":
        swas : List[bool] = [] # whether to use swa for each layer
        for i in range(config.n_layers):
            swa = (i%2 == 1)
            swas.append(swa)

        



    # output layer
    flops += 2 * B * L * D * config.vocab_size

    return flops

def get_params(config: NanoConfig):
    params = 0

    D = config.d_model
    H = config.n_heads

    # embedding layer
    params += config.d_model * config.vocab_size

    # block
    if config.attn_type == "normal" or config.attn_type == "diff":
        mixer_attn_p = D ** 2 + 2 * D * config.n_kv_heads * config.d_head # qkv proj
    if config.attn_type == "nsa":
        mixer_attn_p += 0.25 * D ** 2 + \
            3 * D * H + 2 * config.n_kv_heads * config.nsa_block_size * config.d_head ** 2 + \
            config.n_kv_heads * config.nsa_kernel_size * config.d_head
        # g proj, wk & wv, pe

    if config.model == "gpt":
        mixer_p = mixer_attn_p + D ** 2
    else:
        if config.lin_attn_type == "mamba2":
            mixer_lin_attn_p = 0
        elif config.lin_attn_type == "gdn":
            f = 1 # todo
            mixer_lin_attn_p = 2 * f * D ** 2 + 2 * config.expand_v * D ** 2
            # qk proj, v & gate proj
        
        mixer_proj = D ** 2

        mixer_p = mixer_attn_p + mixer_lin_attn_p + mixer_proj

    mlp_p = 8 * D ** 2

    block_p = mixer_p + mlp_p
    params += block_p * config.n_layers

    # todo: cross layer kv sharing

    # output layer
    params += D * config.vocab_size
    
    return params


