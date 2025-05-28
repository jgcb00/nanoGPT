from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedLinearCrossEntropyLoss, FusedCrossEntropyLoss
from cut_cross_entropy import linear_cross_entropy

from config import NanoConfig
from arch.mlp import MLP
from arch.mixer.mixer_attention import MixerAttention, MixerDiffAttention, MixerNativeSparseAttention
from arch.mixer.mixer_mamba2 import MixerMamba2
from arch.mixer.mixer_gnd import MixerGatedDeltaNet

class Block(nn.Module):
    def __init__(self, config : NanoConfig, swa: bool = False, layer_depth: int = 1, kv_source=None):
        """
        swa: whether to use local attention/SWA for this block, or global
        kv_source: layer to get KV from, if any
        """
        super().__init__()

        if not swa:
            match config.attn_type:
                case "normal":
                    self.attn = MixerAttention(config, swa=swa, kv_share=(kv_source is not None))
                case "diff":
                    self.attn = MixerDiffAttention(config, swa=swa, kv_share=(kv_source is not None), layer_depth=layer_depth)
                case "nsa":
                    self.attn = MixerNativeSparseAttention(config, swa=swa)
                case _:
                    raise ValueError(f"Unknown attention type {config.attn_type}")
        else:
            match config.local_attn_type:
                case "normal":
                    self.attn = MixerAttention(config, swa=swa, kv_share=(kv_source is not None))
                case "diff":
                    self.attn = MixerDiffAttention(config, swa=swa, kv_share=(kv_source is not None), layer_depth=layer_depth)
                case "nsa":
                    self.attn = MixerNativeSparseAttention(config, swa=swa)
                case _:
                    raise ValueError(f"Unknown attention type {config.local_attn_type}")

        match config.lin_attn_type:
            case "mamba2":
                self.lin_attn = MixerMamba2(config=config)
            case "gdn":
                self.lin_attn = MixerGatedDeltaNet(config=config, expand_factor=self.attn.expand_factor)
            case _:
                raise ValueError(f"Unknown linear attention type {config.lin_attn_type}")
            
        self.expand_factor = self.attn.expand_factor

        self.kv_source = kv_source
        self.out_proj = nn.Linear(int(self.expand_factor*config.d_model), config.d_model, bias=False)
        #self.out_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.input_norm = nn.RMSNorm(config.d_model, elementwise_affine=config.rmsnorm_weights, eps=config.eps_rmsnorm)
        self.postmixer_norm = nn.RMSNorm(config.d_model, elementwise_affine=config.rmsnorm_weights, eps=config.eps_rmsnorm)
        self.mlp = MLP(config)
        
        # register here to not break torch_dynamo
        
        if config.layer_norm_scaling and config.layer_norm_scaling == "simple":
            self.register_buffer("layer_norm_scaling_1", torch.tensor(1 / math.sqrt(layer_depth+1)))
            self.register_buffer("layer_norm_scaling_2", torch.tensor(1 / math.sqrt(layer_depth+1)))
        elif config.layer_norm_scaling and config.layer_norm_scaling == "double":
            self.register_buffer("layer_norm_scaling_1", torch.tensor(1 / math.sqrt(2*layer_depth-1)))
            self.register_buffer("layer_norm_scaling_2", torch.tensor(1 / math.sqrt(2*layer_depth)))
        else:
            self.register_buffer("layer_norm_scaling_1", torch.tensor(1.0))
            self.register_buffer("layer_norm_scaling_2", torch.tensor(1.0))

    def forward(self, x, cache=None):
        external_kv = None
        if self.kv_source is not None:
            external_kv = self.kv_source.attn.get_kv()

        if cache is not None:
            attn_cache, lin_attn_cache = cache
        else:
            attn_cache, lin_attn_cache = None, None

        hidden = self.layer_norm_scaling_1 * self.input_norm(x) # (B, L, d_model)
        
        # y_attn and y_lin_attn are (B, L, E*d_model)
        y_attn,     attn_cache     = self.attn(hidden, external_kv=external_kv, cache=attn_cache)
        y_lin_attn, lin_attn_cache = self.lin_attn(hidden, cache=lin_attn_cache)
        x = x + self.out_proj((y_attn + y_lin_attn) / 2)
        x = x + self.mlp(self.layer_norm_scaling_2 * self.postmixer_norm(x))
        return x if cache is None else (x, (attn_cache, lin_attn_cache))

    def get_empty_cache(self):
        # ((k_cache, v_cache, pos), (h_cache, conv_cache))
        return (self.attn.get_empty_cache(), self.lin_attn.get_empty_cache())

class Dragon(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config

        # TODO: fuse the two loops?

        if self.config.global_attn_repart == "hymba":
            swas : List[bool] = [] # whether to use swa for each layer
            for i in range(config.n_layers):
                layer_depth = i + 1

                if config.use_swa:
                    is_first = layer_depth == 1
                    is_middle = layer_depth == (config.n_layers + 1) // 2
                    is_last = layer_depth == config.n_layers
                    swa = not (is_first or is_middle or is_last)
                else:
                    swa = False
                    
                swas.append(swa)

            blocks : List[Block] = []
            for i in range(config.n_layers):
                layer_depth = i + 1
                is_local = swas[i]
                kv_source = None

                if not config.use_kv_sharing:
                    blocks.append(Block(config, swa=is_local, layer_depth=layer_depth, kv_source=kv_source))
                    continue
                
                # KV sharing strategy
                if config.use_swa:
                    # global/local attn: share kv between consecutive local layers (globals are isolated kv-wise)
                    if is_local and i > 0 and swas[i-1]: # prev is local
                        if blocks[i-1].kv_source is None: # prev doesn't have kv source
                            kv_source = blocks[i-1]
                else:
                    # full global attn: share between every 2 layers
                    if i > 0 and i % 2 == 1: # odd layers get KV from previous even layer
                        kv_source = blocks[i-1]
                    
                blocks.append(Block(config, swa=is_local, layer_depth=layer_depth, kv_source=kv_source))
        elif self.config.global_attn_repart == "middle":
            n = config.n_layers
            G = getattr(config, "num_global_layers", 3)  # number of global layers
            base, rem = divmod(n, G)
            group_sizes = [base + (1 if i < rem else 0) for i in range(G)]
            starts = [sum(group_sizes[:i]) for i in range(G)]
            mids = {starts[i] + group_sizes[i] // 2 for i in range(G)}
            swas = [i not in mids for i in range(n)]
            blocks: List[Block] = []
            for i in range(n):
                layer_depth = i + 1
                is_local = swas[i]
                kv_source = None
                if not config.use_kv_sharing:
                    blocks.append(Block(config, swa=is_local, layer_depth=layer_depth))
                    continue

                # share kv cache between consecutive local layers
                if is_local and i > 0 and swas[i-1]:
                    prev = blocks[-1]
                    if prev.kv_source is None:
                        kv_source = prev
                blocks.append(Block(config, swa=is_local, layer_depth=layer_depth, kv_source=kv_source))
        else:
            raise NotImplementedError
            
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            h = nn.ModuleList(blocks),
        ))
        self.final_norm = nn.RMSNorm(config.d_model, elementwise_affine=config.rmsnorm_weights, eps=config.eps_rmsnorm)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size,  dtype=torch.bfloat16, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.006)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.006)

    def forward(self, idx, targets=None, scores=None, caches=None, just_logits=False):
        B, L = idx.size()

        # forward the Dragon model itself
        x = self.transformer.wte(idx) # token embeddings of shape (B, L, d_model)

        if self.config.use_patch_level_training:
            x = x.view(B, L//self.config.patch_size, self.config.patch_size, -1).mean(2) # (B, num_patches, D)
            x = x[:, :-1] # remove the last patch
            targets = targets[:, self.config.patch_size-1:-1] # targets is already shifted by one
            # so we remove only the first patch_size-1 tokens, as well as the last

        if caches is None:
            # regular forward pass
            for block in self.transformer.h:
                x = block(x)
        else:
            # forward pass with caching
            for i, block in enumerate(self.transformer.h):
                x, cache = block(x, cache=caches[i] if caches else None)

                if caches is not None:
                    caches[i] = cache

        x = self.final_norm(x)

        if just_logits:
            logits = self.lm_head(x)
            return logits            

        if targets is not None: # if we are given some desired targets also calculate the loss
            if self.config.use_patch_level_training:
                if self.config.fused_loss_computation:

                    # regular, modified
                    """
                    logits = self.lm_head(x)
                    logits = logits.float() # use tf32/fp32 for logits

                    targets = targets.reshape(-1, self.config.patch_size)

                    loss = 0
                    for i in range(self.config.patch_size):
                        loss += F.cross_entropy(logits.view(-1, logits.size(-1)), targets[:, i], ignore_index=-1)
                    loss /= self.config.patch_size"
                    """

                    # FusedLinearCrossEntropyLoss
                    criterion = FusedLinearCrossEntropyLoss(ignore_index=-1)
                    targets = targets.reshape(-1, self.config.patch_size)

                    loss = 0
                    for i in range(self.config.patch_size):
                        loss += criterion(x, targets[:, i], self.lm_head.weight)
                    loss /= self.config.patch_size

                    # FusedCrossEntropyLoss
                    """
                    criterion = FusedCrossEntropyLoss(ignore_index=-1)
                    logits = self.lm_head(x)
                    logits = logits.float() # use tf32/fp32 for logits
                    targets = targets.reshape(-1, self.config.patch_size)
                    loss = 0
                    for i in range(self.config.patch_size):
                        loss += criterion(logits.view(-1, logits.size(-1)), targets[:, i])
                    loss /= self.config.patch_size"
                    """

                else:
                    logits = self.lm_head(x)
                    logits = logits.float() # use tf32/fp32 for logits

                    targets = targets.reshape(-1, self.config.patch_size)

                    loss = 0
                    log_probs = F.log_softmax(logits.view(-1, logits.size(-1)), dim=1)
                    for i in range(self.config.patch_size):
                        loss += F.nll_loss(log_probs, targets[:, i], ignore_index=-1)
                    loss /= self.config.patch_size
            else:
                if self.config.fused_loss_computation and scores is not None:

                    # x : (B, L, d_model)
                    # targets : (B, L)
                    # scores : (B, L)

                    if self.config.scores_loss_coupling == "sqrt":
                        x = x.to(torch.bfloat16)
                        scores = scores.to(torch.bfloat16)
                        scores = scores - 1.0
                        scores = torch.where(scores < 0.3, scores.new_tensor(0.3), scores)
                        scores = torch.sqrt(scores)
                        scores[:, :32] = 1
                        #print(scores[:, 2500:2520])
                        loss = linear_cross_entropy(x, self.lm_head.weight, targets, reduction='none', ignore_index=-1)
                        loss = loss * scores
                        #print(loss)
                        loss = torch.mean(loss.view(1, -1))
                        #print(loss)
                    elif self.config.scores_loss_coupling == "soft-rho1":
                        x = x.to(torch.bfloat16)
                        scores = scores.to(torch.bfloat16)

                        unscaled_loss = linear_cross_entropy(x, self.lm_head.weight, targets, reduction='none', ignore_index=-1)

                        alpha = 0.5
                        smooth_weights = torch.sigmoid(alpha * (scores-scores.mean()))

                        loss = (unscaled_loss * smooth_weights).sum() / smooth_weights.sum()
                    else:
                        raise ValueError(f"Unknown scores_loss_coupling {self.config.scores_loss_coupling}")
                    
                elif self.config.fused_loss_computation:
                    criterion = FusedLinearCrossEntropyLoss(ignore_index=-1)
                    loss = criterion(x, targets, self.lm_head.weight)
                else:
                    logits = self.lm_head(x)
                    logits = logits.float() # use tf32/fp32 for logits
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return loss

        elif caches is None: # inference without caching (not recommended)
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            return logits
        else: # inference
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            return logits, caches
