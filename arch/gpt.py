from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedLinearCrossEntropyLoss, FusedCrossEntropyLoss
from cut_cross_entropy import linear_cross_entropy

from config import NanoConfig
from arch.mlp import MLP
from arch.mixer.mixer_attention import MixerAttention, MixerDiffAttention


class Block(nn.Module):
    def __init__(self, config: NanoConfig, layer_depth: int = 1, kv_source=None):
        """
        kv_source: layer to get KV from, if any
        """
        super().__init__()

        match config.attn_type:
            case "normal":
                self.attn = MixerAttention(
                    config, swa=False, kv_share=(kv_source is not None)
                )
            case "diff":
                self.attn = MixerDiffAttention(
                    config,
                    swa=False,
                    kv_share=(kv_source is not None),
                    layer_depth=layer_depth,
                )
            case _:
                raise ValueError(f"Unknown attention type {config.attn_type}")

        self.expand_factor = self.attn.expand_factor

        self.kv_source = kv_source
        self.out_proj = nn.Linear(
            int(self.expand_factor * config.d_model), config.d_model, bias=False
        )
        self.input_norm = nn.RMSNorm(
            config.d_model, elementwise_affine=config.rmsnorm_weights
        )
        self.postmixer_norm = nn.RMSNorm(
            config.d_model, elementwise_affine=config.rmsnorm_weights
        )
        self.mlp = MLP(config)

        # register here to not break torch_dynamo
        self.register_buffer(
            "layer_norm_scaling",
            torch.tensor(
                1 / math.sqrt(layer_depth) if config.layer_norm_scaling else 1.0
            ),
        )

    def forward(self, x, cache=None):
        external_kv = None
        if self.kv_source is not None:
            external_kv = self.kv_source.attn.get_kv()

        if cache is not None:
            attn_cache = cache
        else:
            attn_cache = None

        hidden = self.layer_norm_scaling * self.input_norm(x)  # (B, L, d_model)

        # y_attn and y_lin_attn are (B, L, E*d_model)
        y_attn, attn_cache = self.attn(
            hidden, external_kv=external_kv, cache=attn_cache
        )
        x = x + self.out_proj(y_attn)
        x = x + self.mlp(self.layer_norm_scaling * self.postmixer_norm(x))
        return x if cache is None else (x, attn_cache)

    def get_empty_cache(self):
        # (k_cache, v_cache, pos)
        return self.attn.get_empty_cache()


class GPT(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config

        if self.config.use_kv_sharing:
            blocks = []
            for i in range(config.n_layers):
                if i % 2 == 0:
                    blocks.append(Block(config, layer_depth=i + 1))
                else:
                    blocks.append(
                        Block(config, layer_depth=i + 1, kv_source=blocks[i - 1])
                    )
        else:
            blocks = [Block(config, layer_depth=i + 1) for i in range(config.n_layers)]

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.d_model),
                h=nn.ModuleList(blocks),
            )
        )
        self.input_norm = nn.RMSNorm(
            config.d_model, elementwise_affine=config.rmsnorm_weights
        )
        self.final_norm = nn.RMSNorm(
            config.d_model, elementwise_affine=config.rmsnorm_weights
        )
        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, dtype=torch.bfloat16, bias=False
        )
        # self.lm_head.weight.data.zero_()

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
        x = self.transformer.wte(idx)  # token embeddings of shape (B, L, d_model)

        if self.config.use_patch_level_training:
            x = x.view(B, L // self.config.patch_size, self.config.patch_size, -1).mean(
                2
            )  # (B, num_patches, D)
            x = x[:, :-1]  # remove the last patch
            targets = targets[
                :, self.config.patch_size - 1 : -1
            ]  # targets is already shifted by one
            # so we remove only the first patch_size-1 tokens, as well as the last

        x = self.input_norm(x)

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

        if (
            targets is not None
        ):  # if we are given some desired targets also calculate the loss
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
                    logits = logits.float()  # use tf32/fp32 for logits

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
                        scores = torch.where(
                            scores < 0.3, scores.new_tensor(0.3), scores
                        )
                        scores = torch.sqrt(scores)
                        scores[:, :32] = 1
                        # print(scores[:, 2500:2520])
                        loss = linear_cross_entropy(
                            x,
                            self.lm_head.weight,
                            targets,
                            reduction="none",
                            ignore_index=-1,
                        )
                        loss = loss * scores
                        # print(loss)
                        loss = torch.mean(loss.view(1, -1))
                        # print(loss)
                    elif self.config.scores_loss_coupling == "soft-rho1":
                        x = x.to(torch.bfloat16)
                        scores = scores.to(torch.bfloat16)

                        unscaled_loss = linear_cross_entropy(
                            x,
                            self.lm_head.weight,
                            targets,
                            reduction="none",
                            ignore_index=-1,
                        )

                        alpha = 0.5
                        smooth_weights = torch.sigmoid(alpha * (scores - scores.mean()))

                        loss = (
                            unscaled_loss * smooth_weights
                        ).sum() / smooth_weights.sum()
                    else:
                        raise ValueError(
                            f"Unknown scores_loss_coupling {self.config.scores_loss_coupling}"
                        )

                elif self.config.fused_loss_computation:
                    criterion = FusedLinearCrossEntropyLoss(ignore_index=-1)
                    loss = criterion(x, targets, self.lm_head.weight)
                else:
                    logits = self.lm_head(x)
                    logits = logits.float()  # use tf32/fp32 for logits
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-1,
                    )
            return loss

        elif caches is None:  # inference without caching (not recommended)
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            logits = logits.float()  # use tf32/fp32 for logits
            return logits
        else:  # inference
            logits = self.lm_head(x)
            logits = logits.float()  # use tf32/fp32 for logits
            return logits, caches
