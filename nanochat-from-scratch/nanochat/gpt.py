"""
SERIES PRINCIPLES
==========================

I'm doing this with you so you can learn through my own problem-solving process.
I will write pseudocode for each function before implementing it.
I demonstrate everything through small examples.
I'm going to do this in one take.

"""

"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass
from turtle import forward

import torch
import torch.nn as nn

from nanochat.common import get_dist_info
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    seq_len = 1024
    vocab_size = 50304
    n_layer = 12
    n_head = 6
    # n_kv_head = 6,3,2,1
    n_kv_head = 6
    emb_dim = 768
    head_dim = emb_dim / n_head

def RMS(x, epsilon):
    # x = B x seq_len x emb_dim
    # x_denominator = B x seq_len x 1
    # x / sqrt(E(x^2) + e)
    # sum(x_0^^2…x_n^^2)
    return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)+epsilon)

def relu(x):
    # x = B x seq_len x emb_dim
    # for i in range(len(x[-1]))
    return x


class CasualSelfAttention(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()

class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.emd_dim, 4 * config.emb_dim, bias=False)
        self.down_proj = nn.Linear(config.emd_dim * 4, config.emb_dim, bias=False)

    def forward(self, x):
        return self.down_proj((relu(self.up_proj(x)).square()))

class Block(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.attn = CasualSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(RMS(x))
        x = x + self.mlp(RMS(x))
        return x

class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config
        self.wte = nn.Embedding(config.vocab_size, config.emb_dim)
        self.blocks = [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
        self.lm_head(config.seq_len, config.vocab_size, bias=False)