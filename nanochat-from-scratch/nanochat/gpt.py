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
import torch.nn.functional as F

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

@torch.compile
def RMS(x, epsilon):
    # x = B x seq_len x emb_dim
    # x_denominator = B x seq_len x 1
    # x / sqrt(E(x^2) + e)
    # sum(x_0^^2…x_n^^2)
    return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)+epsilon)

@torch.compile
def relu(x):
    # x = B x seq_len x emb_dim
    # for i in range(len(x[-1]))
    return x * (x > 0)


class CasualSelfAttention(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.model_dim = config.model_dim
        self.head_dim = self.model_dim // self.n_head
        self.n_kv_head = config.n_kv_head
        assert self.model_dim % self.n_head == 0
        assert self.n_kv_head <= self.n_head
        assert self.n_head % self.n_kv_head == 0
        self.q = nn.Linear(self.model_dim,self.n_head*self.head_dim, bias=False)
        self.k = nn.Linear(self.model_dim,self.n_head*self.head_dim, bias=False)
        self.v = nn.Linear(self.model_dim,self.n_head*self.head_dim, bias=False)
        self.proj = nn.Linear(self.model_dim, self.model_dim, bias=False)

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
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.emb_dim)
        self.blocks = [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
        self.lm_head(config.seq_len, config.vocab_size, bias=False)
        cos, sin = self._precompute_rotary_embeddings(config.seq_len*10, config.head_dim)
        self.register_buffer("cos", cos, persistent=True)
        self.register_buffer("sin", sin, persistent=True)

    def forward(self):
        pass

    def get_device(self):
        return self.wte.weight.device

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.get_device()
        # stride over the channels
        # omega = 1/(theta^(2k/d))
        # 0, 2… 128
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        # shape = (64)
        inv_freq = 1.0 / (base**(2*channel_range/head_dim))
        # shape = (seq_len)
        token_index = torch.arange(seq_len, dtype=torch.float32, device=device)
        # shape = (seq_len x head_dim/2)
        freqs = torch.outer(token_index, inv_freq)
        cos, sin = torch.cos(freqs), torch.sin(freqs)
        # shape = (seq_len x head_dim/2)
        # shape = (b x seq_len x head_dim/2)
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos = torch.unsqueeze(torch.unsqueeze(cos, 0), 2)
        sin = torch.unsqueeze(torch.unsqueeze(sin, 0), 2)
        # shape = (1 x seq_len x 1 x head_dim/2)
        return cos, sin


    @torch.inference_mode
    def generate(self, tokens, max_tokens, temperature=1, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            # x = seq_len x emb_dim
            # emb_dim x vocab_size
            logits = self.forward(ids)
            # Batch x seq_len x vocab_size
            logits = logits[:, -1, :]
            if top_k is not None:
                # Batch x 1 x k
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[:, [-1]], -float('inf'), logits)
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), 1)
            token = next_ids.item()
            yield token
