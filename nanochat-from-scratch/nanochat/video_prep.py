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
    sequence_length = 1024
    vocab_size = 50304
    n_layer = 12
    # query heads
    n_head = 6 
    # kv heads (make this less for GQA or 1 for MQA)
    n_kv_head = 6
    model_dim = 768
    head_dim = model_dim/n_head

def norm(x):
    # rms norm just normalizes along the model dimension, which means it takes the
    # mean across all model dims and normalizes each token by that
    return F.rms_norm(x, (x.size(-1),))

def apply_rope(x, rope):
    """
    x:   (b, n, dim)
    rope: (b, n, dim/2, 2, 2)
    returns rotated x, same shape (b, n, dim)
    """
    b, n, _ = x.shape
    # reshape x → (b, n, half, 2)
    x2 = x.view(b, n, -1, 2)

    # apply 2×2 rotation:
    x0 = x2[..., 0]   # (b,n,half)
    x1 = x2[..., 1]   # (b,n,half)

    r00 = rope[..., 0, 0]
    r01 = rope[..., 0, 1]
    r10 = rope[..., 1, 0]
    r11 = rope[..., 1, 1]

    y0 = r00 * x0 + r01 * x1
    y1 = r10 * x0 + r11 * x1

    # combine back to dim
    y = torch.stack([y0, y1], dim=-1).reshape(b, n, -1)

    return y


class CausalSelfAttention(nn.Module):
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
    
    def forward(self, x, cos_sin, kv_cache):
        # x: (B, T, C)
        # B: batch size
        # T: sequence length
        # C: embedding size (n_embd)
        # q: (B, T, n_head, head_dim)
        # k, v: (B, T, n_kv_head, head_dim)
        B, T, _ = x.size()

        q = self.q(x).view(B, T, self.n_head, self.head_dim)
        k = self.k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v(x).view(B, T, self.n_kv_head, self.head_dim)

        if kv_cache:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)
        Tkv = k.size(2)



# We inherit from nn.Module so that all submodules (like our Linear layers) are properly registered
# as part of the module. This enables features like automatic device transfer, parameter management,
# and saving/loading via .state_dict().
class MLP(nn.Module):
    '''inherits from nn.Module'''
    def __init__(self, config):
        # call nn.Module init method
        super().__init__()
        self.up_proj = nn.Linear(4 * config.model_dim, config.model_dim, bias=False)
        self.down_proj = nn.Linear(config.model_dim, 4 * config.model_dim, bias=False)

    def forward(self, x):
        return self.down_proj((F.relu(self.up_proj(x))).square())

class Block(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        # self
        self.mlp = MLP(config)
        self.attn = CausalSelfAttention(config, layer_idx)

    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.model_dim),
            "h": nn.ModuleList([Block(config, layer) for layer in range(self.config.n_layer)])
        })
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
    
    def _precompute_rotary_embeddings(self, multiplier, theta=10000.0, device=None):
        """
        returns rotation matrices of shape (b, n, h, dim/2, 2, 2)
        """
        if device is None:
            device = self.wte.weight.device
        assert self.config.model_dim % 2 == 0
        half = self.config.model_dim // 2          
        # number of rotary pairs

        # frequencies: shape (half,)
        scale = torch.arange(0, half, device=device, dtype=torch.float32)
        inv_theta = theta ** (-scale / half)
        positions = torch.arange(multiplier*self.config.seq_len, dtype=torch.float32, device=device)
        radians = torch.outer(positions, inv_theta)
        cos, sin = radians.cos(), radians.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        # Build 2×2 rotation matrix for each pair
        # (cos, -sin; sin, cos)
        rope = torch.stack([
            torch.stack([cos, -sin], dim=-1),   # (n,half,2)
            torch.stack([sin,  cos], dim=-1),   # (n,half,2)
        ], dim=-2)                               # → (n,half,2,2)
        return rope

    def __init_weights(self):
        self.apply(self._init_weights)
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            '''
            Zeroing only down_proj makes the MLP's output initially zero, ensuring no contribution from untrained blocks at the start. up_proj is left untouched so the internal activations can still learn and propagate gradients. This helps stabilize early training.
            '''
            torch.nn.init.zeros_(block.mlp.down_proj)
            torch.nn.init.zeros_(block.attn.down_proj)
        # self.cos, self.sin = 
        if self.transfoerm.wte.weight.device.type == "cude":
            self.transformer.wte.to(dtype=torch.bfloat16)


# @torch.inference_mode()