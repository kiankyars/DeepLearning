import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functionl as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from transformers import GPT2TokenizerFast
import time


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb, heads, num_blocks, seq_len, ff_dim) -> None:
        super().__init__()
        self.positional_encoding = SPE(emb, seq_len)
        self.token_embedding = nn.Embedding(vocab_size, emb)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(num_blocks)])
        self.norm = LayerNorm(emb)
        self.head = nn.Linear(emb, vocab_size)

class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask, ff_dim) -> None:
        super().__init__()
        self.attention = MHSA()
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, emb)
        )
        self.norm1 = LayerNorm(emb)
        self.norm2 = LayerNorm(emb)
    def forward(self, x):
        x = x + self.attention(self.norm(x))
        return x + self.ff(self.norm2(x))
    
class SPE(nn.Module):
    def __init__(self, emb, seq_len) -> None:
        super().__init__()
        pe = torch.zeros(seq_len, emb)
        position = torch.arange(0, seq_len, 2, dtype=torch.float)
        div_term = torch.exp(torch.arange(0, emb, 2, dtype=torch.float)*(-torch.log(torch.tensor(1000))/emb))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        self.register('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1)]

        pass
class MHSA(nn.Module):
    def __init__(self, emb, heads, mask, seq_len) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = emb // self.heads
        self.emb = emb
        self.mask = mask
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues = nn.Linear(emb, emb, bias=False)
        self.toekys = nn.Linear(emb, emb, bias=False)
        self.unify = nn.Linear(emb, emb)
        if mask:
            self.register_buffer("mask", torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())
        else:
            self.mask = None
    def forward(self, x):
        b, seq_len, emb = x.size()
        q = self.queries(x).view(b, seq_len, self.heads, self.head_dim).transpose(1,2)
        k = self.keys(x).view(b, seq_len, self.heads, self.head_dim).transpose(1,2)
        v = self.values(x).view(b, seq_len, self.heads, self.head_dim).transpose(1,2)
        attn_scores = (q @ k.tranpose(-2,-1)) / (self.head_dim ** 0.5)
        if self.mask:
            attn_scores = attn_scores.masked_fill(self.mask[])
class LayerNorm(nn.Module):
    def __init__(self, emb, eps=10E-5) -> None:
        super().__init__()
        self.gamma = nn.parameter(torch.ones(emb))
        self.beta = nn.parameter(torch.zeros(emb))
        self.eps=eps
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False))
        return self.gamma * (x-mean)/std + self.beta



class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass