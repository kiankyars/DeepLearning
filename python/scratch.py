from torch import nn
from torch.nn import Module
# import torch.nn as nn
import torch

class Decoder(Module):
    def __init__(self, ff_dim, heads, seq_len, num_blocks, mask, mod_dim, v_size):
        super().__init__()
        self.embedding = nn.Embedding(v_size, mod_dim)
        self.pe = SPE(seq_len, mod_dim)
        self.blocks = nn.ModuleList(Transformer(heads, mod_dim, seq_len, ff_dim, mask) for _ in range(num_blocks))
        self.norm = LN(mod_dim)
        self.head = nn.Linear(seq_len, v_size)
        self.softmax = nn.Softmax(v_size)
    def forward(self, x):
        x = self.embedding(x)
        x = self.pe(x)
        for layer in self.blocks:
            x = layer(x)
        x = self.norm(x)
        # batch, seq_len, mod_dim, we want the last seq_len for every batch, keeping the contextualized representation (mod_dim) 
        logits = self.head(x[:-1:])
        # if you want to return the token probabilities
        # return self.softmax(logits)
        return logits

class SPE(Module):
    def __init__(self, seq_len, mod_dim):
        super().__init__()
        # position vector
        positions = torch.zeros(seq_len)
        # embedding vector
        div_term = torch.exp(torch.arange(0, mod_dim, 2)*-torch.log10(1000)/mod_dim)
        pe = torch.zeros(seq_len, mod_dim)
        pe[0::2] = torch.sin(positions*div_term)
        pe[1::2] = torch.cos(positions*div_term)
        torch.register_buffer('pe', pe)
    def forward(self, x):
        # adds self.pe to x up to the seq_len dimension, and all mod_dims
        return x + self.pe[x.size(1):]

class LN(Module):
    def __init__(self, mod_dim, eps=10E-5):
        # get last dimension of batch (model_dimension)
        super().__init__()
        self.beta = torch.ones(mod_dim)
        self.gamma = torch.zeros(mod_dim)
        self.eps = eps

    def forward(self, x):
        # accept batch of (batch_size, seq_len, model_dim)
        # return normalized version of same tuple
        # this is why we need keep_dim, so tensors are of shape
        # (b,s,1), then we can broadcast x-mean with no problem
        mean = torch.mean(x,-1,keepdim=True)
        var = ((x-mean)**2).mean(-1,keepdim=True)
        std = torch.sqrt(var + self.eps)
        y = (x-mean)/std
        return y*self.gamma + self.beta

class Transformer(Module):
    def __init__(self, heads, mod_dim, seq_len, ff_dim, mask):
        super().__init__()
        self.l1 = LN(mod_dim)
        self.l2 = LN(mod_dim)
        self.attention = MHSA(heads, seq_len, mod_dim, mask)
        self.ff = nn.Sequential(
            nn.Linear(mod_dim,ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, mod_dim)
        )
    def forward(self, x):
        # pre-normalization
        attention = self.attention(self.l1(x))
        x = attention + x
        ff = self.ff(self.l2(x))
        return ff + x

class MHSA(Module):
    def __init__(self, heads, seq_len, mod_dim, mask):
        super().__init__()
        self.tokeys = nn.Linear(mod_dim, mod_dim, bias=False)
        self.tovalues = nn.Linear(mod_dim, mod_dim, bias=False)
        self.toqueries = nn.Linear(mod_dim, mod_dim, bias=False)
        self.unify = nn.Linear(mod_dim, mod_dim)
        self.heads = heads
        self.head_dim = mod_dim // heads
        assert self.head_dim * heads == mod_dim
        if mask:
            self.mask_matrix = torch.triu(torch.zeros((seq_len, seq_len)), diagonal=1).bool()
        self.mask = mask
    def forward(self, x):
        # batch, seq_len, mod_dim is the current dimension of x
        batch, seq_len, mod_dim = x.size()
        queries = self.toqueries(x); values = self.tovalues(x); keys = self.tokeys(x)
        matrices = [keys, queries, values]
        # we want k, q, v to be of dimension batch, head, seq_len, self.head_dim
        for i in range(len(matrices)):
            matrices[i] = matrices[i].view(batch, seq_len, self.heads, self.head_dim)
        """ 
         # - fold heads into the batch dimension
    keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
    queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
    values = values.transpose(1, 2).contiguous().view(b * h, t, s)
You can avoid these calls to contiguous() by using reshape() instead of view() but 
I prefer to make it explicit when we are copying a tensor, and when we are just viewing it.
 See this notebook for an explanation of the difference.
           """
        
        for i in range(len(matrices)):
            # Since the head and batch dimension are not next to each other, we need to transpose before we reshape.
            # (This is costly, but it seems to be unavoidable.)
            matrices[i] = matrices[i].transpose(1,2)
        # now we have batch, head, seq_len, self.head_dim, let's calculate attention scores
        # attention(q,v,t) = (softmax(QK^T)/sqrt(self.head_dim))V
        keys, queries, values = matrices
        attention_scores = queries @ keys.transpose(2,3)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=float))
        if self.mask:
            torch.masked_fill(attention_scores, self.mask_matrix, float('-inf'))

        """ dot = torch.bmm(queries, keys.transpose(1, 2))

        indices = torch.triu_indices(t, t, offset=1)
        dot[:, indices[0], indices[1]] = float('-inf')

        dot = F.softmax(dot, dim=2) """
        weighted_attention_scores = nn.functional.softmax(attention_scores,-1)
        context_matrix = weighted_attention_scores @ values
        # we now have the context matrix in dimension
        # batch, head, seq_len, self.head_dim, want to get back to # batch, seq_len, mod_dim
        context_matrix = context_matrix.transpose(1, 2).contiguous()
        # batch, seq_len, head, self.head_dim
        context_matrix = context_matrix.view(batch, seq_len, mod_dim)
        # batch, seq_len, mod_dim
        # all that's left is to pass through the final linear layer now that all of the heads are calculated!
        return self.unify(context_matrix)