import torch
import math
from torch.nn import Linear
from torch.nn import Module
from torch import nn
import torch.nn as nn


linear = Linear(3, 3)
inputs = torch.rand(3, 3)

a= linear(inputs)
# tensor([[-0.4848,  0.4174, -0.1633],
#     [-0.4428,  0.5404, -0.1470],
#     [-0.7272,  0.0770,  0.0321]], grad_fn=<AddmmBackward0)
b = (inputs @ linear.weight.T).add(linear.bias)
# tensor([[-0.4848,  0.4174, -0.1633],
#     [-0.4428,  0.5404, -0.1470],
#     [-0.7272,  0.0770,  0.0321]], grad_fn=<AddBackward0)
print(a,b)
i = torch.rand(1,3)
layer = Linear(3, 2, False)
print(layer.weight)
print(layer(i))
x = torch.randn(1, requires_grad=True) + torch.randn(1)
print(x)
y = torch.randn(2, requires_grad=True).sum()
print(y)



class PSE(Module):

    def __init__(self, model_dim, seq_len):
        self.pe = torch.zeros(seq_len, model_dim)
        positions = torch.arrange(0, seq_len)
        div_term = torch.exp(torch.arrange(0,model_dim,2)*-torch.log(10000)/model_dim)
        self.pe[0::2] = torch.sin(positions*div_term)
        self.pe[1::2] = torch.cos(positions*div_term)
        self.register_buffer('pe', self.pe)

    def forward(self, input):
        '''
        accepts an input of the form (batch, sequence, model_dim)
        returns output of the same form w/ positional embedding
        '''
        # go up to the size of the input vector
        input = input + self.pe[:input.size(1)]

class LN(Module):
    
    def __init__(self, emb, eps=10E-5):
        self.emb = emb
        self.gamma = torch.ones(emb)
        self.beta = torch.zeros(emb)
        self.eps = eps

    def forward(self, inputs):
        mean = inputs.mean(dim=-1, keep_dim=True)
        var = ((inputs - mean)**2).mean(dim=-1, keep_dim=True)
        std = (var + self.eps).sqrt()
        out = (inputs - mean)/std
        y = out*self.gamma + self.beta

        return y


class TransformerBlock(Module):

    def __init__(self, mask, heads, model_dim, ff_dim, seq_len):
        super().__init__()

        self.attention = MHSA(model_dim, heads, mask, seq_len)

        self.l1 = LN(model_dim)

        self.ff = torch.nn.Sequential(
            Linear(model_dim, ff_dim),
            torch.nn.ReLU(),
            Linear(ff_dim, model_dim)
        )

        self.l2 = LN(model_dim)

    def forward(self, input):

        attention = self.attention(self.l1(input))
        x = input + attention
        ff = self.ff(self.l2(x))
        return x + ff
        attention = self.attention(self.l1(input))  # LayerNorm before attention
        x = attention + input  # Residual connection
        ff = self.ff(self.l2(x))  # LayerNorm before feed-forward
        return ff + x  # Residual connection
    
class AGT(Module):
    def __init__(self, seq_len, model_dim, n_blocks, n_heads, ff_dim, v_size, mask=True):
        super().__init__()
        self.norm = LN(model_dim)
        self.blocks = torch.nn.ModuleList(TransformerBlock(mask, n_heads, model_dim, ff_dim, seq_len) for _ in range(n_blocks))
        self.seq_len = seq_len
        self.v_size = v_size
        self.model_dim = model_dim
        self.head = Linear(model_dim, v_size)

    def forward(self, x):

        x = torch.nn.Embedding(self.seq_len, self.model_dim)
        x = PSE(self.model_dim, self.seq_len)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # norm gives tensor of batch, seq len, model_dim
        # we want to pass the last sequence of every batch through the head
        logits = self.head(x[:,-1,:])
        # shape is now batch_size by model_dim
        return logits


class MHSA(Module):
    def __init__(self, mask, emb, heads, seq_len) -> None:
        super().__init__()
        self.mask = mask
        self.heads = heads

        self.head_dim = emb//heads
        # NB the integer division
        assert emb == self.head_dim * heads

        self.to_keys = Linear(emb, emb, bias=False)
        self.to_queries = Linear(emb, emb, bias=False)
        self.to_values = Linear(emb, emb, bias=False)
        self.unify = Linear(emb, emb)
        self.mask_matrix = torch.triu(torch.ones((seq_len,seq_len)), diagonal=1).bool().to(x.device)

    def forward(self, x):
        batch_size, seq_len, model_dim = x.size()
        # we want to get batch x head x seq_len x head_model_dim
        keys = self.to_keys(x)
        values = self.to_values(x)
        queries = self.to_queries(x)
        matrices = [keys, values, queries]

        # still at batch x head x model_dim, but it's with the down projected attention heads
        # previous implementation where I did matrix = matrix change was naive and doesn't work
        # because for matrix in matrices, creates a temp variable for matrices, matrix does not refer to
        # [keys, values, queries], it's just a temporary instance
        for i in range(len(matrices)):
            matrices[i] = matrices[i].view(batch_size, seq_len, self.heads, self.head_dim)
        # transpose to get batch_size, self.heads, seq_len, self.head_dim
        for i in range(len(matrices)):
            matrices[i] = matrices[i].transpose(1, 2)
        keys, values, queries = matrices
        # now we can calculate the attention score matrix: softmax(QK^T)
        # queries and keys are of shape batch_size, self.heads, seq_len, self.head_dim
        # need to transpose seq_len and head dim in keys matrix
        # we want to get shape batch_size, self.heads, seq_len, seq_len, so there is an attention score for each token
        attention_score = queries @ keys.transpose(2,3)
        attention_score = attention_score/ torch.sqrt(torch.tensor(self.head_dim, dtype=float))
        # apply token mask before softmax
        if self.mask:
            # The mask is created on the CPU by default, but attention_score might 
            # be on a GPU. You need to move the mask to the same device as attention_score:
            attention_score = attention_score.masked_fill(self.mask_matrix, float('-inf'))
        # attention = F.softmax(dot, dim=-1)
        attention_score = torch.softmax(attention_score, dim=-1)
        # attention_score has shape shape batch_size, self.heads, seq_len, self.head_dim, we need to get back to
        # batch x head x model_dim
        context_matrix = attention_score @ values

        context_matrix = context_matrix.transpose(1,2).contiguous()
        # contiguous is required for view()
        out = context_matrix.view(batch_size, seq_len, model_dim)

        return self.unify(out)
    




def sinusoidal_encoding_2(position, d_model):
    pe = torch.zeros(position, d_model)
    for i in range(position):
        for k in range(0, d_model, 2):
            pe[i, k] = math.sin(i / (10000 ** (k / d_model)))
            pe[i, k+1] = math.cos(i / (10000 ** (k / d_model)))
    return pe