import torch
import torch.nn as nn
from modules import TransformerBlock

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, emb, heads, num_layers, seq_len, ff_dim):
        super().__init__()
        self.emb = emb
        self.token_embedding = nn.Embedding(vocab_size, emb)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, emb))  # Learned positional encoding
        self.layers = nn.ModuleList([TransformerBlock(emb, heads, mask=True, ff_dim=ff_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(emb)
        self.head = nn.Linear(emb, vocab_size)

    def forward(self, x):
        # Token embedding
        x = self.token_embedding(x)  # Shape: (batch_size, seq_len, emb)

        # Add positional encoding
        x = x + self.positional_encoding  # Shape: (batch_size, seq_len, emb)

        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x)  # Shape: (batch_size, seq_len, emb)

        # Final layer normalization
        x = self.norm(x)  # Shape: (batch_size, seq_len, emb)

        # Model head (output logits for final token)
        logits = self.head(x[:, -1, :])  # Shape: (batch_size, vocab_size)

        return logits