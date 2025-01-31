import torch
import torch.nn as nn
from modules import TransformerBlock, SinusoidalPositionalEncoding, LayerNormalization

# Example input
batch_size = 2
seq_len = 10
x = torch.randint(0, 10000, (batch_size, seq_len))  # Random token IDs (vocab_size=10000)

class AutoregressiveTransformer(nn.Module):
    '''
    Key Components:

    Token Embedding: Maps input tokens to dense vectors.
    Positional Encoding: Adds positional information to embeddings.
    Transformer Blocks: Stack of num_blocks transformer blocks.
    Final LayerNorm: Normalizes the output of the last transformer block.
    Model Head: Projects the final token's embedding to logits.
    '''
    def __init__(self, vocab_size, emb, heads, num_blocks, seq_len, ff_dim):
        super().__init__()
        self.emb = emb
        self.token_embedding = nn.Embedding(vocab_size, emb)
        self.positional_encoding = SinusoidalPositionalEncoding(emb, seq_len)  # Learned positional encoding
        self.blocks = nn.ModuleList([TransformerBlock(emb, heads, mask=True, ff_dim=ff_dim) for _ in range(num_blocks)])
        self.norm = LayerNormalization(emb)
        self.head = nn.Linear(emb, vocab_size)

    def forward(self, x):
        # Token embedding
        x = self.token_embedding(x)  # Shape: (batch_size, seq_len, emb)

        # Add positional encoding
        x = self.positional_encoding(x)  # Shape: (batch_size, seq_len, emb)

        # Pass through transformer blocks
        for layer in self.blocks:
            x = layer(x)  # Shape: (batch_size, seq_len, emb)

        # Final layer normalization
        x = self.norm(x)  # Shape: (batch_size, seq_len, emb)

        # Model head (output logits for final token)
        logits = self.head(x[:, -1, :])  # Shape: (batch_size, vocab_size)

        return logits

# Create the model
vocab_size = 10000
num_blocks = 6
heads = 8
ff_dim = 2048
model_dim = 512
model = AutoregressiveTransformer(vocab_size, model_dim, heads, num_blocks, seq_len, ff_dim)

# Forward pass
logits = model(x)
print(logits.shape)  # Output shape: (batch_size, vocab_size)