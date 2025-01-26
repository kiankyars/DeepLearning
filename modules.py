import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """
        The MultiHeadSelfAttention class implements multi-head self-attention, which computes relationships between all elements in a sequence.

        Args:
            emb (int): The embedding dimension of the input.
            heads (int): The number of attention heads.
            mask (bool): Whether to apply a causal mask to prevent attending to future tokens.
        """
        super().__init__()
        self.emb = emb  # Embedding dimension
        self.heads = heads  # Number of attention heads
        self.mask = mask  # Whether to mask future tokens

        # Each head will work with a smaller embedding dimension
        self.head_dim = emb // heads
        assert self.head_dim * heads == emb, "Embedding dimension must be divisible by the number of heads."

        # Linear layers to project input into queries, keys, and values
        # it's emb to emb because it's multi-head attention, if we used single-head attention,
        # then we would go from emb to emb/h, but we have 8 Q, W, K matrices stacked together
        self.to_queries = nn.Linear(emb, emb, bias=False)  # Projects input to queries
        self.to_keys = nn.Linear(emb, emb, bias=False)  # Projects input to keys
        self.to_values = nn.Linear(emb, emb, bias=False)  # Projects input to values

        # Linear layer to unify the outputs of all heads
        self.unify_heads = nn.Linear(emb, emb)

    def forward(self, x):
        """
        Forward pass for multi-head self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        batch_size, seq_len, emb_dim = x.size()

        # Project input into queries, keys, and values
        queries = self.to_queries(x)  # Shape: (batch_size, seq_len, emb_dim)
        keys = self.to_keys(x)  # Shape: (batch_size, seq_len, emb_dim)
        values = self.to_values(x)  # Shape: (batch_size, seq_len, emb_dim)

        # Reshape queries, keys, and values for multi-head attention
        # Split the embedding dimension into `heads` separate heads
        # we now have batch_size number of 3-D (sequence, heads, head_dim tensors)
        queries = queries.view(batch_size, seq_len, self.heads, self.head_dim)  # Shape: (batch_size, seq_len, heads, head_dim)
        keys = keys.view(batch_size, seq_len, self.heads, self.head_dim)  # Shape: (batch_size, seq_len, heads, head_dim)
        values = values.view(batch_size, seq_len, self.heads, self.head_dim)  # Shape: (batch_size, seq_len, heads, head_dim)

        # Swap dimensions to fold heads into the batch dimension
        # This allows parallel computation of attention for all heads
        # Now, the heads dimension is treated as part of the batch dimension, allowing us to process all heads in parallel.
        queries = queries.transpose(1, 2)  # Shape: (batch_size, heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # Shape: (batch_size, heads, seq_len, head_dim)
        values = values.transpose(1, 2)  # Shape: (batch_size, heads, seq_len, head_dim)

        # Compute scaled dot-product attention
        # Attention(Q, K, V) = softmax(Q @ K^T / sqrt(head_dim)) @ V
        dot = torch.matmul(queries, keys.transpose(-2, -1))  # Shape: (batch_size, heads, seq_len, seq_len)
        dot = dot / math.sqrt(self.head_dim)  # Scale by sqrt(head_dim)

        # Apply masking (if enabled) to prevent attending to future tokens
        if self.mask:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)  # Upper triangular mask
            dot = dot.masked_fill(mask, float('-inf'))  # Mask out future tokens

        '''
        Apply softmax to get attention probabilities
        The softmax is applied along the last dimension (dim=-1), which corresponds to the sequence length (seq_len).
        This ensures that the attention scores for each query sum to 1, representing a probability distribution over the keys.
        Example:
        The dot product tensor has shape (2, 8, 10, 10).
        After applying softmax, the shape remains (2, 8, 10, 10), but each row in the last dimension now sums to 1.
        '''
        attention = F.softmax(dot, dim=-1)  # Shape: (batch_size, heads, seq_len, seq_len)

        # Apply attention to values
        # Attention(Q, K, V) = attention-score matrix @ V (weighted average)
        out = torch.matmul(attention, values)  # Shape: (batch_size, heads, seq_len, head_dim)

        # Swap dimensions back and unify heads
        out = out.transpose(1, 2).contiguous()  # Shape: (batch_size, seq_len, heads, head_dim)
        out = out.view(batch_size, seq_len, emb_dim)  # Shape: (batch_size, seq_len, emb_dim)

        # Unify the outputs of all heads
        out = self.unify_heads(out)  # Shape: (batch_size, seq_len, emb_dim)

        return out
    

class TransformerBlock(nn.Module):
    """
    The TransformerBlock class combines self-attention with a feed-forward network, using residual connections and layer normalization.
    """
    def __init__(self, emb, heads=8, mask=False, ff_dim=2048):
        """
        Args:
            emb (int): The embedding dimension of the input.
            heads (int): The number of attention heads.
            mask (bool): Whether to apply a causal mask in self-attention.
            ff_dim (int): The hidden dimension of the feed-forward network.
        """
        super().__init__()

        # Multi-head self-attention layer
        self.attention = MultiHeadSelfAttention(emb, heads, mask)

        # Layer normalization for the attention output
        self.norm1 = LayerNormalization(emb)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(emb, ff_dim),  # Expand dimension
            nn.ReLU(),  # Non-linearity
            nn.Linear(ff_dim, emb)  # Project back to embedding dimension
        )

        # Layer normalization for the feed-forward output
        self.norm2 = LayerNormalization(emb)

    def forward(self, x):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        # Self-attention with residual connection
        attention_output = self.attention(x)  # Shape: (batch_size, seq_len, emb_dim)
        x = x + attention_output  # Add residual connection
        x = self.norm1(x)  # Apply layer normalization

        # Feed-forward network with residual connection
        ff_output = self.feed_forward(x)  # Shape: (batch_size, seq_len, emb_dim)
        x = x + ff_output  # Add residual connection
        x = self.norm2(x)  # Apply layer normalization

        return x
    

class LayerNormalization:
    def __init__(self, parameters_shape, eps=1e-5):
        """
        Args:
            parameters_shape (tuple): The shape of the input tensor's last dimensions.
            eps (float): A small value to avoid division by zero.
        """
        # 512 if we use that as our d_model
        self.parameters_shape = parameters_shape  # Shape of the last dimensions (e.g., embedding dimension)
        self.eps = eps  # Small constant for numerical stability
        self.gamma = nn.Parameter(torch.ones(parameters_shape))  # Learnable scaling parameter
        self.beta = nn.Parameter(torch.zeros(parameters_shape))  # Learnable shifting parameter

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, ..., parameters_shape).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as inputs.
        """
        # Compute mean and variance over the last dimensions
        mean = inputs.mean(dim=-1, keepdim=True)  # Mean over the last dimensions
        var = ((inputs - mean) ** 2).mean(dim=-1, keepdim=True)  # Variance over the last dimensions
        std = (var + self.eps).sqrt()  # Standard deviation with numerical stability

        # Normalize the input
        y = (inputs - mean) / std  # Normalized tensor

        # Apply learnable scaling (gamma) and shifting (beta)
        out = self.gamma * y + self.beta  # Scale and shift

        return out


# Example input
batch_size = 2
seq_len = 10
emb_dim = 512
x = torch.randn(batch_size, seq_len, emb_dim)  # Random input tensor

# Create a Transformer block
transformer_block = TransformerBlock(emb=emb_dim, heads=8, mask=False)

# Forward pass
output = transformer_block(x)
print(output.shape)  # Output shape: (batch_size, seq_len, emb_dim)