# Deepseek transcript for architecture guidance

Steps to Implement a Decoder-Only Transformer
`
Token Embedding:
Map input tokens (integers) to dense vectors of size emb.
Use nn.Embedding for this.
Positional Encoding:
Add positional information to the token embeddings.
Use sinusoidal positional encoding or learned positional embeddings.
Stack Transformer Blocks:
Stack multiple TransformerBlock layers to form the decoder.
Final Layer Normalization:
Apply layer normalization to the output of the last transformer block.
Model Head:
Use a linear layer to project the final token's embedding to logits (vocabulary size).
Loss Function:
Use cross-entropy loss to compare logits with ground truth tokens.
Training Loop:
Implement the forward pass, compute loss, and update model parameters.


Swapping Dimensions Back and Unifying Heads
`out = out.transpose(1, 2).contiguous()  # Shape: (batch_size, seq_len, heads, head_dim)
out = out.view(batch_size, seq_len, emb_dim)  # Shape: (batch_size, seq_len, emb_dim)`
Intuition:

After computing the attention-weighted values, we need to combine the outputs of all heads into a single representation.
out.transpose(1, 2) swaps the heads and seq_len dimensions, resulting in shape (batch_size, seq_len, heads, head_dim).
.view(batch_size, seq_len, emb_dim) reshapes the tensor by concatenating the heads and head_dim dimensions, resulting in shape (batch_size, seq_len, emb_dim).



Example:

out has shape (2, 8, 10, 64).
After .transpose(1, 2), the shape becomes (2, 10, 8, 64).
After .view(2, 10, 512), the shape becomes (2, 10, 512).
The 8 heads, each with 64 dimensions, are concatenated back into a single 512-dimensional representation.
Summary of Tensor Shapes

Step	                    Tensor Shape
Input (x)	                (batch_size, seq_len, emb_dim)
After linear projections	(batch_size, seq_len, emb_dim)
After reshaping for heads	(batch_size, seq_len, heads, head_dim)
After swapping dimensions	(batch_size, heads, seq_len, head_dim)
After dot product	        (batch_size, heads, seq_len, seq_len)
After softmax	            (batch_size, heads, seq_len, seq_len)
After applying to values	(batch_size, heads, seq_len, head_dim)
After swapping back	        (batch_size, seq_len, heads, head_dim)
After view	                (batch_size, seq_len, emb_dim)
After unifying heads	    (batch_size, seq_len, emb_dim)


Example with concrete numbers:

batch_size = 2
seq_len = 10
emb_dim = 512
heads = 8
head_dim = 64

Input: x has shape                                           (2, 10, 512).
Linear Projections: queries, keys, and values have shape     (2, 10, 512).
Reshape for Heads: queries, keys, and values are reshaped to (2, 10, 8, 64).
Swap Dimensions: queries, keys, and values are transposed to (2, 8, 10, 64).
Dot Product: The dot product results in a tensor of shape    (2, 8, 10, 10).
Softmax: The softmax output has shape                        (2, 8, 10, 10).
Apply to Values: The weighted sum of values has shape        (2, 8, 10, 64).
Swap Back: The output is transposed to                       (2, 10, 8, 64).
Unify Heads: The output is reshaped to                       (2, 10, 512).