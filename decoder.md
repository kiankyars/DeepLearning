2. Steps to Implement a Decoder-Only Transformer

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