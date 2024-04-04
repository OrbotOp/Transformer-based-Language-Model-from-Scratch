import torch
from torch import nn


class SelfAttentionHead(nn.Module):
    """
    Implements a single self-attention head. This module computes attention
    weights and applies these weights to the input sequence to focus on
    relevant parts of the input when generating each part of the output sequence.
    """
    def __init__(self, in_size, out_size):
        super().__init__()
        self.head_size = out_size  # Size of each attention head output.
        # Linear transformations for the input sequence to get keys, queries, and values.
        self.K = nn.Linear(in_size, self.head_size, bias=False)
        self.Q = nn.Linear(in_size, self.head_size, bias=False)
        self.V = nn.Linear(in_size, self.head_size, bias=False)

    def forward(self, x):
        # Compute keys, queries, and values.
        keys = self.K(x)
        queries = self.Q(x)
        keys_t = keys.transpose(1, 2)  # Transpose keys for matrix multiplication with queries.
        # Compute scaled dot-product attention scores.
        autocorrs = (queries @ keys_t) * (self.head_size ** -0.5)
        # Mask out upper triangle so each position can only attend to previous positions.
        autocorrs = torch.tril(autocorrs)
        # Replace masked positions with a very large negative number to get zero in softmax.
        autocorrs = autocorrs.masked_fill(autocorrs == 0, float('-inf'))
        # Apply softmax to get attention weights.
        autocorrs = torch.softmax(autocorrs, dim=-1)
        values = self.V(x)  # Compute values.
        # Apply attention weights to values.
        return autocorrs @ values

class MultiHeadAttention(nn.Module):
    """
    Implements multi-head self-attention by combining multiple SelfAttentionHead
    instances. This allows the model to focus on different parts of the input
    for different tasks (e.g., one head might focus on syntax, another on semantics).
    """
    def __init__(self, head_count, in_size, out_size):
        super().__init__()
        # Create a list (ModuleList) of self-attention heads.
        self.heads = nn.ModuleList([
            SelfAttentionHead(in_size, out_size // head_count) for _ in range(head_count)
        ])
        self.layerNorm = nn.LayerNorm(out_size)  # Layer normalization for stabilizing learning.

    def forward(self, x):
        # Concatenate the output of each head and apply layer normalization.
        return self.layerNorm(torch.cat([head(x) for head in self.heads], dim=-1))

class MLP(nn.Module):
    """
    Implements a simple multilayer perceptron (MLP) with one hidden layer, used
    within the Transformer block to process the output of the self-attention mechanism.
    """
    def __init__(self, embed_size):
        super().__init__()
        # Define the MLP layers.
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),  # Non-linearity
            nn.Linear(embed_size * 4, embed_size)
        )
        self.layerNorm = nn.LayerNorm(embed_size)  # Layer normalization.

    def forward(self, x):
        # Apply MLP and layer normalization.
        return self.layerNorm(self.mlp(x))

class TransformerBlock(nn.Module):
    """
    Implements a Transformer block, which combines multi-head self-attention
    and an MLP. Each part is followed by a residual connection and layer normalization.
    """
    def __init__(self, head_count, in_size, out_size):
        super().__init__()
        # The communication part (multi-head self-attention).
        self.comm = MultiHeadAttention(head_count=head_count, in_size=in_size, out_size=out_size)
        # The thinking part (MLP).
        self.think = MLP(embed_size=out_size)

    def forward(self, x):
        # Apply self-attention and MLP, with residual connections and layer normalization.
        return x + self.think(x + self.comm(x))
