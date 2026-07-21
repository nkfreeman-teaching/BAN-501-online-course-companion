"""
Deep Dive: Transformer Architecture - Numerical Examples

This script demonstrates key concepts from the Transformer Architecture Deep Dive
with concrete calculations.
Run with: pixi run python course-companion-computations/deep_dive_transformer_examples.py

References in Course Companion:
- demo_positional_encoding()    -> Deep Dive Transformer, Positional Encoding section
- demo_embedding_lookup()       -> Deep Dive Transformer, Token Embedding section
- demo_scaling_effect()         -> Deep Dive Transformer, Attention Scaling section
- demo_attention_scores()       -> Deep Dive Transformer, Self-Attention section
- demo_multihead_reshape()      -> Deep Dive Transformer, Multi-Head Attention section
- demo_ffn_forward()            -> Deep Dive Transformer, FFN section
- demo_layer_norm()             -> Deep Dive Transformer, Layer Normalization section
- demo_causal_masking()         -> Deep Dive Transformer, Causal Masking section

Last updated: 2026-01-02
"""

import numpy as np
import torch
import torch.nn as nn


def demo_positional_encoding():
    """
    Show how sinusoidal positional encoding creates unique, bounded values per position.
    Demonstrates the interleaving of sine and cosine functions.
    """
    print("=" * 60)
    print("POSITIONAL ENCODING VALUES")
    print("=" * 60)

    d_model = 8
    positions = [0, 1, 10, 100]

    def get_positional_encoding(pos, d_model):
        """Compute sinusoidal PE for a single position."""
        pe = np.zeros(d_model)
        for i in range(d_model):
            if i % 2 == 0:
                # Even indices: sine
                pe[i] = np.sin(pos / (10000 ** (i / d_model)))
            else:
                # Odd indices: cosine
                pe[i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
        return pe

    print(f"\nPositional encodings for d_model={d_model}")
    print(f"Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))")
    print(f"         PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))")

    print(f"\n{'Position':>10}", end="")
    for i in range(d_model):
        func = "sin" if i % 2 == 0 else "cos"
        print(f"{f'dim{i}({func})':>10}", end="")
    print()
    print("-" * (10 + 10 * d_model))

    for pos in positions:
        pe = get_positional_encoding(pos, d_model)
        print(f"{pos:>10}", end="")
        for val in pe:
            print(f"{val:>10.4f}", end="")
        print()

    print(f"\nKey observations:")
    print(f"  - All values bounded between -1 and 1 (sin/cos range)")
    print(f"  - Position 0 has a distinctive pattern (0, 1, 0, 1, ...)")
    print(f"  - Each position has a unique encoding")
    print(f"  - Lower dimensions change faster (higher frequency)")
    print(f"  - Higher dimensions change slower (lower frequency)")


def demo_embedding_lookup():
    """
    Show how token IDs map to embedding vectors via table lookup.
    Demystifies the embedding layer as simple indexing.
    """
    print("\n" + "=" * 60)
    print("EMBEDDING LOOKUP")
    print("=" * 60)

    vocab_size = 10
    d_model = 4

    # Create embedding with fixed seed for reproducibility
    torch.manual_seed(42)
    embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=d_model,
    )

    # Sample token IDs
    token_ids = torch.tensor([2, 5, 7])

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Embedding dimension: {d_model}")
    print(f"Token IDs to look up: {token_ids.tolist()}")

    print(f"\nEmbedding matrix (first 5 rows):")
    print(f"{'Row':>5}", end="")
    for i in range(d_model):
        print(f"{'dim' + str(i):>10}", end="")
    print()
    print("-" * (5 + 10 * d_model))

    for i in range(5):
        print(f"{i:>5}", end="")
        for val in embedding.weight[i].detach().numpy():
            print(f"{val:>10.4f}", end="")
        print()

    print(f"\nLooking up token_ids {token_ids.tolist()}:")
    output = embedding(token_ids)

    print(f"\n{'Token ID':>10} {'Embedding Vector':>40}")
    print("-" * 55)
    for tid, vec in zip(token_ids.tolist(), output.detach().numpy()):
        vec_str = "[" + ", ".join(f"{v:.4f}" for v in vec) + "]"
        print(f"{tid:>10} {vec_str:>40}")

    print(f"\nVerification: embedding(2) == embedding.weight[2]")
    print(f"  embedding(2):       {output[0].detach().numpy()}")
    print(f"  embedding.weight[2]: {embedding.weight[2].detach().numpy()}")
    print(f"  Match: {torch.allclose(output[0], embedding.weight[2])}")


def demo_scaling_effect():
    """
    Show why scaling by sqrt(d_k) prevents softmax saturation.
    Compares attention distributions with and without scaling.
    """
    print("\n" + "=" * 60)
    print("SCALING EFFECT ON SOFTMAX")
    print("=" * 60)

    def softmax(x):
        """Compute softmax along last dimension."""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum()

    # Typical dot product magnitudes scale with sqrt(d_k)
    # With d_k=64, dot products might be around 8 (sqrt(64))
    # With d_k=512, dot products might be around 22 (sqrt(512))

    # Simulate unscaled attention scores for d_k=64
    d_k_small = 64
    raw_scores_small = np.array([8.0, 4.0, 2.0, 1.0])  # Typical magnitudes

    # Simulate unscaled attention scores for d_k=512
    d_k_large = 512
    raw_scores_large = raw_scores_small * np.sqrt(d_k_large / d_k_small)  # Scale up

    print(f"\nScenario: 4 tokens, comparing d_k={d_k_small} vs d_k={d_k_large}")
    print(f"\nRaw attention scores (before scaling):")
    print(f"  d_k={d_k_small}: {raw_scores_small}")
    print(f"  d_k={d_k_large}: {raw_scores_large.round(2)}")

    # Without scaling
    print(f"\nSoftmax WITHOUT scaling:")
    attn_unscaled_small = softmax(raw_scores_small)
    attn_unscaled_large = softmax(raw_scores_large)
    print(f"  d_k={d_k_small}: {np.round(attn_unscaled_small, 4)}")
    print(f"  d_k={d_k_large}: {np.round(attn_unscaled_large, 4)}")

    # With scaling by sqrt(d_k)
    print(f"\nSoftmax WITH scaling by sqrt(d_k):")
    scaled_small = raw_scores_small / np.sqrt(d_k_small)
    scaled_large = raw_scores_large / np.sqrt(d_k_large)
    attn_scaled_small = softmax(scaled_small)
    attn_scaled_large = softmax(scaled_large)
    print(f"  d_k={d_k_small}: {np.round(attn_scaled_small, 4)}")
    print(f"  d_k={d_k_large}: {np.round(attn_scaled_large, 4)}")

    print(f"\nKey observations:")
    print(f"  - Without scaling, large d_k produces near-one-hot distribution")
    print(f"  - d_k={d_k_large} unscaled: max weight = {attn_unscaled_large.max():.4f} (nearly all attention on one token)")
    print(f"  - With scaling, distributions are similar regardless of d_k")
    print(f"  - Scaled distributions maintain smoother gradients during training")

    # Show entropy as measure of "peakiness"
    def entropy(p):
        return -np.sum(p * np.log(p + 1e-10))

    print(f"\nEntropy (higher = more uniform):")
    print(f"  Unscaled d_k={d_k_small}: {entropy(attn_unscaled_small):.4f}")
    print(f"  Unscaled d_k={d_k_large}: {entropy(attn_unscaled_large):.4f}")
    print(f"  Scaled d_k={d_k_small}:   {entropy(attn_scaled_small):.4f}")
    print(f"  Scaled d_k={d_k_large}:   {entropy(attn_scaled_large):.4f}")


def demo_attention_scores():
    """
    Compute Q @ K^T step by step for a 4-token sequence.
    Shows how attention scores are computed from query-key dot products.
    """
    print("\n" + "=" * 60)
    print("ATTENTION SCORES COMPUTATION")
    print("=" * 60)

    np.random.seed(42)

    seq_len = 4
    d_k = 4

    # Create Q and K matrices (seq_len x d_k)
    Q = np.array([
        [1.0, 0.0, 1.0, 0.0],   # Token 0: "The"
        [0.5, 0.5, 0.0, 1.0],   # Token 1: "cat"
        [0.0, 1.0, 0.5, 0.5],   # Token 2: "sat"
        [1.0, 1.0, 0.0, 0.0],   # Token 3: "down"
    ])

    K = np.array([
        [1.0, 0.0, 0.5, 0.5],   # Token 0: "The"
        [0.0, 1.0, 0.0, 1.0],   # Token 1: "cat"
        [0.5, 0.5, 1.0, 0.0],   # Token 2: "sat"
        [0.0, 0.0, 1.0, 1.0],   # Token 3: "down"
    ])

    print(f"\nSequence: ['The', 'cat', 'sat', 'down'] (4 tokens)")
    print(f"d_k = {d_k}")

    print(f"\nQuery matrix Q ({seq_len}x{d_k}):")
    for i, row in enumerate(Q):
        print(f"  Token {i}: {row}")

    print(f"\nKey matrix K ({seq_len}x{d_k}):")
    for i, row in enumerate(K):
        print(f"  Token {i}: {row}")

    # Compute Q @ K^T
    scores = Q @ K.T
    print(f"\nRaw attention scores Q @ K^T ({seq_len}x{seq_len}):")
    print(f"         Token_0  Token_1  Token_2  Token_3")
    for i, row in enumerate(scores):
        print(f"Token_{i}: {row[0]:>7.2f} {row[1]:>7.2f} {row[2]:>7.2f} {row[3]:>7.2f}")

    # Scale by sqrt(d_k)
    scaled_scores = scores / np.sqrt(d_k)
    print(f"\nScaled scores (/ sqrt({d_k}) = / {np.sqrt(d_k):.2f}):")
    print(f"         Token_0  Token_1  Token_2  Token_3")
    for i, row in enumerate(scaled_scores):
        print(f"Token_{i}: {row[0]:>7.2f} {row[1]:>7.2f} {row[2]:>7.2f} {row[3]:>7.2f}")

    # Apply softmax row-wise
    def softmax_rows(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    attention_weights = softmax_rows(scaled_scores)
    print(f"\nAttention weights (softmax of scaled scores):")
    print(f"         Token_0  Token_1  Token_2  Token_3   Sum")
    for i, row in enumerate(attention_weights):
        print(f"Token_{i}: {row[0]:>7.3f} {row[1]:>7.3f} {row[2]:>7.3f} {row[3]:>7.3f}  {row.sum():.3f}")

    print(f"\nInterpretation:")
    print(f"  - Row i shows how much token i attends to each other token")
    print(f"  - Each row sums to 1 (probability distribution)")
    print(f"  - Token 0 ('The') attends most to Token 2 ('sat'): {attention_weights[0, 2]:.3f}")
    print(f"  - Token 1 ('cat') attends most to Token 3 ('down'): {attention_weights[1, 3]:.3f}")


def demo_multihead_reshape():
    """
    Visualize how tensors are reshaped for multi-head attention.
    Shows the split from (batch, seq, d_model) to (batch, heads, seq, d_head).
    """
    print("\n" + "=" * 60)
    print("MULTI-HEAD ATTENTION RESHAPE")
    print("=" * 60)

    batch_size = 1
    seq_len = 4
    d_model = 8
    n_heads = 2
    d_head = d_model // n_heads

    print(f"\nConfiguration:")
    print(f"  batch_size = {batch_size}")
    print(f"  seq_len = {seq_len}")
    print(f"  d_model = {d_model}")
    print(f"  n_heads = {n_heads}")
    print(f"  d_head = d_model / n_heads = {d_head}")

    # Create input tensor with recognizable values
    # Each position gets values like [0.0, 0.1, 0.2, ..., 0.7] for token 0
    x = torch.zeros(batch_size, seq_len, d_model)
    for t in range(seq_len):
        for d in range(d_model):
            x[0, t, d] = t + d * 0.1

    print(f"\nInput shape: {tuple(x.shape)} (batch, seq, d_model)")
    print(f"\nInput tensor values:")
    print(f"  Token 0: {x[0, 0].numpy().round(1)}")
    print(f"  Token 1: {x[0, 1].numpy().round(1)}")
    print(f"  Token 2: {x[0, 2].numpy().round(1)}")
    print(f"  Token 3: {x[0, 3].numpy().round(1)}")

    # Step 1: Reshape to (batch, seq, n_heads, d_head)
    x_reshaped = x.view(batch_size, seq_len, n_heads, d_head)
    print(f"\nStep 1: Reshape to (batch, seq, n_heads, d_head)")
    print(f"  New shape: {tuple(x_reshaped.shape)}")
    print(f"  Token 0, Head 0: {x_reshaped[0, 0, 0].numpy().round(1)}")
    print(f"  Token 0, Head 1: {x_reshaped[0, 0, 1].numpy().round(1)}")

    # Step 2: Transpose to (batch, n_heads, seq, d_head)
    x_transposed = x_reshaped.transpose(1, 2)
    print(f"\nStep 2: Transpose dims 1 and 2 to (batch, n_heads, seq, d_head)")
    print(f"  New shape: {tuple(x_transposed.shape)}")

    print(f"\nHead 0 (processes dims 0-3 of each token):")
    for t in range(seq_len):
        print(f"  Token {t}: {x_transposed[0, 0, t].numpy().round(1)}")

    print(f"\nHead 1 (processes dims 4-7 of each token):")
    for t in range(seq_len):
        print(f"  Token {t}: {x_transposed[0, 1, t].numpy().round(1)}")

    print(f"\nKey insight:")
    print(f"  - Each head processes a different 'slice' of the embedding")
    print(f"  - Head 0 sees dimensions [0, 1, 2, 3]")
    print(f"  - Head 1 sees dimensions [4, 5, 6, 7]")
    print(f"  - Heads can learn different attention patterns in parallel")


def demo_ffn_forward():
    """
    Show FFN expansion and contraction with ReLU activation.
    Demonstrates the information bottleneck and sparsification.
    """
    print("\n" + "=" * 60)
    print("FEED-FORWARD NETWORK FORWARD PASS")
    print("=" * 60)

    d_model = 4
    d_ff = 16  # 4x expansion
    expansion_factor = d_ff / d_model

    print(f"\nFFN Configuration:")
    print(f"  d_model = {d_model}")
    print(f"  d_ff = {d_ff} ({expansion_factor}x expansion)")
    print(f"  Architecture: Linear({d_model} -> {d_ff}) -> ReLU -> Linear({d_ff} -> {d_model})")

    # Create FFN with fixed weights for reproducibility
    torch.manual_seed(42)
    W1 = torch.randn(d_ff, d_model) * 0.5
    b1 = torch.zeros(d_ff)
    W2 = torch.randn(d_model, d_ff) * 0.5
    b2 = torch.zeros(d_model)

    # Input vector
    x = torch.tensor([1.0, -0.5, 0.8, 0.2])

    print(f"\nInput vector: {x.numpy()}")

    # Step 1: First linear layer
    h1 = x @ W1.T + b1
    print(f"\nStep 1: h1 = x @ W1^T + b1")
    print(f"  Shape: ({d_model},) @ ({d_model}, {d_ff})^T = ({d_ff},)")
    print(f"  h1 (first 8 values): {h1[:8].detach().numpy().round(3)}")
    print(f"  h1 (last 8 values):  {h1[8:].detach().numpy().round(3)}")

    # Step 2: ReLU activation
    h2 = torch.relu(h1)
    n_positive = (h2 > 0).sum().item()
    print(f"\nStep 2: h2 = ReLU(h1)")
    print(f"  h2 (first 8 values): {h2[:8].detach().numpy().round(3)}")
    print(f"  h2 (last 8 values):  {h2[8:].detach().numpy().round(3)}")
    print(f"  Active neurons: {n_positive}/{d_ff} ({100*n_positive/d_ff:.0f}%)")
    print(f"  Zeros from ReLU: {d_ff - n_positive}/{d_ff} ({100*(d_ff-n_positive)/d_ff:.0f}%)")

    # Step 3: Second linear layer
    output = h2 @ W2.T + b2
    print(f"\nStep 3: output = h2 @ W2^T + b2")
    print(f"  Shape: ({d_ff},) @ ({d_ff}, {d_model})^T = ({d_model},)")
    print(f"  output: {output.detach().numpy().round(3)}")

    print(f"\nDimension journey: {d_model} -> {d_ff} -> {d_model}")
    print(f"\nWhy this works:")
    print(f"  - Expansion to {d_ff} creates rich intermediate representation")
    print(f"  - ReLU zeros out ~50% of neurons (sparse activation)")
    print(f"  - Different inputs activate different neuron subsets")
    print(f"  - Compression back to {d_model} forces learning useful features")


def demo_layer_norm():
    """
    Show layer normalization with learned scale and shift.
    Demonstrates normalization to mean=0, std=1, then rescaling.
    """
    print("\n" + "=" * 60)
    print("LAYER NORMALIZATION")
    print("=" * 60)

    # Input with varying magnitudes
    x = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # Shape: (1, 4)
    d_model = 4

    print(f"\nInput: {x[0].numpy()}")
    print(f"  Mean: {x.mean().item():.2f}")
    print(f"  Std:  {x.std(unbiased=False).item():.2f}")

    # Step 1: Compute mean and variance
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    std = torch.sqrt(var + 1e-6)

    print(f"\nStep 1: Compute statistics along d_model dimension")
    print(f"  mean = {mean.item():.2f}")
    print(f"  var = {var.item():.2f}")
    print(f"  std = {std.item():.2f}")

    # Step 2: Normalize to mean=0, std=1
    x_norm = (x - mean) / std

    print(f"\nStep 2: Normalize: x_norm = (x - mean) / std")
    print(f"  x_norm: {x_norm[0].detach().numpy().round(4)}")
    print(f"  x_norm mean: {x_norm.mean().item():.6f} (should be ~0)")
    print(f"  x_norm std:  {x_norm.std(unbiased=False).item():.6f} (should be ~1)")

    # Step 3: Apply learned scale (gamma) and shift (beta)
    gamma = torch.tensor([1.0, 2.0, 0.5, 1.5])  # Learned scale
    beta = torch.tensor([0.0, 1.0, -0.5, 0.0])  # Learned shift

    print(f"\nStep 3: Apply learned parameters")
    print(f"  gamma (scale): {gamma.numpy()}")
    print(f"  beta (shift):  {beta.numpy()}")

    output = gamma * x_norm + beta

    print(f"\nOutput = gamma * x_norm + beta")
    print(f"  output: {output[0].detach().numpy().round(4)}")

    # Verify with PyTorch LayerNorm
    print(f"\nVerification with nn.LayerNorm:")
    layer_norm = nn.LayerNorm(
        normalized_shape=d_model,
        elementwise_affine=True,
    )
    # Set custom gamma and beta
    with torch.no_grad():
        layer_norm.weight.copy_(gamma)
        layer_norm.bias.copy_(beta)

    pytorch_output = layer_norm(x)
    print(f"  PyTorch output: {pytorch_output[0].detach().numpy().round(4)}")
    print(f"  Match: {torch.allclose(output, pytorch_output, atol=1e-4)}")

    print(f"\nWhy LayerNorm:")
    print(f"  - Normalizes across d_model (not across batch)")
    print(f"  - Stabilizes hidden state magnitudes during training")
    print(f"  - Learned gamma/beta allow model to undo normalization if needed")


def demo_causal_masking():
    """
    Show how causal masking enforces autoregressive attention.
    Demonstrates -inf masking and its effect on softmax.
    """
    print("\n" + "=" * 60)
    print("CAUSAL MASKING")
    print("=" * 60)

    seq_len = 4

    # Create example attention scores (before softmax)
    np.random.seed(42)
    scores = np.random.randn(seq_len, seq_len).round(2)

    print(f"\nSequence: ['The', 'cat', 'sat', 'down'] (4 tokens)")
    print(f"\nRaw attention scores (before masking):")
    print(f"         Token_0  Token_1  Token_2  Token_3")
    for i, row in enumerate(scores):
        print(f"Token_{i}: {row[0]:>7.2f} {row[1]:>7.2f} {row[2]:>7.2f} {row[3]:>7.2f}")

    # Create causal mask (upper triangular = -inf)
    mask = np.triu(np.ones((seq_len, seq_len)) * float('-inf'), k=1)

    print(f"\nCausal mask (upper triangle = -inf):")
    print(f"         Token_0  Token_1  Token_2  Token_3")
    for i, row in enumerate(mask):
        row_str = [f"{v:>7.2f}" if v != float('-inf') else "   -inf" for v in row]
        print(f"Token_{i}: {row_str[0]} {row_str[1]} {row_str[2]} {row_str[3]}")

    # Apply mask
    masked_scores = scores + mask

    print(f"\nMasked scores (scores + mask):")
    print(f"         Token_0  Token_1  Token_2  Token_3")
    for i, row in enumerate(masked_scores):
        row_str = [f"{v:>7.2f}" if v != float('-inf') else "   -inf" for v in row]
        print(f"Token_{i}: {row_str[0]} {row_str[1]} {row_str[2]} {row_str[3]}")

    # Apply softmax row-wise
    def softmax_rows(x):
        # Handle -inf by setting exp(-inf) = 0
        exp_x = np.exp(np.where(x == float('-inf'), -1e9, x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    attention_weights = softmax_rows(masked_scores)

    print(f"\nAttention weights after softmax:")
    print(f"         Token_0  Token_1  Token_2  Token_3   Sum")
    for i, row in enumerate(attention_weights):
        print(f"Token_{i}: {row[0]:>7.3f} {row[1]:>7.3f} {row[2]:>7.3f} {row[3]:>7.3f}  {row.sum():.3f}")

    print(f"\nInterpretation:")
    print(f"  - Token 0 ('The') can only attend to itself")
    print(f"  - Token 1 ('cat') can attend to tokens 0-1")
    print(f"  - Token 2 ('sat') can attend to tokens 0-2")
    print(f"  - Token 3 ('down') can attend to all tokens 0-3")

    print(f"\nWhy this matters for generation:")
    print(f"  - When predicting token 2, model can't 'peek' at tokens 3+")
    print(f"  - Enforces left-to-right autoregressive generation")
    print(f"  - Same mask used during training and inference")
    print(f"  - exp(-inf) = 0, so future tokens get zero attention weight")


if __name__ == "__main__":
    demo_positional_encoding()
    demo_embedding_lookup()
    demo_scaling_effect()
    demo_attention_scores()
    demo_multihead_reshape()
    demo_ffn_forward()
    demo_layer_norm()
    demo_causal_masking()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
