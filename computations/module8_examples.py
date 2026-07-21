"""
Module 8: Natural Language Processing - Numerical Examples

This script demonstrates key concepts from Module 8 with concrete calculations.
Run with: pixi run python course-companion-computations/module8_examples.py

References in Course Companion:
- demo_tfidf_by_hand()              -> Module 8, Section 8.1 (TF-IDF)
- demo_bow_sparsity()               -> Module 8, Section 8.1 (BoW Dimensionality)
- demo_embedding_similarity()       -> Module 8, Section 8.1 (Word Embeddings)
- demo_vanishing_gradient()         -> Module 8, Section 8.2 (Vanishing Gradient)
- demo_rnn_hidden_state()           -> Module 8, Section 8.2 (RNN Hidden State)
- demo_self_attention()             -> Module 8, Section 8.3 (Self-Attention)
- demo_positional_encoding()        -> Module 8, Section 8.3 (Positional Encoding)
- demo_bert_vs_gpt_scale()          -> Module 8, Section 8.4 (Model Comparison)

Last updated: 2026-01-02
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def demo_tfidf_by_hand():
    """
    Calculate TF-IDF step by step to show how rare words get higher weights.
    Demonstrates the dampening effect of log on document frequency.
    """
    print("=" * 60)
    print("TF-IDF CALCULATION BY HAND")
    print("=" * 60)

    # Small corpus
    corpus = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the cat chased the dog",
        "bankruptcy filing announced today",
    ]
    n_docs = len(corpus)

    print(f"\nCorpus ({n_docs} documents):")
    for i, doc in enumerate(corpus):
        print(f"  Doc {i+1}: \"{doc}\"")

    # Words to analyze
    words = ["the", "cat", "bankruptcy"]

    print(f"\nStep-by-step TF-IDF for selected words in Doc 1:")
    print("-" * 50)

    doc1_words = corpus[0].split()
    doc1_word_count = len(doc1_words)

    for word in words:
        # Term frequency in doc 1
        tf = doc1_words.count(word) / doc1_word_count

        # Document frequency (how many docs contain the word)
        df = sum(1 for doc in corpus if word in doc.split())

        # IDF with standard formula: log(N/df) + 1 (sklearn default adds 1)
        idf = np.log(n_docs / df) + 1

        # TF-IDF
        tfidf = tf * idf

        print(f"\n  Word: \"{word}\"")
        print(f"    TF (in Doc 1): {doc1_words.count(word)}/{doc1_word_count} = {tf:.3f}")
        print(f"    DF (docs containing word): {df}/{n_docs}")
        print(f"    IDF: log({n_docs}/{df}) + 1 = {idf:.3f}")
        print(f"    TF-IDF: {tf:.3f} × {idf:.3f} = {tfidf:.3f}")

    # Verify with sklearn
    print(f"\n" + "-" * 50)
    print("Verification with sklearn TfidfVectorizer:")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    print(f"\n  Doc 1 TF-IDF values (normalized):")
    doc1_tfidf = X[0].toarray()[0]
    for word in words:
        if word in feature_names:
            idx = list(feature_names).index(word)
            print(f"    \"{word}\": {doc1_tfidf[idx]:.3f}")

    print(f"\nKey insight:")
    print(f"  - 'the' appears in ALL docs → low IDF → low TF-IDF")
    print(f"  - 'bankruptcy' appears in 1 doc → high IDF → high TF-IDF")
    print(f"  - TF-IDF downweights common words, upweights distinctive ones")


def demo_bow_sparsity():
    """
    Demonstrate the sparsity problem with Bag of Words representations.
    Compare 10,000-dim sparse BoW to 300-dim dense embeddings.
    """
    print("\n" + "=" * 60)
    print("BAG OF WORDS SPARSITY PROBLEM")
    print("=" * 60)

    # Simulate vocabulary size
    vocab_size = 10000
    embedding_dim = 300

    # A typical sentence
    sentence = "I love machine learning"
    n_words = len(sentence.split())

    print(f"\nSentence: \"{sentence}\" ({n_words} words)")
    print(f"\nRepresentation comparison:")
    print("-" * 50)

    # BoW representation
    bow_nonzero = n_words  # At most n unique words
    bow_zeros = vocab_size - bow_nonzero
    bow_sparsity = bow_zeros / vocab_size * 100

    print(f"\nBag of Words (vocab = {vocab_size:,}):")
    print(f"  Vector dimension: {vocab_size:,}")
    print(f"  Non-zero entries: {bow_nonzero}")
    print(f"  Zero entries: {bow_zeros:,}")
    print(f"  Sparsity: {bow_sparsity:.2f}%")
    print(f"  Memory (float32): {vocab_size * 4 / 1024:.1f} KB per document")

    # Embedding representation (average of word embeddings)
    print(f"\nWord Embeddings (dim = {embedding_dim}):")
    print(f"  Vector dimension: {embedding_dim}")
    print(f"  Non-zero entries: {embedding_dim} (dense)")
    print(f"  Zero entries: 0")
    print(f"  Sparsity: 0%")
    print(f"  Memory (float32): {embedding_dim * 4 / 1024:.1f} KB per document")

    # Comparison
    compression = vocab_size / embedding_dim
    print(f"\nComparison:")
    print(f"  Dimensionality reduction: {compression:.0f}x ({vocab_size:,} → {embedding_dim})")
    print(f"  Memory savings: {(1 - embedding_dim/vocab_size) * 100:.1f}%")

    print(f"\nKey insight:")
    print(f"  - BoW creates huge sparse vectors (mostly zeros)")
    print(f"  - Embeddings are dense and much smaller")
    print(f"  - Embeddings also capture semantic similarity (BoW doesn't)")


def demo_embedding_similarity():
    """
    Demonstrate how word embeddings capture semantic similarity using cosine similarity.
    Uses synthetic embeddings to show the concept.
    """
    print("\n" + "=" * 60)
    print("EMBEDDING SIMILARITY")
    print("=" * 60)

    np.random.seed(42)

    # Create synthetic embeddings that demonstrate relationships
    # We'll create a "gender" direction and a "royalty" direction
    dim = 50

    # Base vectors (random)
    base = {
        "man": np.random.randn(dim),
        "woman": np.random.randn(dim),
        "king": np.random.randn(dim),
        "queen": np.random.randn(dim),
        "banana": np.random.randn(dim),
        "apple": np.random.randn(dim),
    }

    # Make gender pairs similar
    gender_direction = np.random.randn(dim) * 0.5
    royalty_direction = np.random.randn(dim) * 0.5

    embeddings = {
        "man": base["man"],
        "woman": base["man"] + gender_direction,  # man + gender = woman
        "king": base["man"] + royalty_direction,  # man + royalty = king
        "queen": base["man"] + gender_direction + royalty_direction,  # king + gender = queen
        "banana": base["banana"],
        "apple": base["banana"] + np.random.randn(dim) * 0.3,  # Similar to banana (fruit)
    }

    # Normalize all embeddings
    for word in embeddings:
        embeddings[word] = embeddings[word] / np.linalg.norm(embeddings[word])

    def cosine_similarity(a, b):
        return np.dot(a, b)  # Already normalized

    print(f"\nCosine similarity between word pairs:")
    print("-" * 40)

    pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("king", "man"),
        ("king", "banana"),
        ("apple", "banana"),
    ]

    for w1, w2 in pairs:
        sim = cosine_similarity(embeddings[w1], embeddings[w2])
        print(f"  {w1:8} ↔ {w2:8}: {sim:+.3f}")

    # Demonstrate king - man + woman ≈ queen
    print(f"\n" + "-" * 40)
    print("Analogy: king - man + woman = ?")

    result = embeddings["king"] - embeddings["man"] + embeddings["woman"]
    result = result / np.linalg.norm(result)  # Normalize

    print(f"\nSimilarity of (king - man + woman) to each word:")
    for word, emb in embeddings.items():
        sim = cosine_similarity(result, emb)
        marker = " ← closest!" if word == "queen" else ""
        print(f"  {word:8}: {sim:+.3f}{marker}")

    print(f"\nKey insight:")
    print(f"  - Similar words have high cosine similarity (close to 1)")
    print(f"  - Unrelated words have low similarity (close to 0)")
    print(f"  - Vector arithmetic captures relationships: king - man + woman ≈ queen")


def demo_vanishing_gradient():
    """
    Demonstrate why gradients vanish in deep/long sequences.
    Shows multiplicative decay through many timesteps.
    """
    print("\n" + "=" * 60)
    print("VANISHING GRADIENT PROBLEM")
    print("=" * 60)

    print(f"\nIn backpropagation through time, gradients are multiplied at each step.")
    print(f"If each multiplication factor < 1, gradients shrink exponentially.\n")

    # Typical gradient scaling factors
    factors = [0.9, 0.5, 0.25]
    timesteps = [10, 50, 100]

    print(f"Gradient decay with different scaling factors:")
    print("-" * 55)
    print(f"{'Timesteps':>12}", end="")
    for f in factors:
        print(f"{'factor=' + str(f):>14}", end="")
    print()
    print("-" * 55)

    for t in timesteps:
        print(f"{t:>12}", end="")
        for f in factors:
            remaining = f ** t
            if remaining < 0.0001:
                print(f"{'< 0.0001':>14}", end="")
            else:
                print(f"{remaining:>14.6f}", end="")
        print()

    # Detailed walkthrough for 0.9^100
    print(f"\n" + "-" * 55)
    print(f"Detailed: How 0.9^100 vanishes")
    print("-" * 55)

    factor = 0.9
    milestones = [1, 10, 25, 50, 75, 100]

    for t in milestones:
        grad = factor ** t
        print(f"  After {t:>3} timesteps: 0.9^{t:<3} = {grad:.10f}")

    print(f"\nInterpretation:")
    print(f"  - At timestep 100, gradient from word 1 is reduced to {0.9**100 * 100:.4f}% of its original value")
    print(f"  - The network effectively 'forgets' early inputs")
    print(f"  - tanh derivative max is 1, but average is ~0.5-0.7 in practice")
    print(f"  - This is why standard RNNs struggle with long sequences")


def demo_rnn_hidden_state():
    """
    Step through RNN hidden state updates on a simple example.
    Shows how information flows through the sequence.
    """
    print("\n" + "=" * 60)
    print("RNN HIDDEN STATE EVOLUTION")
    print("=" * 60)

    np.random.seed(42)

    # Simplified RNN: h_t = tanh(W_xh * x_t + W_hh * h_{t-1})
    # Using small dimensions for clarity

    input_dim = 4  # Word embedding dimension
    hidden_dim = 3  # Hidden state dimension

    # Initialize weights (small for stability)
    W_xh = np.random.randn(hidden_dim, input_dim) * 0.5
    W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.5

    # Simulated word embeddings for "I love ML"
    words = ["I", "love", "ML"]
    embeddings = {
        "I": np.array([0.2, -0.1, 0.3, 0.1]),
        "love": np.array([0.8, 0.5, -0.2, 0.3]),
        "ML": np.array([0.1, 0.4, 0.6, -0.1]),
    }

    print(f"\nProcessing sequence: {' → '.join(words)}")
    print(f"Input dimension: {input_dim}, Hidden dimension: {hidden_dim}")
    print("-" * 55)

    h = np.zeros(hidden_dim)  # Initial hidden state

    for t, word in enumerate(words):
        x = embeddings[word]

        # RNN update
        h_input = W_xh @ x + W_hh @ h
        h_new = np.tanh(h_input)

        print(f"\nTimestep {t+1}: \"{word}\"")
        print(f"  Input x_{t+1}:     [{', '.join(f'{v:+.2f}' for v in x)}]")
        print(f"  Previous h_{t}:   [{', '.join(f'{v:+.2f}' for v in h)}]")
        print(f"  W_xh @ x:         [{', '.join(f'{v:+.2f}' for v in W_xh @ x)}]")
        print(f"  W_hh @ h:         [{', '.join(f'{v:+.2f}' for v in W_hh @ h)}]")
        print(f"  Pre-activation:   [{', '.join(f'{v:+.2f}' for v in h_input)}]")
        print(f"  New h_{t+1} (tanh): [{', '.join(f'{v:+.2f}' for v in h_new)}]")

        h = h_new

    print(f"\n" + "-" * 55)
    print(f"Final hidden state encodes the entire sequence.")
    print(f"Information from 'I' has been transformed through 3 steps.")
    print(f"\nKey insight:")
    print(f"  - Each hidden state combines current input + previous state")
    print(f"  - tanh squashes values to [-1, 1]")
    print(f"  - Same weights W_xh, W_hh used at every timestep (weight sharing)")


def demo_self_attention():
    """
    Compute self-attention step by step for a small example.
    Shows Q, K, V computation and attention weights.
    """
    print("\n" + "=" * 60)
    print("SELF-ATTENTION STEP BY STEP")
    print("=" * 60)

    np.random.seed(42)

    # Simple 3-word sentence with 4-dim embeddings
    words = ["The", "cat", "sat"]
    d_model = 4  # Embedding dimension
    d_k = 3  # Query/Key dimension

    # Word embeddings (rows)
    X = np.array([
        [0.1, 0.2, 0.3, 0.4],   # The
        [0.5, 0.6, -0.2, 0.1],  # cat
        [0.2, -0.1, 0.4, 0.3],  # sat
    ])

    # Projection matrices (simplified, normally learned)
    W_Q = np.random.randn(d_model, d_k) * 0.5
    W_K = np.random.randn(d_model, d_k) * 0.5
    W_V = np.random.randn(d_model, d_k) * 0.5

    print(f"\nInput: \"{' '.join(words)}\"")
    print(f"Embedding dimension: {d_model}, Query/Key dimension: {d_k}")
    print("-" * 55)

    # Step 1: Compute Q, K, V
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    print(f"\nStep 1: Compute Q, K, V (X @ W_Q, X @ W_K, X @ W_V)")
    print(f"\n  Q (queries):")
    for i, word in enumerate(words):
        print(f"    {word:4}: [{', '.join(f'{v:+.2f}' for v in Q[i])}]")

    print(f"\n  K (keys):")
    for i, word in enumerate(words):
        print(f"    {word:4}: [{', '.join(f'{v:+.2f}' for v in K[i])}]")

    # Step 2: Compute attention scores (Q @ K^T)
    scores = Q @ K.T
    print(f"\nStep 2: Attention scores (Q @ K^T)")
    print(f"  Raw scores matrix:")
    print(f"         {'   '.join(words)}")
    for i, word in enumerate(words):
        print(f"    {word:4}: [{', '.join(f'{v:+.2f}' for v in scores[i])}]")

    # Step 3: Scale by sqrt(d_k)
    scale = np.sqrt(d_k)
    scaled_scores = scores / scale
    print(f"\nStep 3: Scale by √d_k = √{d_k} = {scale:.2f}")
    print(f"  Scaled scores:")
    for i, word in enumerate(words):
        print(f"    {word:4}: [{', '.join(f'{v:+.2f}' for v in scaled_scores[i])}]")

    # Step 4: Softmax
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    attention_weights = softmax(scaled_scores)
    print(f"\nStep 4: Softmax (normalize to probabilities)")
    print(f"  Attention weights:")
    print(f"         {'   '.join(words)}")
    for i, word in enumerate(words):
        print(f"    {word:4}: [{', '.join(f'{v:.2f}' for v in attention_weights[i])}]")

    # Step 5: Weighted sum of values
    output = attention_weights @ V
    print(f"\nStep 5: Output = Attention weights @ V")
    print(f"  Output vectors:")
    for i, word in enumerate(words):
        print(f"    {word:4}: [{', '.join(f'{v:+.2f}' for v in output[i])}]")

    print(f"\nInterpretation:")
    print(f"  - Each word's output is a weighted combination of all values")
    print(f"  - Attention weights show what each word 'looks at'")
    cat_attends_to = words[np.argmax(attention_weights[1])]
    print(f"  - 'cat' attends most strongly to '{cat_attends_to}'")


def demo_positional_encoding():
    """
    Visualize positional encoding patterns.
    Shows how different dimensions encode position at different frequencies.
    """
    print("\n" + "=" * 60)
    print("POSITIONAL ENCODING PATTERNS")
    print("=" * 60)

    d_model = 8  # Small for visualization
    max_pos = 10

    print(f"\nPositional encoding formula:")
    print(f"  PE(pos, 2i)   = sin(pos / 10000^(2i/d))")
    print(f"  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))")
    print(f"\nWith d_model={d_model}, showing positions 0-{max_pos-1}")
    print("-" * 70)

    # Compute positional encodings
    PE = np.zeros((max_pos, d_model))
    for pos in range(max_pos):
        for i in range(d_model // 2):
            denominator = 10000 ** (2 * i / d_model)
            PE[pos, 2*i] = np.sin(pos / denominator)
            PE[pos, 2*i + 1] = np.cos(pos / denominator)

    # Show wavelengths for each dimension pair
    print(f"\nWavelength for each dimension pair:")
    for i in range(d_model // 2):
        wavelength = 2 * np.pi * (10000 ** (2 * i / d_model))
        print(f"  Dims {2*i},{2*i+1}: wavelength ≈ {wavelength:.1f} positions")

    # Show encoding values
    print(f"\nPositional encoding values (dimensions 0-3 only):")
    print(f"{'Pos':>4} | {'dim0':>7} {'dim1':>7} {'dim2':>7} {'dim3':>7}")
    print("-" * 45)
    for pos in range(max_pos):
        values = " ".join(f"{PE[pos, d]:+.3f}" for d in range(4))
        print(f"{pos:>4} | {values}")

    # Show how position differences are encoded
    print(f"\n" + "-" * 70)
    print(f"Key property: Relative positions can be computed from encodings")
    print(f"\nDistance between position encodings (Euclidean):")

    for delta in [1, 2, 5]:
        distances = []
        for pos in range(max_pos - delta):
            dist = np.linalg.norm(PE[pos + delta] - PE[pos])
            distances.append(dist)
        avg_dist = np.mean(distances)
        print(f"  Positions {delta} apart: avg distance = {avg_dist:.3f}")

    print(f"\nKey insight:")
    print(f"  - Low dimensions: high frequency (change rapidly with position)")
    print(f"  - High dimensions: low frequency (change slowly)")
    print(f"  - Like clock hands: seconds change fast, hours change slow")
    print(f"  - Model can learn to attend to relative positions")


def demo_bert_vs_gpt_scale():
    """
    Compare BERT and GPT model scales and their implications.
    """
    print("\n" + "=" * 60)
    print("BERT VS GPT: SCALE AND COST COMPARISON")
    print("=" * 60)

    models = {
        "BERT-base": {
            "params": 110_000_000,
            "layers": 12,
            "hidden": 768,
            "heads": 12,
            "context": 512,
            "type": "Encoder",
        },
        "BERT-large": {
            "params": 340_000_000,
            "layers": 24,
            "hidden": 1024,
            "heads": 16,
            "context": 512,
            "type": "Encoder",
        },
        "GPT-2": {
            "params": 1_500_000_000,
            "layers": 48,
            "hidden": 1600,
            "heads": 25,
            "context": 1024,
            "type": "Decoder",
        },
        "GPT-3": {
            "params": 175_000_000_000,
            "layers": 96,
            "hidden": 12288,
            "heads": 96,
            "context": 2048,
            "type": "Decoder",
        },
    }

    print(f"\nModel size comparison:")
    print("-" * 70)
    print(f"{'Model':<12} {'Type':<10} {'Parameters':>15} {'Layers':>8} {'Hidden':>8}")
    print("-" * 70)

    for name, specs in models.items():
        params_str = f"{specs['params']/1e9:.1f}B" if specs['params'] >= 1e9 else f"{specs['params']/1e6:.0f}M"
        print(f"{name:<12} {specs['type']:<10} {params_str:>15} {specs['layers']:>8} {specs['hidden']:>8}")

    # Cost comparison (rough estimates)
    print(f"\n" + "-" * 70)
    print(f"Inference cost comparison (rough estimates):")
    print("-" * 70)

    bert_base_cost = 1  # Baseline
    print(f"  BERT-base (110M):  {bert_base_cost}x (baseline)")
    print(f"  BERT-large (340M): ~3x BERT-base")
    print(f"  GPT-2 (1.5B):      ~14x BERT-base")
    print(f"  GPT-3 (175B):      ~1600x BERT-base")

    # When to use which
    print(f"\n" + "-" * 70)
    print(f"Decision guide:")
    print("-" * 70)

    print(f"""
  Use BERT (encoder) when:
    ✓ Classification, NER, sentiment analysis
    ✓ You have labeled training data
    ✓ Need consistent, deterministic outputs
    ✓ Cost/latency matters (can run locally)
    ✓ Domain-specific fine-tuning needed

  Use GPT (decoder) when:
    ✓ Text generation, summarization
    ✓ Zero/few-shot learning (no training data)
    ✓ Conversational AI, chatbots
    ✓ Creative tasks (marketing copy, etc.)
    ✓ Rapid prototyping before fine-tuning
""")

    print(f"Key insight:")
    print(f"  - BERT: Smaller, cheaper, better for understanding tasks with labeled data")
    print(f"  - GPT: Larger, expensive, better for generation and zero-shot tasks")
    print(f"  - Production systems often use fine-tuned BERT for cost efficiency")


if __name__ == "__main__":
    demo_tfidf_by_hand()
    demo_bow_sparsity()
    demo_embedding_similarity()
    demo_vanishing_gradient()
    demo_rnn_hidden_state()
    demo_self_attention()
    demo_positional_encoding()
    demo_bert_vs_gpt_scale()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
