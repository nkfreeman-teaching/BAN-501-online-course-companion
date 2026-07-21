"""
Module 7: Computer Vision & CNNs - Numerical Examples

This script demonstrates key concepts from Module 7 with concrete calculations.
Run with: pixi run python course-companion-computations/module7_examples.py

References in Course Companion:
- demo_fc_parameter_explosion()      -> Module 7, Section 7.1 (Why FC Fails)
- demo_convolution_by_hand()         -> Module 7, Section 7.2 (Convolution Operation)
- demo_output_size_formula()         -> Module 7, Section 7.2 (Output Dimensions)
- demo_cnn_vs_fc_parameters()        -> Module 7, Section 7.2 (Parameter Efficiency)
- demo_pooling_dimension_tracking()  -> Module 7, Section 7.2 (Pooling)
- demo_transfer_learning_comparison() -> Module 7, Section 7.3 (Transfer Learning)

Last updated: 2026-01-02
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def demo_fc_parameter_explosion():
    """
    Show how fully connected networks explode in parameters for images.
    Demonstrates why FC is impractical for image data.
    """
    print("=" * 60)
    print("PARAMETER EXPLOSION IN FULLY CONNECTED NETWORKS")
    print("=" * 60)

    # Different image sizes
    image_configs = [
        ("MNIST (grayscale)", 28, 28, 1),
        ("CIFAR-10 (color)", 32, 32, 3),
        ("Small photo (color)", 224, 224, 3),
        ("HD photo (color)", 1920, 1080, 3),
    ]

    hidden_neurons = 1000  # First hidden layer size

    print(f"\nFirst FC layer with {hidden_neurons:,} hidden neurons:")
    print(f"{'Image Type':<25} {'Dimensions':<20} {'Input Features':<15} {'Parameters':<15}")
    print("-" * 80)

    for name, h, w, c in image_configs:
        input_features = h * w * c
        # Parameters = (input × hidden) + hidden (bias)
        parameters = input_features * hidden_neurons + hidden_neurons
        dim_str = f"{h}×{w}×{c}"
        print(f"{name:<25} {dim_str:<20} {input_features:>13,} {parameters:>13,}")

    print(f"\nInterpretation:")
    print(f"  - MNIST: ~800K parameters - manageable")
    print(f"  - CIFAR-10: ~3M parameters - still okay")
    print(f"  - 224×224 photo: ~150M parameters - first layer alone!")
    print(f"  - HD photo: ~6 BILLION parameters - impossible")
    print(f"\nThis is why fully connected networks don't scale to real images.")


def demo_convolution_by_hand():
    """
    Step through convolution operation on a small image.
    Shows exactly how filter weights combine with image patches.
    """
    print("\n" + "=" * 60)
    print("CONVOLUTION BY HAND")
    print("=" * 60)

    # 5×5 grayscale image (simple gradient)
    image = np.array([
        [10, 10, 10, 10, 10],
        [10, 50, 50, 50, 10],
        [10, 50, 100, 50, 10],
        [10, 50, 50, 50, 10],
        [10, 10, 10, 10, 10],
    ], dtype=float)

    # Horizontal edge detector (Sobel-like)
    filter_h = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ], dtype=float)

    # Vertical edge detector
    filter_v = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], dtype=float)

    print(f"\nInput image (5×5):")
    print(image.astype(int))

    print(f"\nHorizontal edge detector (3×3):")
    print(filter_h.astype(int))

    print(f"\nVertical edge detector (3×3):")
    print(filter_v.astype(int))

    # Manual convolution (no padding, stride=1)
    # Output will be 3×3
    output_h = np.zeros((3, 3))
    output_v = np.zeros((3, 3))

    print(f"\nConvolution process (stride=1, no padding):")
    print(f"Output size: (5-3)/1 + 1 = 3×3")

    # Show one calculation in detail
    print(f"\nDetailed calculation for position (1,1) with horizontal filter:")
    patch = image[1:4, 1:4]
    print(f"Image patch:")
    print(patch.astype(int))
    print(f"\nElement-wise multiply with filter:")
    elementwise = patch * filter_h
    print(elementwise.astype(int))
    result = np.sum(elementwise)
    print(f"\nSum all values: {result:.0f}")

    # Complete convolution
    for i in range(3):
        for j in range(3):
            patch = image[i:i + 3, j:j + 3]
            output_h[i, j] = np.sum(patch * filter_h)
            output_v[i, j] = np.sum(patch * filter_v)

    print(f"\nFull output (horizontal edge detector):")
    print(output_h.astype(int))

    print(f"\nFull output (vertical edge detector):")
    print(output_v.astype(int))

    print(f"\nInterpretation:")
    print(f"  - Horizontal filter: detects top-to-bottom intensity changes")
    print(f"  - Output shows edges where values transition from light to dark")
    print(f"  - The center column of vertical output is 0 (symmetric left/right)")
    print(f"  - Edge detectors respond strongly at boundaries, weakly in uniform regions")


def demo_output_size_formula():
    """
    Apply the output size formula to trace dimensions through a CNN.
    Formula: output = (input - kernel + 2*padding) / stride + 1
    """
    print("\n" + "=" * 60)
    print("OUTPUT SIZE FORMULA IN ACTION")
    print("=" * 60)

    print(f"\nFormula: output = (W - K + 2P) / S + 1")
    print(f"  W = input size, K = kernel size, P = padding, S = stride")

    # Trace through a typical CNN
    layers = [
        ("Input", None, None, None, None),
        ("Conv1 (3×3, pad=1)", 3, 1, 1, "conv"),
        ("MaxPool (2×2)", 2, 0, 2, "pool"),
        ("Conv2 (3×3, pad=1)", 3, 1, 1, "conv"),
        ("MaxPool (2×2)", 2, 0, 2, "pool"),
        ("Conv3 (3×3, pad=1)", 3, 1, 1, "conv"),
        ("MaxPool (2×2)", 2, 0, 2, "pool"),
    ]

    input_size = 32  # Starting with 32×32 image
    current_size = input_size

    print(f"\nTracing a 32×32 image through a CNN:")
    print(f"{'Layer':<25} {'Calculation':<30} {'Output Size':<15}")
    print("-" * 70)

    for name, k, p, s, layer_type in layers:
        if layer_type is None:
            print(f"{'Input':<25} {'':<30} {current_size}×{current_size}")
        else:
            calc = f"({current_size} - {k} + 2×{p}) / {s} + 1"
            new_size = (current_size - k + 2 * p) // s + 1
            print(f"{name:<25} {calc:<30} {new_size}×{new_size}")
            current_size = new_size

    print(f"\nKey observations:")
    print(f"  - Conv with padding='same' (P=K//2): preserves spatial dimensions")
    print(f"  - MaxPool with stride=2: halves spatial dimensions")
    print(f"  - 32×32 → 16×16 → 8×8 → 4×4 after 3 pooling layers")
    print(f"  - Final 4×4 feature maps are flattened for the classifier")

    # Show effect of different padding
    print(f"\n--- Effect of padding on a 32×32 input with 3×3 conv ---")
    print(f"{'Padding':<15} {'Calculation':<25} {'Output':<10}")
    print("-" * 55)
    for p in [0, 1, 2]:
        calc = f"(32 - 3 + 2×{p}) / 1 + 1"
        out = (32 - 3 + 2 * p) // 1 + 1
        print(f"P = {p:<11} {calc:<25} {out}×{out}")


def demo_cnn_vs_fc_parameters():
    """
    Compare parameter counts between CNN and FC for the same task.
    Shows the ~100x efficiency of convolutional layers.
    """
    print("\n" + "=" * 60)
    print("CNN VS FULLY CONNECTED: PARAMETER COMPARISON")
    print("=" * 60)

    # Task: 32×32×3 RGB image → 64 output features
    input_h, input_w, input_c = 32, 32, 3
    output_channels = 64

    # Fully connected approach
    fc_input = input_h * input_w * input_c  # 3072
    fc_weights = fc_input * output_channels
    fc_bias = output_channels
    fc_total = fc_weights + fc_bias

    # CNN approach: 3×3 conv
    kernel_size = 3
    conv_weights = kernel_size * kernel_size * input_c * output_channels  # 3×3×3×64
    conv_bias = output_channels
    conv_total = conv_weights + conv_bias

    print(f"\nTask: Map {input_h}×{input_w}×{input_c} input to {output_channels} features")
    print(f"\n{'Approach':<25} {'Weights':<15} {'Bias':<10} {'Total':<15}")
    print("-" * 65)
    print(f"{'Fully Connected':<25} {fc_weights:>13,} {fc_bias:>8,} {fc_total:>13,}")
    print(f"{'Conv2d (3×3)':<25} {conv_weights:>13,} {conv_bias:>8,} {conv_total:>13,}")
    print(f"{'Ratio (FC / Conv)':<25} {'':<15} {'':<10} {fc_total / conv_total:>13.1f}x")

    print(f"\n--- Why CNNs use fewer parameters ---")
    print(f"\nFully Connected:")
    print(f"  - Every input pixel connects to every output neuron")
    print(f"  - {fc_input:,} inputs × {output_channels} outputs = {fc_weights:,} weights")

    print(f"\nConvolutional (3×3):")
    print(f"  - Each filter: {kernel_size}×{kernel_size}×{input_c} = {kernel_size * kernel_size * input_c} weights")
    print(f"  - {output_channels} filters × {kernel_size * kernel_size * input_c} = {conv_weights:,} weights")
    print(f"  - Same weights reused at every spatial position (weight sharing)")

    print(f"\nThis {fc_total / conv_total:.0f}x reduction comes from:")
    print(f"  1. Local connectivity: each neuron sees only a 3×3 patch")
    print(f"  2. Weight sharing: same filter applied across all positions")


def demo_pooling_dimension_tracking():
    """
    Track how dimensions shrink through multiple pooling layers.
    Shows the compression of spatial information.
    """
    print("\n" + "=" * 60)
    print("POOLING DIMENSION TRACKING")
    print("=" * 60)

    # Start with ImageNet-standard 224×224 image
    starting_sizes = [224, 112, 64, 32]

    for start_size in starting_sizes:
        print(f"\n--- Starting from {start_size}×{start_size} ---")
        current = start_size
        layer = 0
        print(f"{'After Pool #':<15} {'Size':<15} {'Total Pixels':<15} {'Reduction':<15}")
        print("-" * 60)
        print(f"{'Input':<15} {current}×{current:<8} {current * current:>13,} {'(baseline)':<15}")

        original_pixels = current * current
        while current >= 2:
            layer += 1
            current = current // 2
            current_pixels = current * current
            reduction = original_pixels / current_pixels
            print(f"{'Pool ' + str(layer):<15} {current}×{current:<8} {current_pixels:>13,} {reduction:>13.0f}x smaller")

            if current <= 4:
                break

    print(f"\n--- Typical ResNet progression (224×224 input) ---")
    # ResNet uses stride-2 conv in first layer, then pooling
    stages = [
        ("Input", 224, 224),
        ("Conv1 (stride=2)", 224, 112),
        ("MaxPool", 112, 56),
        ("Stage 2", 56, 56),
        ("Stage 3 (stride=2)", 56, 28),
        ("Stage 4 (stride=2)", 28, 14),
        ("Stage 5 (stride=2)", 14, 7),
        ("Global AvgPool", 7, 1),
    ]

    print(f"{'Stage':<25} {'Input':<10} {'Output':<10} {'Pixels':<15}")
    print("-" * 60)
    for name, inp, out in stages:
        pixels = out * out
        print(f"{name:<25} {inp}×{inp:<5} {out}×{out:<5} {pixels:>13,}")

    print(f"\nInterpretation:")
    print(f"  - 224×224 = 50,176 spatial positions")
    print(f"  - 7×7 = 49 spatial positions (1000x compression)")
    print(f"  - Global average pool: 1 value per channel")
    print(f"  - Spatial information is progressively abstracted into 'what' not 'where'")


def demo_transfer_learning_comparison():
    """
    Demonstrate the value of transfer learning with a simple simulation.
    Shows how pre-learned features outperform random features.
    """
    print("\n" + "=" * 60)
    print("TRANSFER LEARNING: PRE-TRAINED VS RANDOM FEATURES")
    print("=" * 60)

    np.random.seed(42)

    # Simulate scenario: classifying images with limited data
    # Pre-trained CNN has learned to extract meaningful features
    # Random CNN outputs noise

    n_samples = 600  # Total samples
    n_classes = 5

    # Create underlying "true" features that determine the class
    # This simulates real visual patterns (edges, textures, shapes)
    true_features = np.random.randn(n_samples, 20)

    # Assign classes based on true features (linear boundaries)
    class_weights = np.random.randn(20, n_classes)
    class_scores = true_features @ class_weights
    y = np.argmax(class_scores, axis=1)

    # Pre-trained features: CNN has learned to extract the true features
    # plus some noise (but the signal is preserved)
    X_pretrained = np.hstack([
        true_features,  # The 20 meaningful features
        np.random.randn(n_samples, 80) * 0.1,  # 80 minor features
    ])

    # Random features: untrained CNN outputs essentially noise
    # True features are buried and scrambled
    X_random = np.random.randn(n_samples, 100)

    # Split into fixed test set
    test_size = 100
    X_pre_test, X_pre_train = X_pretrained[:test_size], X_pretrained[test_size:]
    X_rand_test, X_rand_train = X_random[:test_size], X_random[test_size:]
    y_test, y_train = y[:test_size], y[test_size:]

    train_sizes = [25, 50, 100, 200, 500]

    print(f"\nScenario: 5-class image classification with limited training data")
    print(f"Comparing logistic regression trained on extracted features:")
    print(f"  - Pre-trained CNN: extracts meaningful visual features")
    print(f"  - Random CNN: outputs essentially random projections")

    print(f"\n{'Training Size':<15} {'Pre-trained Acc':<18} {'Random Acc':<18} {'Difference':<15}")
    print("-" * 70)

    for train_size in train_sizes:
        # Train on subset
        X_pre_sub = X_pre_train[:train_size]
        X_rand_sub = X_rand_train[:train_size]
        y_sub = y_train[:train_size]

        # Train classifiers
        clf_pre = LogisticRegression(
            max_iter=2000,
            random_state=42,
            solver='lbfgs',
            C=1.0,
        )
        clf_pre.fit(X_pre_sub, y_sub)
        acc_pre = accuracy_score(y_test, clf_pre.predict(X_pre_test))

        clf_rand = LogisticRegression(
            max_iter=2000,
            random_state=42,
            solver='lbfgs',
            C=1.0,
        )
        clf_rand.fit(X_rand_sub, y_sub)
        acc_rand = accuracy_score(y_test, clf_rand.predict(X_rand_test))

        diff = acc_pre - acc_rand
        print(f"{train_size:<15} {acc_pre:>16.1%} {acc_rand:>16.1%} {diff:>+13.1%}")

    print(f"\nInterpretation:")
    print(f"  - Random features ≈ 20% accuracy (near chance for 5 classes)")
    print(f"  - Pre-trained features achieve much higher accuracy")
    print(f"  - The gap is largest with small training sets")
    print(f"  - Pre-trained CNN 'already knows how to see' - you just teach it your task")


if __name__ == "__main__":
    demo_fc_parameter_explosion()
    demo_convolution_by_hand()
    demo_output_size_formula()
    demo_cnn_vs_fc_parameters()
    demo_pooling_dimension_tracking()
    demo_transfer_learning_comparison()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
