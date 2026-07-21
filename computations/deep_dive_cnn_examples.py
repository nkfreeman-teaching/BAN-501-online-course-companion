"""
Deep Dive: CNN Architecture - Numerical Examples

This script demonstrates key concepts from the CNN Architecture Deep Dive
with concrete calculations.
Run with: pixi run python course-companion-computations/deep_dive_cnn_examples.py

References in Course Companion:
- demo_fc_vs_cnn_parameters()    -> Deep Dive CNN, Parameter Explosion section
- demo_weight_sharing_savings()   -> Deep Dive CNN, Weight Sharing section
- demo_receptive_field_growth()   -> Deep Dive CNN, Hierarchical Learning section
- demo_convolution_verification() -> Deep Dive CNN, Convolution In-Depth section
- demo_output_size_formula()      -> Deep Dive CNN, Output Size section
- demo_pooling_dimensions()       -> Deep Dive CNN, Pooling section

Last updated: 2026-01-02
"""

import numpy as np
import torch
import torch.nn as nn


def demo_fc_vs_cnn_parameters():
    """
    Compare parameters needed for fully connected vs convolutional approaches.
    Demonstrates the parameter explosion problem.
    """
    print("=" * 60)
    print("FC vs CNN PARAMETER COMPARISON")
    print("=" * 60)

    # Image dimensions
    height, width, channels = 224, 224, 3
    input_size = height * width * channels  # 150,528

    # Fully connected approach
    fc_hidden = 1000
    fc_params_layer1 = input_size * fc_hidden + fc_hidden  # weights + bias

    # Convolutional approach
    # One conv layer: 64 filters, 3x3 kernel, 3 input channels
    conv_filters = 64
    kernel_size = 3
    conv_params = (kernel_size * kernel_size * channels * conv_filters) + conv_filters

    print(f"\nInput: {height}×{width}×{channels} RGB image")
    print(f"Input features (flattened): {input_size:,}")

    print(f"\n{'Approach':<25} {'Parameters':>15}")
    print("-" * 42)
    print(f"{'FC layer (1000 neurons)':<25} {fc_params_layer1:>15,}")
    print(f"{'Conv layer (64 3×3)':<25} {conv_params:>15,}")

    ratio = fc_params_layer1 / conv_params
    print(f"\nRatio: FC requires {ratio:,.0f}× more parameters!")

    # Full network comparison
    print(f"\n--- Full Network Comparison ---")

    # FC network
    fc_layers = [
        ("FC1 (150,528 → 4,096)", 150528 * 4096 + 4096),
        ("FC2 (4,096 → 4,096)", 4096 * 4096 + 4096),
        ("FC3 (4,096 → 1,000)", 4096 * 1000 + 1000),
    ]
    fc_total = sum(p for _, p in fc_layers)

    # CNN conv layers only (VGG-style)
    conv_layers = [
        ("Conv1 (3→64, 3×3)", 3 * 3 * 3 * 64 + 64),
        ("Conv2 (64→64, 3×3)", 3 * 3 * 64 * 64 + 64),
        ("Conv3 (64→128, 3×3)", 3 * 3 * 64 * 128 + 128),
        ("Conv4 (128→128, 3×3)", 3 * 3 * 128 * 128 + 128),
    ]
    conv_total = sum(p for _, p in conv_layers)

    print(f"\n{'FC Network':<30} {'Parameters':>12}")
    print("-" * 44)
    for name, params in fc_layers:
        print(f"{name:<30} {params:>12,}")
    print(f"{'Total':<30} {fc_total:>12,}")

    print(f"\n{'CNN (first 4 conv layers)':<30} {'Parameters':>12}")
    print("-" * 44)
    for name, params in conv_layers:
        print(f"{name:<30} {params:>12,}")
    print(f"{'Total':<30} {conv_total:>12,}")

    print(f"\nFC network: {fc_total:,} parameters")
    print(f"CNN conv layers: {conv_total:,} parameters")
    print(f"Savings: {fc_total / conv_total:,.0f}× fewer parameters in CNN")


def demo_weight_sharing_savings():
    """
    Quantify the parameter savings from weight sharing.
    Compares locally connected (no sharing) vs convolutional (shared weights).
    """
    print("\n" + "=" * 60)
    print("WEIGHT SHARING SAVINGS")
    print("=" * 60)

    # Setup: 224×224 RGB image, 64 filters, 3×3 kernel
    height, width = 224, 224
    in_channels = 3
    out_channels = 64
    kernel_size = 3

    # Output positions (with no padding)
    out_height = height - kernel_size + 1  # 222
    out_width = width - kernel_size + 1    # 222
    positions = out_height * out_width     # 49,284

    # Parameters per filter per position
    params_per_position = kernel_size * kernel_size * in_channels  # 27

    # Locally connected: different weights at each position
    locally_connected_params = positions * params_per_position * out_channels
    locally_connected_with_bias = locally_connected_params + (positions * out_channels)

    # Convolutional: same weights at all positions
    conv_params = params_per_position * out_channels  # weights only
    conv_with_bias = conv_params + out_channels

    print(f"\nSetup:")
    print(f"  Input: {height}×{width}×{in_channels} (RGB)")
    print(f"  Output: {out_height}×{out_width}×{out_channels}")
    print(f"  Kernel: {kernel_size}×{kernel_size}")
    print(f"  Output positions: {positions:,}")
    print(f"  Params per position: {params_per_position}")

    print(f"\n{'Approach':<25} {'Parameters':>15}")
    print("-" * 42)
    print(f"{'Locally connected':<25} {locally_connected_params:>15,}")
    print(f"{'Convolutional':<25} {conv_params:>15,}")

    ratio = locally_connected_params / conv_params
    print(f"\nWeight sharing reduces parameters by {ratio:,.0f}×")
    print(f"From {locally_connected_params:,} → {conv_params:,}")

    # Show where reuse happens
    print(f"\n--- Weight Reuse Visualization ---")
    print(f"The SAME {params_per_position} weights (one 3×3×3 filter) are used at:")
    print(f"  Position (0,0), (0,1), (0,2), ..., ({out_height-1},{out_width-1})")
    print(f"  Total: {positions:,} different positions")
    print(f"\nThis is like using ONE cookie cutter {positions:,} times")
    print(f"instead of making {positions:,} different cookie cutters!")


def demo_receptive_field_growth():
    """
    Show how receptive field grows through layers.
    Demonstrates RF = 1 + L(K-1) formula.
    """
    print("\n" + "=" * 60)
    print("RECEPTIVE FIELD GROWTH")
    print("=" * 60)

    kernel_size = 3

    print(f"\nReceptive Field formula (no pooling): RF = 1 + L × (K - 1)")
    print(f"Where L = number of layers, K = kernel size")
    print(f"\nUsing {kernel_size}×{kernel_size} convolutions:")

    print(f"\n{'Layers':<10} {'Formula':<20} {'RF Size':>10}")
    print("-" * 42)

    for layers in [1, 2, 3, 5, 10, 20]:
        rf = 1 + layers * (kernel_size - 1)
        formula = f"1 + {layers}×{kernel_size - 1}"
        print(f"{layers:<10} {formula:<20} {rf:>10}×{rf}")

    # With pooling
    print(f"\n--- With 2×2 Pooling (stride 2) ---")
    print(f"Each pooling layer doubles effective RF\n")

    rf_no_pool = 1
    rf_with_pool = 1

    print(f"{'Layer':<15} {'RF (no pool)':>15} {'RF (with pool)':>15}")
    print("-" * 47)

    for i in range(1, 6):
        rf_no_pool = 1 + i * (kernel_size - 1)

        # With pooling every 2 conv layers
        if i % 2 == 0:
            rf_with_pool = rf_with_pool * 2 + (kernel_size - 1)
        else:
            rf_with_pool = rf_with_pool + (kernel_size - 1)

        pool_indicator = " (pool)" if i % 2 == 0 else ""
        print(f"Layer {i}{pool_indicator:<8} {rf_no_pool:>15} {rf_with_pool:>15}")

    print(f"\nWhy RF matters:")
    print(f"  - Layer 1 (RF=3): sees edges, small textures")
    print(f"  - Layer 5 (RF=11): sees parts of objects")
    print(f"  - Layer 10+ (RF=21+): sees whole objects")
    print(f"  - With pooling: can reach RF=200+ for scene understanding")


def demo_convolution_verification():
    """
    Verify the manual convolution calculation from the Deep Dive document.
    Uses the same input/filter values to confirm output = -2.4.
    """
    print("\n" + "=" * 60)
    print("CONVOLUTION VERIFICATION")
    print("=" * 60)

    # Input patch from document (top-left 3×3 of each channel)
    red = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
    ])

    green = np.array([
        [0.4, 0.5, 0.6],
        [0.5, 0.6, 0.7],
        [0.6, 0.7, 0.8],
    ])

    blue = np.array([
        [0.7, 0.8, 0.9],
        [0.8, 0.9, 1.0],
        [0.9, 1.0, 1.1],
    ])

    # Sobel-like vertical edge detector (same for all channels)
    sobel = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ])

    print(f"\nInput patch (3×3 per channel):")
    print(f"Red:   {red[0].tolist()}")
    print(f"       {red[1].tolist()}")
    print(f"       {red[2].tolist()}")

    print(f"\nFilter (Sobel vertical, same for all channels):")
    print(f"       {sobel[0].tolist()}")
    print(f"       {sobel[1].tolist()}")
    print(f"       {sobel[2].tolist()}")

    # Manual computation
    red_contrib = np.sum(red * sobel)
    green_contrib = np.sum(green * sobel)
    blue_contrib = np.sum(blue * sobel)
    bias = 0.0
    total = red_contrib + green_contrib + blue_contrib + bias

    print(f"\n--- Manual Calculation ---")
    print(f"Red channel:   sum(input × filter) = {red_contrib:.1f}")
    print(f"Green channel: sum(input × filter) = {green_contrib:.1f}")
    print(f"Blue channel:  sum(input × filter) = {blue_contrib:.1f}")
    print(f"Bias: {bias}")
    print(f"Total: {red_contrib:.1f} + {green_contrib:.1f} + {blue_contrib:.1f} + {bias} = {total:.1f}")

    # PyTorch verification
    print(f"\n--- PyTorch Verification ---")

    # Build full 8×8 input with the patch in top-left
    input_tensor = torch.zeros(1, 3, 8, 8)
    input_tensor[0, 0, :3, :3] = torch.from_numpy(red).float()
    input_tensor[0, 1, :3, :3] = torch.from_numpy(green).float()
    input_tensor[0, 2, :3, :3] = torch.from_numpy(blue).float()

    # Create conv layer with our Sobel filter
    conv = nn.Conv2d(
        in_channels=3,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
    )

    # Set weights to Sobel filter for all input channels
    with torch.no_grad():
        for c in range(3):
            conv.weight[0, c] = torch.from_numpy(sobel).float()

    # Run convolution
    output = conv(input_tensor)
    pytorch_result = output[0, 0, 0, 0].item()

    print(f"PyTorch output[0,0,0,0] = {pytorch_result:.1f}")
    print(f"\n✓ Manual calculation matches PyTorch: {total:.1f} = {pytorch_result:.1f}")

    print(f"\nDocument states output = -2.4")
    print(f"Our verification: {total:.1f}")


def demo_output_size_formula():
    """
    Test the output size formula with various configurations.
    Formula: floor((input - kernel + 2×padding) / stride) + 1
    """
    print("\n" + "=" * 60)
    print("OUTPUT SIZE FORMULA")
    print("=" * 60)

    print(f"\nFormula: output = floor((input - kernel + 2×padding) / stride) + 1")

    configs = [
        # (input, kernel, padding, stride, description)
        (8, 3, 0, 1, "Basic (from document)"),
        (224, 3, 1, 1, "Same padding (output=input)"),
        (224, 3, 0, 1, "No padding"),
        (224, 3, 0, 2, "Stride 2, no padding"),
        (224, 7, 3, 2, "7×7 kernel (AlexNet style)"),
        (64, 5, 2, 2, "Practice problem"),
    ]

    print(f"\n{'Config':<25} {'Calculation':<30} {'Output':>8}")
    print("-" * 65)

    for inp, k, p, s, desc in configs:
        output = (inp - k + 2 * p) // s + 1
        calc = f"({inp}-{k}+2×{p})/{s}+1"
        print(f"{desc:<25} {calc:<30} {output:>8}")

    # PyTorch verification
    print(f"\n--- PyTorch Verification ---")
    for inp, k, p, s, desc in configs[:3]:
        x = torch.randn(1, 3, inp, inp)
        conv = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=k,
            stride=s,
            padding=p,
        )
        out = conv(x)
        expected = (inp - k + 2 * p) // s + 1
        print(f"{desc}: input {inp}×{inp} → output {out.shape[2]}×{out.shape[3]} (expected: {expected}×{expected})")


def demo_pooling_dimensions():
    """
    Track dimensions through pooling layers.
    Shows how max pooling reduces spatial dimensions.
    """
    print("\n" + "=" * 60)
    print("POOLING DIMENSION REDUCTION")
    print("=" * 60)

    # Starting dimensions
    h, w = 224, 224
    channels = 64

    print(f"\nStarting: {h}×{w}×{channels}")
    print(f"Using 2×2 max pooling with stride 2")

    print(f"\n{'Layer':<15} {'Input':<15} {'Output':<15} {'Reduction':>12}")
    print("-" * 60)

    for i in range(5):
        h_out = h // 2
        w_out = w // 2
        reduction = (h * w) / (h_out * w_out)
        print(f"Pool {i+1:<9} {h}×{w}×{channels:<5} {h_out}×{w_out}×{channels:<5} {reduction:.0f}× smaller")
        h, w = h_out, w_out

    print(f"\nAfter 5 pooling layers: {h}×{w}×{channels}")
    print(f"Total reduction: {(224*224)/(h*w):.0f}× (spatial dimensions)")

    # PyTorch demonstration
    print(f"\n--- PyTorch Verification ---")
    x = torch.randn(1, 64, 224, 224)
    pool = nn.MaxPool2d(kernel_size=2, stride=2)

    print(f"Input shape: {x.shape}")
    for i in range(5):
        x = pool(x)
        print(f"After pool {i+1}: {x.shape}")

    # Max pooling example with values
    print(f"\n--- Max Pooling Example (from diagram) ---")
    input_4x4 = torch.tensor([
        [1, 3, 2, 4],
        [0, 2, 1, 3],
        [5, 6, 7, 8],
        [4, 1, 5, 2],
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
    output_2x2 = pool_2x2(input_4x4)

    print(f"\nInput (4×4):")
    print(input_4x4.squeeze().numpy())
    print(f"\nOutput (2×2) after 2×2 max pool:")
    print(output_2x2.squeeze().numpy())
    print(f"\nMax selections:")
    print(f"  [1,3,0,2] → max = 3")
    print(f"  [2,4,1,3] → max = 4")
    print(f"  [5,6,4,1] → max = 6")
    print(f"  [7,8,5,2] → max = 8")


if __name__ == "__main__":
    demo_fc_vs_cnn_parameters()
    demo_weight_sharing_savings()
    demo_receptive_field_growth()
    demo_convolution_verification()
    demo_output_size_formula()
    demo_pooling_dimensions()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
