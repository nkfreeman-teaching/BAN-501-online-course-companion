"""
Deep Dive: Neural Networks as Universal Approximators - Numerical Examples

This script demonstrates key concepts from the Universal Approximators Deep Dive
with concrete calculations.
Run with: pixi run python course-companion-computations/deep_dive_universal_approx_examples.py

References in Course Companion:
- demo_sigmoid_weight_effects()      -> Deep Dive Universal Approx, Sigmoid Intuition section
- demo_linear_regression_equiv()     -> Deep Dive Universal Approx, Example 1 (Linear Regression)
- demo_logistic_regression_equiv()   -> Deep Dive Universal Approx, Example 2 (Logistic Regression)
- demo_relu_bump_construction()      -> Deep Dive Universal Approx, Example 3 (Step Functions)
- demo_step_approximation()          -> Deep Dive Universal Approx, Example 3 (Step Functions)
- demo_polynomial_approximation()    -> Deep Dive Universal Approx, Example 4 (Polynomial)
- demo_width_vs_accuracy()           -> Deep Dive Universal Approx, Model Selection section

Last updated: 2026-01-02
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, LogisticRegression


def demo_sigmoid_weight_effects():
    """
    Show how weight magnitude controls sigmoid transition steepness.
    Higher weights → sharper transition → more step-like behavior.
    """
    print("=" * 60)
    print("SIGMOID WEIGHT EFFECTS")
    print("=" * 60)

    def sigmoid(x, weight):
        return 1 / (1 + np.exp(-weight * x))

    # Find transition width: x range where output goes from 0.1 to 0.9
    # sigmoid(w*x) = 0.1 → x = -ln(9)/w
    # sigmoid(w*x) = 0.9 → x = ln(9)/w
    # Width = 2*ln(9)/w ≈ 4.39/w

    weights = [1, 2, 5, 10]

    print("\nTransition width (x range for output 0.1 to 0.9):")
    print(f"{'Weight':>8} {'Width':>12} {'Interpretation':>25}")
    print("-" * 50)

    for w in weights:
        width = 2 * np.log(9) / w
        if w == 1:
            interp = "Gradual S-curve"
        elif w == 2:
            interp = "Moderate transition"
        elif w == 5:
            interp = "Sharp transition"
        else:
            interp = "Nearly step-like"
        print(f"{w:>8} {width:>12.3f} {interp:>25}")

    print("\nAt specific x values (weight=1 vs weight=5):")
    print(f"{'x':>6} {'sigmoid(1*x)':>14} {'sigmoid(5*x)':>14}")
    print("-" * 38)
    for x in [-2, -1, -0.5, 0, 0.5, 1, 2]:
        s1 = sigmoid(x, weight=1)
        s5 = sigmoid(x, weight=5)
        print(f"{x:>6.1f} {s1:>14.4f} {s5:>14.4f}")

    print("\nInterpretation:")
    print("  - Weight=1: Output changes gradually across x in [-2, 2]")
    print("  - Weight=5: Output jumps from ~0 to ~1 in narrow band around x=0")
    print("  - This is how sigmoids can approximate hard step boundaries")


def demo_linear_regression_equiv():
    """
    Demonstrate sklearn and PyTorch produce identical results for linear regression.
    Linear regression IS a neural network with 0 hidden layers.
    """
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION: SKLEARN VS PYTORCH EQUIVALENCE")
    print("=" * 60)

    # Generate data: y = 2*x1 + 3*x2 + 1 + noise
    np.random.seed(42)
    torch.manual_seed(42)
    n_samples = 1000

    X = np.random.randn(n_samples, 2).astype(np.float32)
    true_coef = np.array([2.0, 3.0])
    true_intercept = 1.0
    noise = 0.1 * np.random.randn(n_samples).astype(np.float32)
    y = X @ true_coef + true_intercept + noise

    print(f"\nTrue model: y = 2*x1 + 3*x2 + 1 + noise")
    print(f"Samples: {n_samples}, Noise std: 0.1")

    # SKLEARN
    sklearn_model = LinearRegression()
    sklearn_model.fit(X, y)

    # PYTORCH - equivalent neural network
    class LinearRegressionNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            return self.linear(x)

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).reshape(-1, 1).float()

    nn_model = LinearRegressionNN(input_dim=2)
    optimizer = torch.optim.Adam(
        nn_model.parameters(),
        lr=0.01,
    )
    criterion = nn.MSELoss()

    # Train until convergence
    for epoch in range(2000):
        optimizer.zero_grad()
        predictions = nn_model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()

    # Extract coefficients
    nn_coef = nn_model.linear.weight.data.numpy().flatten()
    nn_intercept = nn_model.linear.bias.data.numpy()[0]

    print("\nCoefficient comparison:")
    print(f"{'':>15} {'True':>10} {'sklearn':>10} {'PyTorch':>10}")
    print("-" * 50)
    print(f"{'x1 coefficient':>15} {true_coef[0]:>10.4f} {sklearn_model.coef_[0]:>10.4f} {nn_coef[0]:>10.4f}")
    print(f"{'x2 coefficient':>15} {true_coef[1]:>10.4f} {sklearn_model.coef_[1]:>10.4f} {nn_coef[1]:>10.4f}")
    print(f"{'intercept':>15} {true_intercept:>10.4f} {sklearn_model.intercept_:>10.4f} {nn_intercept:>10.4f}")

    diff_coef = np.abs(sklearn_model.coef_ - nn_coef).max()
    diff_int = abs(sklearn_model.intercept_ - nn_intercept)
    print(f"\nMax coefficient difference: {diff_coef:.6f}")
    print(f"Intercept difference: {diff_int:.6f}")

    print("\nInterpretation:")
    print("  - Both converge to the same solution (differences < 0.001)")
    print("  - sklearn uses closed-form matrix inverse: O(n^3)")
    print("  - PyTorch uses gradient descent: scales to larger data")
    print("  - Linear regression IS a neural network with 0 hidden layers")


def demo_logistic_regression_equiv():
    """
    Demonstrate sklearn and PyTorch produce identical results for logistic regression.
    Logistic regression IS a neural network with sigmoid activation.
    """
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION: SKLEARN VS PYTORCH EQUIVALENCE")
    print("=" * 60)

    # Generate binary classification data
    np.random.seed(42)
    torch.manual_seed(42)
    n_samples = 1000

    X = np.random.randn(n_samples, 2).astype(np.float32)
    # Decision boundary: x1 + 2*x2 > 0.5
    y = (X[:, 0] + 2 * X[:, 1] > 0.5).astype(np.float32)

    print(f"\nDecision boundary: x1 + 2*x2 = 0.5")
    print(f"Samples: {n_samples}, Class balance: {y.mean():.1%} positive")

    # SKLEARN — disable regularization with C=np.inf (sklearn 1.8 deprecated penalty=None)
    sklearn_model = LogisticRegression(
        C=np.inf,
        max_iter=1000,
    )
    sklearn_model.fit(X, y)

    # PYTORCH - equivalent neural network
    class LogisticRegressionNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).reshape(-1, 1).float()

    nn_model = LogisticRegressionNN(input_dim=2)
    optimizer = torch.optim.Adam(
        nn_model.parameters(),
        lr=0.05,
    )
    criterion = nn.BCELoss()

    # Train until convergence
    for epoch in range(2000):
        optimizer.zero_grad()
        predictions = nn_model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()

    # Extract coefficients
    nn_coef = nn_model.linear.weight.data.numpy().flatten()
    nn_intercept = nn_model.linear.bias.data.numpy()[0]

    # Compute accuracy
    sklearn_pred = sklearn_model.predict(X)
    sklearn_acc = (sklearn_pred == y).mean()

    with torch.no_grad():
        nn_pred = (nn_model(X_tensor) > 0.5).numpy().flatten()
    nn_acc = (nn_pred == y).mean()

    print("\nCoefficient comparison:")
    print(f"{'':>15} {'sklearn':>12} {'PyTorch':>12}")
    print("-" * 42)
    print(f"{'x1 coefficient':>15} {sklearn_model.coef_[0, 0]:>12.4f} {nn_coef[0]:>12.4f}")
    print(f"{'x2 coefficient':>15} {sklearn_model.coef_[0, 1]:>12.4f} {nn_coef[1]:>12.4f}")
    print(f"{'intercept':>15} {sklearn_model.intercept_[0]:>12.4f} {nn_intercept:>12.4f}")

    print(f"\nAccuracy comparison:")
    print(f"  sklearn: {sklearn_acc:.4f}")
    print(f"  PyTorch: {nn_acc:.4f}")

    # Coefficient ratio (should match true ratio of 1:2)
    sklearn_ratio = sklearn_model.coef_[0, 1] / sklearn_model.coef_[0, 0]
    nn_ratio = nn_coef[1] / nn_coef[0]
    print(f"\nCoefficient ratio (x2/x1, should be ~2.0):")
    print(f"  sklearn: {sklearn_ratio:.4f}")
    print(f"  PyTorch: {nn_ratio:.4f}")

    print("\nInterpretation:")
    print("  - Both achieve identical accuracy and similar coefficients")
    print("  - Coefficient magnitudes may differ (only ratio matters for boundary)")
    print("  - Logistic regression IS a neural network with sigmoid activation")


def demo_relu_bump_construction():
    """
    Show step-by-step how three ReLUs create a localized "bump" function.
    This is the building block for universal approximation.
    """
    print("\n" + "=" * 60)
    print("RELU BUMP CONSTRUCTION")
    print("=" * 60)

    def relu(x):
        return np.maximum(0, x)

    print("\nFormula: bump(x) = ReLU(x) - 2*ReLU(x-1) + ReLU(x-2)")
    print("This rises from 0 to 1 over [0,1], then falls back to 0 over [1,2]")

    x_values = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])

    print(f"\n{'x':>6} {'ReLU(x)':>10} {'ReLU(x-1)':>12} {'ReLU(x-2)':>12} {'bump':>8} {'Shape':>10}")
    print("-" * 64)

    for x in x_values:
        r1 = relu(x)
        r2 = relu(x - 1)
        r3 = relu(x - 2)
        bump = r1 - 2 * r2 + r3

        if x <= 0 or x >= 2:
            shape = "Outside"
        elif x < 1:
            shape = "Rising"
        elif x == 1:
            shape = "Peak"
        else:
            shape = "Falling"

        print(f"{x:>6.1f} {r1:>10.2f} {r2:>12.2f} {r3:>12.2f} {bump:>8.2f} {shape:>10}")

    print("\nGeneralizing: ReLU(x-a) - 2*ReLU(x-c) + ReLU(x-b), with c midway")
    print("between a and b, makes a triangular bump that rises from 0 at x=a")
    print("to a peak of (c-a) at x=c, then falls back to 0 at x=b.")
    print("  - Peak location: x = c")
    print("  - Width: b - a")

    print("\nWith many such localized bumps at different positions and heights,")
    print("you can approximate any continuous function!")


def demo_step_approximation():
    """
    Show more neurons → better approximation of a step function.
    """
    print("\n" + "=" * 60)
    print("STEP FUNCTION APPROXIMATION")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # Generate training data for step function at x=0
    n_samples = 500
    X = np.random.uniform(
        low=-3,
        high=3,
        size=(n_samples, 1),
    ).astype(np.float32)
    y = (X > 0).astype(np.float32)  # Step at x=0

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    # Test data for evaluation
    X_test = np.linspace(-3, 3, 1000).reshape(-1, 1).astype(np.float32)
    y_test = (X_test > 0).astype(np.float32)
    X_test_tensor = torch.from_numpy(X_test).float()

    def train_network(n_hidden, epochs=2000):
        model = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid(),
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
        )
        criterion = nn.BCELoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()

        # Compute test MSE
        with torch.no_grad():
            pred_test = model(X_test_tensor).numpy()
        mse = np.mean((pred_test - y_test) ** 2)
        return mse, model

    print("\nTarget: step function at x=0 (output 0 for x<0, 1 for x>0)")
    print(f"{'Neurons':>10} {'MSE':>12} {'Quality':>20}")
    print("-" * 45)

    neuron_counts = [2, 5, 10, 20, 50]
    for n in neuron_counts:
        mse, _ = train_network(n)
        if mse > 0.1:
            quality = "Rough approximation"
        elif mse > 0.05:
            quality = "Moderate"
        elif mse > 0.01:
            quality = "Good"
        elif mse > 0.005:
            quality = "Very good"
        else:
            quality = "Nearly exact"
        print(f"{n:>10} {mse:>12.4f} {quality:>20}")

    print("\nInterpretation:")
    print("  - 2 neurons: Can only create ~2 kinks, poor approximation")
    print("  - 5-10 neurons: Visible improvement, captures basic shape")
    print("  - 20+ neurons: Approaches step function closely")
    print("  - 50 neurons: Nearly indistinguishable from true step")


def demo_polynomial_approximation():
    """
    Show how ReLU networks approximate x² with piecewise linear segments.
    """
    print("\n" + "=" * 60)
    print("POLYNOMIAL APPROXIMATION: LEARNING x^2 WITH RELU")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # Generate training data for y = x²
    n_samples = 500
    X = np.random.uniform(
        low=-2,
        high=2,
        size=(n_samples, 1),
    ).astype(np.float32)
    y = (X ** 2).astype(np.float32)

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    # Test data
    X_test = np.linspace(-2, 2, 100).reshape(-1, 1).astype(np.float32)
    y_test = X_test ** 2
    X_test_tensor = torch.from_numpy(X_test).float()

    def train_network(n_hidden, epochs=3000):
        model = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
        )
        criterion = nn.MSELoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred_test = model(X_test_tensor).numpy()
        mse = np.mean((pred_test - y_test) ** 2)
        return mse, model

    print("\nTarget: y = x^2 (smooth parabola)")
    print("Network: 1 hidden layer with ReLU → piecewise linear output")
    print(f"\n{'Neurons':>10} {'MSE':>12} {'Max Kinks':>12} {'Approximation':>20}")
    print("-" * 60)

    neuron_counts = [2, 5, 10, 20, 50]
    for n in neuron_counts:
        mse, _ = train_network(n)
        # Each neuron can contribute up to 1 kink
        if mse > 0.1:
            approx = "2 line segments"
        elif mse > 0.01:
            approx = "Coarse piecewise"
        elif mse > 0.001:
            approx = "Fine piecewise"
        else:
            approx = "Nearly smooth"
        print(f"{n:>10} {mse:>12.6f} {n:>12} {approx:>20}")

    print("\nInterpretation:")
    print("  - ReLU networks create PIECEWISE LINEAR functions")
    print("  - Each neuron adds one potential 'corner point'")
    print("  - More neurons → more corners → smoother-looking approximation")
    print("  - It's not truly learning x^2, but approximating it with line segments")


def demo_width_vs_accuracy():
    """
    Compare neurons needed for different target functions.
    Provides practical guidance on network sizing.
    """
    print("\n" + "=" * 60)
    print("WIDTH VS ACCURACY TRADE-OFF")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # Define target functions
    def step_func(x):
        return (x > 0).astype(np.float32)

    def square_func(x):
        return (x ** 2).astype(np.float32)

    def sin_func(x):
        return np.sin(2 * np.pi * x).astype(np.float32)

    functions = [
        ("Step at 0", step_func, (-3, 3), True),     # Binary output
        ("x^2", square_func, (-2, 2), False),        # Regression
        ("sin(2*pi*x)", sin_func, (-1, 1), False),   # Regression
    ]

    def train_for_target(target_func, x_range, is_classification, n_hidden, epochs=3000):
        n_samples = 500
        X = np.random.uniform(
            low=x_range[0],
            high=x_range[1],
            size=(n_samples, 1),
        ).astype(np.float32)
        y = target_func(X)

        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()

        # Test data
        X_test = np.linspace(x_range[0], x_range[1], 200).reshape(-1, 1).astype(np.float32)
        y_test = target_func(X_test)
        X_test_tensor = torch.from_numpy(X_test).float()

        if is_classification:
            model = nn.Sequential(
                nn.Linear(1, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, 1),
                nn.Sigmoid(),
            )
            criterion = nn.BCELoss()
        else:
            model = nn.Sequential(
                nn.Linear(1, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, 1),
            )
            criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
        )

        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred_test = model(X_test_tensor).numpy()
        mse = np.mean((pred_test - y_test) ** 2)
        return mse

    print("\nNeurons needed to achieve target MSE:")
    print(f"{'Function':>15} {'MSE < 0.01':>15} {'MSE < 0.001':>15}")
    print("-" * 50)

    for name, func, x_range, is_class in functions:
        neurons_01 = "?"
        neurons_001 = "?"

        for n in [5, 10, 20, 30, 50, 75, 100]:
            mse = train_for_target(func, x_range, is_class, n)
            if neurons_01 == "?" and mse < 0.01:
                neurons_01 = str(n)
            if neurons_001 == "?" and mse < 0.001:
                neurons_001 = str(n)

        print(f"{name:>15} {neurons_01:>15} {neurons_001:>15}")

    print("\nInterpretation:")
    print("  - Step functions: Sharp transitions need moderate neurons")
    print("  - Polynomials: Smooth curves need more neurons for accuracy")
    print("  - Oscillating functions (sin): Need many neurons to capture wiggles")
    print("  - Rule of thumb: More 'wiggly' = more neurons needed")


if __name__ == "__main__":
    demo_sigmoid_weight_effects()
    demo_linear_regression_equiv()
    demo_logistic_regression_equiv()
    demo_relu_bump_construction()
    demo_step_approximation()
    demo_polynomial_approximation()
    demo_width_vs_accuracy()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
