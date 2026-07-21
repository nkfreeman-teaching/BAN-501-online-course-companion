"""
Deep Dive: Surprising Phenomena in Modern Deep Learning - Numerical Examples

This script demonstrates key concepts from the Surprising Phenomena deep dive
with concrete calculations.

Run with: pixi run python course-companion-computations/deep_dive_surprising_phenomena_examples.py

References in Course Companion:
- demo_bias_variance_tradeoff()       -> Classical View section (new example)
- demo_double_descent_random_features() -> Phenomenon 1: Double Descent
- demo_interpolation_threshold_peak() -> Phenomenon 1: Double Descent (new example)
- demo_grokking_simulation()          -> Phenomenon 2: Grokking (simplified simulation)
- demo_weight_norm_evolution()        -> Phenomenon 2: Grokking (new example)
- demo_emergence_metric_mirage()      -> Phenomenon 3: Emergent Abilities
- demo_phase_transitions_comparison() -> Connecting the Phenomena (new example)

Last updated: 2026-01-04
"""

import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# ============================================================================
# CLASSICAL VIEW
# ============================================================================


def demo_bias_variance_tradeoff():
    """
    Demonstrate the classical U-curve with polynomial regression.

    Shows how test error first decreases then increases as model complexity
    (polynomial degree) increases.
    """
    print("\n" + "=" * 60)
    print("BIAS-VARIANCE TRADEOFF: Classical U-Curve")
    print("=" * 60)

    # Generate data: true function is quadratic with noise
    np.random.seed(42)
    n_samples = 50
    X = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
    y_true = 0.5 * X.ravel()**2 - X.ravel() + 1  # True quadratic
    y = y_true + np.random.randn(n_samples) * 0.5  # Add noise

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # True function for test evaluation
    y_test_true = 0.5 * X_test.ravel()**2 - X_test.ravel() + 1

    print(f"\nDataset: {len(X_train)} training, {len(X_test)} test samples")
    print(f"True function: 0.5x² - x + 1")
    print()

    print(f"{'Degree':<10} {'Train MSE':<12} {'Test MSE':<12} {'Regime':<20}")
    print("-" * 54)

    degrees = [1, 2, 3, 5, 8, 12, 15]

    for deg in degrees:
        # Create polynomial features
        # Standardize the polynomial features before the least-squares fit.
        # High-degree raw powers of x are wildly different in scale (a Vandermonde
        # matrix), which makes the unscaled solve numerically unstable at degrees
        # 12-15; scaling conditions the problem so train error stays monotone.
        model = make_pipeline(
            PolynomialFeatures(degree=deg, include_bias=False),
            StandardScaler(),
            LinearRegression(),
        )
        model.fit(X_train, y_train)

        # Evaluate
        train_mse = mean_squared_error(y_train, model.predict(X_train))
        test_mse = mean_squared_error(y_test, model.predict(X_test))

        # Determine regime
        if deg < 2:
            regime = "Underfitting (high bias)"
        elif deg == 2:
            regime = "Sweet spot"
        elif deg <= 5:
            regime = "Slight overfitting"
        else:
            regime = "Overfitting (high var)"

        print(f"{deg:<10} {train_mse:<12.4f} {test_mse:<12.4f} {regime:<20}")

    print()
    print("Interpretation: Test error follows U-curve. Degree 2 (matching true")
    print("function) gives lowest test error. Higher degrees overfit the noise.")


# ============================================================================
# DOUBLE DESCENT
# ============================================================================


def demo_double_descent_random_features():
    """
    Demonstrate double descent with min-norm linear regression.

    Fixes the sample size and varies the number of features p the model may
    use. The minimum-norm least-squares solution (what gradient descent finds
    in the overparameterized regime) reliably shows the three regimes:
    underparameterized, an interpolation peak at p ~ n, and overparameterized.
    This isotropic-Gaussian setup (Hastie et al., 2019) is well conditioned, so
    the curve is stable and reproducible across library versions.
    """
    print("\n" + "=" * 60)
    print("DOUBLE DESCENT: Min-Norm Linear Regression")
    print("=" * 60)

    # Fixed dataset size; we sweep the feature count p around n.
    n_train = 100
    p_max = 900          # true signal lives in a high-dimensional space
    k_signal = 140       # number of dimensions carrying real signal
    snr = 5.0            # squared norm of the true coefficient vector
    noise = 0.4
    n_trials = 40        # average over random draws to smooth the curve

    print(f"\n{n_train} training points; signal in {k_signal} of {p_max} dimensions.")
    print("For each feature count p we fit the minimum-norm least-squares")
    print(f"solution and average test error over {n_trials} random draws.")
    print()

    print(f"{'Features':<12} {'Ratio to n':<12} {'Test MSE':<12} {'Regime':<25}")
    print("-" * 61)

    # Straddle p = n (skip exactly 100) so the interpolation peak stays finite.
    feature_counts = [10, 25, 45, 65, 80, 95, 105, 130, 175, 250, 400, 700]

    for p in feature_counts:
        errors = []
        for trial in range(n_trials):
            rng = np.random.RandomState(trial)
            beta = np.zeros(p_max)
            beta[:k_signal] = rng.randn(k_signal)
            beta *= np.sqrt(snr) / np.linalg.norm(beta)

            X_train = rng.randn(n_train, p_max)
            y_train = X_train @ beta + noise * rng.randn(n_train)
            X_test = rng.randn(3000, p_max)
            y_test = X_test @ beta  # noise-free target

            # Minimum-norm least squares using the first p features.
            coef, *_ = np.linalg.lstsq(X_train[:, :p], y_train, rcond=None)
            errors.append(mean_squared_error(y_test, X_test[:, :p] @ coef))

        test_mse = float(np.mean(errors))
        ratio = p / n_train

        # Determine regime
        if ratio < 0.9:
            regime = "Underparameterized"
        elif ratio <= 1.1:
            regime = "Interpolation peak!"
        else:
            regime = "Overparameterized"

        print(f"{p:<12} {ratio:<12.2f} {test_mse:<12.3f} {regime:<25}")

    print()
    print("Interpretation: In the classical (underparameterized) regime, test")
    print("error climbs as the feature count approaches the number of samples,")
    print("peaking sharply at the interpolation threshold (p ≈ n). Pushing into")
    print("the overparameterized regime, error descends again and drops below the")
    print("best underparameterized model—the second descent.")


def demo_interpolation_threshold_peak():
    """
    Focus on the interpolation threshold to show why it's dangerous.

    Zooms in around n_features ≈ n_samples to show the peak clearly.
    Uses Fourier features for clean demonstration.
    """
    print("\n" + "=" * 60)
    print("INTERPOLATION THRESHOLD: The Dangerous Zone")
    print("=" * 60)

    n_train = 50
    np.random.seed(123)

    X_train = np.random.uniform(0, 2 * np.pi, n_train)
    X_test = np.linspace(0, 2 * np.pi, 500)

    def true_function(x):
        return np.sin(x) + 0.5 * np.sin(2 * x)

    y_train = true_function(X_train) + 0.3 * np.random.randn(n_train)
    y_test = true_function(X_test)

    def fourier_features(X, n_features):
        """Create Fourier basis."""
        features = [np.ones_like(X)]
        for k in range(1, (n_features + 1) // 2 + 1):
            if len(features) < n_features:
                features.append(np.sin(k * X))
            if len(features) < n_features:
                features.append(np.cos(k * X))
        return np.column_stack(features[:n_features])

    print(f"\nDataset: {n_train} training samples")
    print(f"Zooming in around n_features = {n_train}")
    print()

    print(f"{'Features':<10} {'Ratio':<10} {'Test MSE':<12} {'Note':<30}")
    print("-" * 62)

    # Zoom in around the threshold
    feature_counts = [30, 40, 45, 48, 50, 52, 55, 60, 80, 100]

    for n_feat in feature_counts:
        Phi_train = fourier_features(X_train, n_feat)
        Phi_test = fourier_features(X_test, n_feat)

        model = Ridge(alpha=0.01)
        model.fit(Phi_train, y_train)

        test_mse = mean_squared_error(y_test, model.predict(Phi_test))
        ratio = n_feat / n_train

        if n_feat == 50:
            note = "<<< PEAK: exactly n features"
        elif abs(ratio - 1.0) < 0.1:
            note = "Near threshold"
        elif ratio < 1:
            note = ""
        else:
            note = "Descending..."

        print(f"{n_feat:<10} {ratio:<10.2f} {test_mse:<12.4f} {note:<30}")

    print()
    print("Interpretation: Error peaks at or near n_features = n_samples.")
    print("Just slightly more features already shows improvement.")
    print("The interpolation threshold is the worst place to be.")


# ============================================================================
# GROKKING
# ============================================================================


def demo_grokking_simulation():
    """
    Simulate the grokking phenomenon without full neural network training.

    This is a simplified demonstration showing the characteristic pattern:
    - Fast memorization (training accuracy hits 100% early)
    - Delayed generalization (test accuracy improves much later)

    Note: Real grokking requires actual neural network training with weight
    decay on algorithmic tasks. This simulation shows the expected pattern.
    """
    print("\n" + "=" * 60)
    print("GROKKING: Simulated Training Trajectory")
    print("=" * 60)

    print("\nNote: This is a simulation of the grokking pattern.")
    print("Real grokking requires training neural networks with weight decay")
    print("on algorithmic tasks like modular arithmetic.")
    print()

    # Simulate the characteristic grokking curve
    epochs = [100, 1000, 5000, 10000, 15000, 20000, 25000, 30000]

    # Training accuracy: reaches 100% quickly and stays there
    train_acc = [100.0] * len(epochs)

    # Test accuracy: stays low for a long time, then suddenly improves
    # Simulating the S-curve transition
    def grokking_curve(epoch):
        """Simulate test accuracy with delayed generalization."""
        if epoch < 8000:
            # Near random (1/97 ≈ 1% for modular arithmetic with p=97)
            return 1 + np.random.uniform(0, 3)
        elif epoch < 12000:
            # Starting to generalize
            return 5 + (epoch - 8000) / 400
        elif epoch < 20000:
            # Rapid improvement
            return 15 + (epoch - 12000) / 150
        else:
            # Approaching perfect
            base = 70 + (epoch - 20000) / 400
            return min(100.0, base)

    np.random.seed(42)
    test_acc = [grokking_curve(e) for e in epochs]
    # Clamp and format
    test_acc = [round(min(100.0, max(1.0, t)), 1) for t in test_acc]

    # Determine phase
    def get_phase(train, test, epoch):
        if test < 10:
            return "Memorized"
        elif test < 20:
            return "Starting..."
        elif test < 50:
            return "Transition"
        elif test < 90:
            return "Grokking!"
        elif test < 99:
            return "Almost there"
        else:
            return "Full generalization"

    print(f"{'Epoch':<10} {'Train Acc':<14} {'Test Acc':<14} {'Phase':<20}")
    print("-" * 58)

    for i, epoch in enumerate(epochs):
        phase = get_phase(train_acc[i], test_acc[i], epoch)
        train_str = f"{train_acc[i]:.1f}%"
        test_str = f"{test_acc[i]:.1f}%"
        print(f"{epoch:<10} {train_str:<14} {test_str:<14} {phase:<20}")

    print()
    print("Interpretation: The network reaches 100% training accuracy by epoch")
    print("100, but test accuracy stays near random for thousands of epochs.")
    print("Around epoch 10,000-25,000, generalization suddenly improves.")


def demo_weight_norm_evolution():
    """
    Show how regularization (weight decay) affects weight norms over training.

    Demonstrates that weight decay continuously pushes weights toward
    smaller values, eventually favoring simpler generalizing solutions.
    """
    print("\n" + "=" * 60)
    print("WEIGHT NORM EVOLUTION: Regularization Pressure")
    print("=" * 60)

    print("\nSimulating weight norm trajectories with and without weight decay.")
    print()

    epochs = [0, 100, 500, 1000, 5000, 10000, 20000, 30000]

    # Without weight decay: weights grow and stay large (memorization)
    def norm_no_decay(epoch):
        # Grows quickly then plateaus at high value
        return 50 * (1 - np.exp(-epoch / 200)) + 5

    # With weight decay: norm initially grows (for memorization) then decreases
    # as network finds simpler solution
    def norm_with_decay(epoch):
        if epoch < 500:
            # Initial growth for memorization
            return 30 * (1 - np.exp(-epoch / 100)) + 5
        elif epoch < 10000:
            # Slow decay during plateau
            return 30 - 10 * (epoch - 500) / 10000
        else:
            # Transition to simpler solution
            return 20 - 15 * min(1, (epoch - 10000) / 20000)

    no_decay = [round(norm_no_decay(e), 1) for e in epochs]
    with_decay = [round(norm_with_decay(e), 1) for e in epochs]

    print(f"{'Epoch':<10} {'No Weight Decay':<18} {'With Weight Decay':<18} {'Ratio':<10}")
    print("-" * 56)

    for i, epoch in enumerate(epochs):
        ratio = no_decay[i] / with_decay[i] if with_decay[i] > 0 else float('inf')
        print(f"{epoch:<10} {no_decay[i]:<18.1f} {with_decay[i]:<18.1f} {ratio:<10.2f}x")

    print()
    print("Interpretation: Without weight decay, weights stay large after")
    print("memorization (the network has no pressure to simplify). With weight")
    print("decay, norms gradually decrease, pushing toward simpler solutions—")
    print("this is what enables grokking.")


# ============================================================================
# EMERGENT ABILITIES
# ============================================================================


def demo_emergence_metric_mirage():
    """
    Demonstrate how metric choice can create apparent emergence.

    The underlying capability improves smoothly, but exact-match metrics
    show sharp transitions while per-step metrics show gradual improvement.
    """
    print("\n" + "=" * 60)
    print("EMERGENCE MIRAGE: Metric Choice Effects")
    print("=" * 60)

    print("\nSimulating a task requiring 5 correct steps (like multi-digit arithmetic).")
    print("True capability improves smoothly, but metrics tell different stories.")
    print()

    def true_capability(scale):
        """Probability of getting each step right (smooth sigmoid)."""
        return 1 / (1 + np.exp(-(scale - 9) / 2))

    def exact_match(scale):
        """All 5 steps must be correct."""
        p = true_capability(scale)
        return p ** 5

    def per_step_accuracy(scale):
        """Average accuracy per step."""
        return true_capability(scale)

    print(f"{'Scale':<10} {'Per-Step Acc':<15} {'Exact Match':<15} {'Appearance':<25}")
    print("-" * 65)

    scales = [5, 7, 9, 10, 11, 13, 15]

    for s in scales:
        psa = per_step_accuracy(s) * 100
        em = exact_match(s) * 100

        if em < 1 and psa < 20:
            appearance = "Both low"
        elif em < 5:
            appearance = "Gradual vs flat"
        elif em < 30:
            appearance = "Gradual vs jump"
        else:
            appearance = "Both high"

        psa_str = f"{psa:.1f}%"
        em_str = f"{em:.1f}%"
        print(f"{s:<10} {psa_str:<15} {em_str:<15} {appearance:<25}")

    print()
    print("Interpretation: Per-step accuracy improves gradually and smoothly.")
    print("But exact match (requiring all 5 steps correct) shows a sharp")
    print("transition. Same underlying capability, different appearance—")
    print("just from metric choice.")


# ============================================================================
# CONNECTING PHENOMENA
# ============================================================================


def demo_phase_transitions_comparison():
    """
    Compare the phase transition signatures across all three phenomena.

    Shows that all three involve sudden changes at critical thresholds,
    but the axes and mechanisms differ.
    """
    print("\n" + "=" * 60)
    print("PHASE TRANSITIONS: Comparing the Three Phenomena")
    print("=" * 60)

    print("\nAll three phenomena involve sudden changes at critical thresholds.")
    print()

    # Create a comparison table
    print("┌" + "─" * 20 + "┬" + "─" * 20 + "┬" + "─" * 25 + "┐")
    print(f"│{'Phenomenon':<20}│{'X-Axis':<20}│{'Sudden Change':<25}│")
    print("├" + "─" * 20 + "┼" + "─" * 20 + "┼" + "─" * 25 + "┤")
    print(f"│{'Double Descent':<20}│{'Model complexity':<20}│{'Error drops past threshold':<25}│")
    print(f"│{'Grokking':<20}│{'Training epochs':<20}│{'Test acc jumps late':<25}│")
    print(f"│{'Emergence':<20}│{'Model scale (params)':<20}│{'Capability appears':<25}│")
    print("└" + "─" * 20 + "┴" + "─" * 20 + "┴" + "─" * 25 + "┘")

    print()
    print("Key similarity: In each case, performance changes suddenly rather")
    print("than gradually as you cross a critical threshold.")
    print()

    # Simulate all three curves at equivalent points
    print("Normalized comparison (0-1 scale):")
    print()
    print(f"{'Position':<12} {'DD Error':<15} {'Grok Test Acc':<15} {'Emerge Perf':<15}")
    print("-" * 57)

    positions = ["Pre-crit", "Critical", "Post-crit"]
    dd_error = [0.7, 1.0, 0.3]  # Error: high at threshold, low after
    grok_acc = [0.1, 0.3, 0.95]  # Accuracy: low, transitioning, high
    emerge = [0.05, 0.2, 0.85]  # Capability: near-zero, transitioning, high

    for i, pos in enumerate(positions):
        print(f"{pos:<12} {dd_error[i]:<15.2f} {grok_acc[i]:<15.2f} {emerge[i]:<15.2f}")

    print()
    print("Interpretation: All three show non-linear behavior around a")
    print("critical point. This suggests deep learning involves phase")
    print("transitions similar to those in physical systems.")


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("DEEP DIVE: Surprising Phenomena in Modern Deep Learning")
    print("Numerical Examples")
    print("=" * 70)

    # Classical View
    demo_bias_variance_tradeoff()

    # Double Descent
    demo_double_descent_random_features()
    demo_interpolation_threshold_peak()

    # Grokking
    demo_grokking_simulation()
    demo_weight_norm_evolution()

    # Emergent Abilities
    demo_emergence_metric_mirage()

    # Connecting Phenomena
    demo_phase_transitions_comparison()

    print("\n" + "=" * 70)
    print("All demonstrations complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
