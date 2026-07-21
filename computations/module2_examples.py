"""
Module 2: Regression - Numerical Examples

This script demonstrates key concepts from Module 2 with concrete calculations.
Run with: pixi run python course-companion-computations/module2_examples.py

References in Course Companion:
- demo_gradient_descent_convergence() -> Module 2, Section 2.1 (Gradient Descent)
- demo_learning_rate_effects()        -> Module 2, Section 2.1 (Learning Rate)
- demo_vif_multicollinearity()        -> Module 2, Section 2.2 (Multicollinearity)
- demo_lasso_feature_selection()      -> Module 2, Section 2.2 (Lasso Regularization)
- demo_ridge_vs_lasso()               -> Module 2, Section 2.2 (Regularization Comparison)

Last updated: 2026-01-02
"""

import numpy as np
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def demo_gradient_descent_convergence():
    """
    Show how gradient descent iteratively converges to optimal coefficients.
    Demonstrates the algorithm finding the true parameters step by step.
    """
    print("=" * 60)
    print("GRADIENT DESCENT CONVERGENCE")
    print("=" * 60)

    # Generate data with known true parameters
    np.random.seed(42)
    n_samples = 100
    true_beta_0 = 3.0
    true_beta_1 = 2.0

    X = np.random.uniform(
        low=0,
        high=10,
        size=n_samples,
    )
    noise = np.random.normal(
        loc=0,
        scale=1.5,
        size=n_samples,
    )
    y = true_beta_0 + true_beta_1 * X + noise

    # Gradient descent implementation
    learning_rate = 0.02  # Slightly higher for faster convergence demo
    beta_0 = 0.0  # Start at zero
    beta_1 = 0.0

    history = []

    for iteration in range(501):
        # Store current state
        y_pred = beta_0 + beta_1 * X
        mse = np.mean((y - y_pred) ** 2)
        history.append({
            'iteration': iteration,
            'beta_0': beta_0,
            'beta_1': beta_1,
            'mse': mse,
        })

        # Compute gradients
        d_beta_0 = -2 / n_samples * np.sum(y - y_pred)
        d_beta_1 = -2 / n_samples * np.sum((y - y_pred) * X)

        # Update parameters
        beta_0 = beta_0 - learning_rate * d_beta_0
        beta_1 = beta_1 - learning_rate * d_beta_1

    print(f"\nTrue parameters: beta_0 = {true_beta_0}, beta_1 = {true_beta_1}")
    print(f"\nGradient descent progress (learning_rate = {learning_rate}):")
    print(f"{'Iteration':>10} {'beta_0':>10} {'beta_1':>10} {'MSE':>10}")
    print("-" * 45)

    for h in history:
        if h['iteration'] in [0, 10, 50, 100, 200, 500]:
            print(f"{h['iteration']:>10} {h['beta_0']:>10.4f} {h['beta_1']:>10.4f} {h['mse']:>10.4f}")

    print(f"\nFinal estimates: beta_0 = {history[-1]['beta_0']:.4f}, beta_1 = {history[-1]['beta_1']:.4f}")
    print(f"Error from true: beta_0 off by {abs(history[-1]['beta_0'] - true_beta_0):.4f}, "
          f"beta_1 off by {abs(history[-1]['beta_1'] - true_beta_1):.4f}")
    print("\nThe algorithm starts with random guesses and iteratively improves,")
    print("converging toward the true parameters as MSE decreases.")


def demo_learning_rate_effects():
    """
    Demonstrate how learning rate affects gradient descent convergence.
    Shows too small, just right, and too large learning rates.
    """
    print("\n" + "=" * 60)
    print("LEARNING RATE EFFECTS")
    print("=" * 60)

    # Generate simple data
    np.random.seed(42)
    n_samples = 100
    X = np.random.uniform(
        low=0,
        high=10,
        size=n_samples,
    )
    y = 3.0 + 2.0 * X + np.random.normal(
        loc=0,
        scale=1.5,
        size=n_samples,
    )

    def run_gradient_descent(X, y, learning_rate, max_iterations=1000, tolerance=1e-6):
        """Run gradient descent and return convergence info."""
        n = len(X)
        beta_0, beta_1 = 0.0, 0.0
        history = []

        for i in range(max_iterations):
            y_pred = beta_0 + beta_1 * X
            mse = np.mean((y - y_pred) ** 2)
            history.append(mse)

            # Check for divergence
            if mse > 1e10 or np.isnan(mse):
                return {
                    'converged': False,
                    'diverged': True,
                    'iterations': i,
                    'final_mse': float('inf'),
                    'beta_0': beta_0,
                    'beta_1': beta_1,
                }

            # Check convergence
            if i > 0 and abs(history[-1] - history[-2]) < tolerance:
                return {
                    'converged': True,
                    'diverged': False,
                    'iterations': i,
                    'final_mse': mse,
                    'beta_0': beta_0,
                    'beta_1': beta_1,
                }

            # Update
            d_beta_0 = -2 / n * np.sum(y - y_pred)
            d_beta_1 = -2 / n * np.sum((y - y_pred) * X)
            beta_0 = beta_0 - learning_rate * d_beta_0
            beta_1 = beta_1 - learning_rate * d_beta_1

        return {
            'converged': False,
            'diverged': False,
            'iterations': max_iterations,
            'final_mse': mse,
            'beta_0': beta_0,
            'beta_1': beta_1,
        }

    learning_rates = [0.0001, 0.01, 0.1]
    labels = ["Too small", "Just right", "Too large"]

    print(f"\nSame data, different learning rates:")
    print(f"{'Rate':>12} {'Status':>15} {'Iterations':>12} {'Final MSE':>12}")
    print("-" * 55)

    for lr, label in zip(learning_rates, labels):
        result = run_gradient_descent(X, y, lr)
        if result['diverged']:
            status = "DIVERGED"
            mse_str = "inf"
        elif result['converged']:
            status = "Converged"
            mse_str = f"{result['final_mse']:.4f}"
        else:
            status = "Not converged"
            mse_str = f"{result['final_mse']:.4f}"

        print(f"{lr:>12.4f} {status:>15} {result['iterations']:>12} {mse_str:>12}")

    print(f"\nInterpretation:")
    print(f"  - alpha=0.0001 (too small): Converges but requires many iterations")
    print(f"  - alpha=0.01 (just right): Converges quickly to good solution")
    print(f"  - alpha=0.1 (too large): Overshoots and diverges (loss increases!)")
    print(f"\nIf your loss keeps increasing, reduce the learning rate by 10x.")


def demo_vif_multicollinearity():
    """
    Demonstrate how VIF detects multicollinearity.
    Shows VIF values before and after adding a correlated feature.
    """
    print("\n" + "=" * 60)
    print("VIF MULTICOLLINEARITY DETECTION")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 200

    # Create independent features (standardized for cleaner VIF)
    x1 = np.random.randn(n_samples)
    x2 = np.random.randn(n_samples)
    x3 = np.random.randn(n_samples)

    # Create a feature highly correlated with x1
    x4_correlated = x1 + np.random.randn(n_samples) * 0.1  # r ~ 0.99

    # Calculate correlation
    correlation = np.corrcoef(x1, x4_correlated)[0, 1]

    # Add constant for proper VIF calculation
    const = np.ones(n_samples)

    # VIF with independent features only
    X_independent = np.column_stack([const, x1, x2, x3])
    feature_names_ind = ['x1', 'x2', 'x3']

    print(f"\nScenario 1: Three independent features")
    print(f"{'Feature':>10} {'VIF':>10}")
    print("-" * 22)
    for i, name in enumerate(feature_names_ind):
        vif = variance_inflation_factor(X_independent, i + 1)  # Skip constant
        print(f"{name:>10} {vif:>10.2f}")

    # VIF after adding correlated feature
    X_with_collinear = np.column_stack([const, x1, x2, x3, x4_correlated])
    feature_names_col = ['x1', 'x2', 'x3', 'x4 (corr with x1)']

    print(f"\nScenario 2: Added x4 which is highly correlated with x1 (r = {correlation:.3f})")
    print(f"{'Feature':>20} {'VIF':>10}")
    print("-" * 32)
    for i, name in enumerate(feature_names_col):
        vif = variance_inflation_factor(X_with_collinear, i + 1)  # Skip constant
        print(f"{name:>20} {vif:>10.2f}")

    print(f"\nInterpretation:")
    print(f"  - Independent features have VIF near 1 (no multicollinearity)")
    print(f"  - x1 and x4 now have VIF >> 10 (severe multicollinearity)")
    print(f"  - VIF > 5: moderate concern, VIF > 10: must address")
    print(f"  - Solution: remove x4, combine features, or use Ridge regression")


def demo_lasso_feature_selection():
    """
    Demonstrate how Lasso performs automatic feature selection.
    Shows coefficients going to zero as alpha increases.
    """
    print("\n" + "=" * 60)
    print("LASSO FEATURE SELECTION")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 200
    n_features = 10

    # Create features (only first 3 are truly predictive)
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'x{i+1}' for i in range(n_features)]

    # True relationship: y = 5*x1 + 3*x2 - 2*x3 + noise
    # Features x4-x10 are pure noise
    y = 5 * X[:, 0] + 3 * X[:, 1] - 2 * X[:, 2] + np.random.randn(n_samples)

    # Standardize features (required for fair regularization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try different alpha values
    alphas = [0.01, 0.1, 0.5, 1.0]

    print(f"\nTrue model: y = 5*x1 + 3*x2 - 2*x3 (x4-x10 are noise)")
    print(f"\nLasso coefficients at different alpha values:")
    print(f"{'Feature':>8}", end="")
    for alpha in alphas:
        print(f"{'alpha='+str(alpha):>12}", end="")
    print()
    print("-" * 60)

    results = {}
    for alpha in alphas:
        lasso = Lasso(
            alpha=alpha,
            random_state=42,
            max_iter=10000,
        )
        lasso.fit(X_scaled, y)
        results[alpha] = lasso.coef_

    for i, name in enumerate(feature_names):
        print(f"{name:>8}", end="")
        for alpha in alphas:
            coef = results[alpha][i]
            if abs(coef) < 0.001:
                print(f"{'0':>12}", end="")
            else:
                print(f"{coef:>12.3f}", end="")
        print()

    # Count non-zero coefficients
    print(f"\n{'Non-zero':>8}", end="")
    for alpha in alphas:
        n_nonzero = np.sum(np.abs(results[alpha]) > 0.001)
        print(f"{n_nonzero:>12}", end="")
    print()

    print(f"\nInterpretation:")
    print(f"  - At low alpha (0.01): Most coefficients retained")
    print(f"  - As alpha increases: Noise features (x4-x10) shrink to exactly 0")
    print(f"  - At alpha=1.0: Only the true predictors (x1, x2, x3) remain")
    print(f"  - Lasso performs automatic feature selection!")


def demo_ridge_vs_lasso():
    """
    Compare Ridge and Lasso on the same dataset.
    Shows different coefficient shrinkage patterns.
    """
    print("\n" + "=" * 60)
    print("RIDGE VS LASSO COMPARISON")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 200

    # Create correlated features (where Ridge excels)
    x1 = np.random.randn(n_samples)
    x2 = x1 + np.random.randn(n_samples) * 0.5  # Correlated with x1
    x3 = np.random.randn(n_samples)  # Independent
    x4 = np.random.randn(n_samples)  # Noise

    X = np.column_stack([x1, x2, x3, x4])
    feature_names = ['x1', 'x2 (corr)', 'x3', 'x4 (noise)']

    # True relationship uses both correlated features
    y = 3 * x1 + 2 * x2 + 1.5 * x3 + np.random.randn(n_samples) * 0.5

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit models
    ols = LinearRegression()
    ols.fit(X_scaled, y)

    ridge = Ridge(
        alpha=1.0,
        random_state=42,
    )
    ridge.fit(X_scaled, y)

    lasso = Lasso(
        alpha=0.1,
        random_state=42,
        max_iter=10000,
    )
    lasso.fit(X_scaled, y)

    print(f"\nTrue model: y = 3*x1 + 2*x2 + 1.5*x3 (x4 is noise)")
    print(f"Note: x1 and x2 are correlated (r = {np.corrcoef(x1, x2)[0,1]:.2f})")
    print(f"\nCoefficient comparison:")
    print(f"{'Feature':>15} {'OLS':>10} {'Ridge':>10} {'Lasso':>10}")
    print("-" * 50)

    for i, name in enumerate(feature_names):
        print(f"{name:>15} {ols.coef_[i]:>10.3f} {ridge.coef_[i]:>10.3f} {lasso.coef_[i]:>10.3f}")

    print(f"\nKey differences:")
    print(f"  - OLS: Unstable coefficients due to correlation (x1={ols.coef_[0]:.2f}, x2={ols.coef_[1]:.2f})")
    print(f"  - Ridge: Shrinks all coefficients, stabilizes correlated pairs")
    print(f"  - Lasso: May zero out one of correlated pair (keeps one, drops other)")
    print(f"\nWhen to use each:")
    print(f"  - Ridge: When all features are potentially relevant, especially with correlation")
    print(f"  - Lasso: When you want automatic feature selection (sparse model)")


if __name__ == "__main__":
    demo_gradient_descent_convergence()
    demo_learning_rate_effects()
    demo_vif_multicollinearity()
    demo_lasso_feature_selection()
    demo_ridge_vs_lasso()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
