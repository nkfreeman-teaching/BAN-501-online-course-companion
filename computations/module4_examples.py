"""
Module 4: Ensemble Methods - Numerical Examples

This script demonstrates key concepts from Module 4 with concrete calculations.
Run with: pixi run python course-companion-computations/module4_examples.py

References in Course Companion:
- demo_ensemble_variance_correlation()  -> Module 4, Section 4.1 (Ensemble Variance)
- demo_bootstrap_sampling()             -> Module 4, Section 4.2 (Bootstrap Aggregating)
- demo_rf_vs_single_tree()              -> Module 4, Section 4.2 (Random Forests)
- demo_oob_vs_cv()                      -> Module 4, Section 4.2 (OOB Error)
- demo_gradient_boosting_steps()        -> Module 4, Section 4.3 (Gradient Boosting)
- demo_learning_rate_effects()          -> Module 4, Section 4.3 (Learning Rate)
- demo_early_stopping()                 -> Module 4, Section 4.3 (Early Stopping)
Last updated: 2026-01-02
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)


def demo_ensemble_variance_correlation():
    """
    Show how correlation between models affects ensemble variance.
    Demonstrates the formula: Var(ensemble) = ρσ² + (1-ρ)σ²/n
    """
    print("=" * 60)
    print("ENSEMBLE VARIANCE AND CORRELATION")
    print("=" * 60)

    # Parameters
    individual_variance = 100  # σ² = 100 (arbitrary units)
    n_models = 10

    print(f"\nSetup: {n_models} models, each with variance σ² = {individual_variance}")
    print(f"\nFormula: Var(ensemble) = ρσ² + (1-ρ)σ²/n")
    print(f"\n{'Correlation (ρ)':>15} {'Ensemble Var':>15} {'Reduction':>15}")
    print("-" * 50)

    correlations = [0.0, 0.25, 0.5, 0.75, 1.0]

    for rho in correlations:
        # Ensemble variance formula
        ensemble_var = (
            rho * individual_variance
            + (1 - rho) * individual_variance / n_models
        )
        reduction = (1 - ensemble_var / individual_variance) * 100

        print(f"{rho:>15.2f} {ensemble_var:>15.1f} {reduction:>14.1f}%")

    print(f"\nKey insights:")
    print(f"  - ρ=0 (independent): Variance drops to σ²/n = {individual_variance/n_models:.1f} ({(1-1/n_models)*100:.0f}% reduction)")
    print(f"  - ρ=1 (identical): No reduction at all (averaging clones doesn't help)")
    print(f"  - ρ=0.5 (typical): Still get {(1 - (0.5*individual_variance + 0.5*individual_variance/n_models)/individual_variance)*100:.0f}% reduction")
    print(f"\nThis is why Random Forest samples features—to reduce ρ between trees!")


def demo_bootstrap_sampling():
    """
    Demonstrate bootstrap sampling and verify the ~63.2% unique observation rate.
    """
    print("\n" + "=" * 60)
    print("BOOTSTRAP SAMPLING IN ACTION")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 1000
    n_bootstrap_samples = 100

    print(f"\nOriginal dataset: {n_samples} observations")
    print(f"Creating {n_bootstrap_samples} bootstrap samples...")

    unique_fractions = []

    for i in range(n_bootstrap_samples):
        # Bootstrap sample: sample with replacement
        bootstrap_indices = np.random.choice(
            n_samples,
            size=n_samples,
            replace=True,
        )
        unique_count = len(np.unique(bootstrap_indices))
        unique_fractions.append(unique_count / n_samples)

    mean_unique = np.mean(unique_fractions)
    std_unique = np.std(unique_fractions)
    theoretical = 1 - np.exp(-1)  # 1 - e^(-1) ≈ 0.632

    print(f"\nResults across {n_bootstrap_samples} bootstrap samples:")
    print(f"  Mean unique fraction:     {mean_unique:.3f}")
    print(f"  Std deviation:            {std_unique:.3f}")
    print(f"  Theoretical (1 - e⁻¹):    {theoretical:.3f}")
    print(f"  Difference from theory:   {abs(mean_unique - theoretical):.4f}")

    # Show one specific bootstrap sample
    print(f"\nExample: One bootstrap sample breakdown:")
    example_indices = np.random.choice(
        n_samples,
        size=n_samples,
        replace=True,
    )
    unique_in_sample = len(np.unique(example_indices))
    oob_count = n_samples - unique_in_sample

    print(f"  Unique observations (in-bag):  {unique_in_sample} ({unique_in_sample/n_samples:.1%})")
    print(f"  Never selected (out-of-bag):   {oob_count} ({oob_count/n_samples:.1%})")

    print(f"\nWhy ~36.8% are out-of-bag:")
    print(f"  P(row i never selected) = (1 - 1/n)^n")
    print(f"  As n → ∞: (1 - 1/n)^n → e⁻¹ ≈ 0.368")
    print(f"  So ~36.8% out-of-bag, ~63.2% in-bag")


def demo_rf_vs_single_tree():
    """
    Compare Random Forest to a single decision tree on the same data.
    Shows variance reduction through ensemble.
    """
    print("\n" + "=" * 60)
    print("RANDOM FOREST VS SINGLE TREE")
    print("=" * 60)

    np.random.seed(42)

    # Create a moderately complex classification problem
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    # Run multiple train/test splits to measure variance
    n_trials = 20
    single_tree_scores = []
    rf_scores = []

    print(f"\nRunning {n_trials} random train/test splits...")

    for trial in range(n_trials):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.3,
            random_state=trial,
        )

        # Single deep tree
        tree = DecisionTreeClassifier(
            max_depth=None,
            random_state=42,
        )
        tree.fit(X_train, y_train)
        single_tree_scores.append(tree.score(X_test, y_test))

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        rf_scores.append(rf.score(X_test, y_test))

    print(f"\nResults across {n_trials} different train/test splits:")
    print(f"{'Model':>20} {'Mean Acc':>12} {'Std Dev':>12} {'Min':>10} {'Max':>10}")
    print("-" * 70)
    print(f"{'Single Tree':>20} {np.mean(single_tree_scores):>12.3f} {np.std(single_tree_scores):>12.3f} "
          f"{np.min(single_tree_scores):>10.3f} {np.max(single_tree_scores):>10.3f}")
    print(f"{'Random Forest (100)':>20} {np.mean(rf_scores):>12.3f} {np.std(rf_scores):>12.3f} "
          f"{np.min(rf_scores):>10.3f} {np.max(rf_scores):>10.3f}")

    variance_reduction = (1 - np.var(rf_scores) / np.var(single_tree_scores)) * 100

    print(f"\nKey observations:")
    print(f"  - Random Forest variance is {variance_reduction:.0f}% lower than single tree")
    print(f"  - RF mean accuracy is {(np.mean(rf_scores) - np.mean(single_tree_scores))*100:.1f} points higher")
    print(f"  - RF is more stable: predictions vary less across different splits")


def demo_oob_vs_cv():
    """
    Show that OOB error approximates cross-validation error.
    """
    print("\n" + "=" * 60)
    print("OOB ERROR VS CROSS-VALIDATION")
    print("=" * 60)

    np.random.seed(42)

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    # Random Forest with OOB scoring
    rf = RandomForestClassifier(
        n_estimators=100,
        oob_score=True,
        random_state=42,
    )
    rf.fit(X, y)
    oob_accuracy = rf.oob_score_

    # Cross-validation for comparison
    cv_scores = cross_val_score(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        X, y,
        cv=5,
    )

    print(f"\nRandom Forest with 100 trees on {len(X)} samples:")
    print(f"\n{'Metric':>25} {'Value':>15}")
    print("-" * 45)
    print(f"{'OOB Accuracy':>25} {oob_accuracy:>15.4f}")
    print(f"{'5-Fold CV Mean':>25} {np.mean(cv_scores):>15.4f}")
    print(f"{'5-Fold CV Std':>25} {np.std(cv_scores):>15.4f}")
    print(f"{'Difference (OOB - CV)':>25} {oob_accuracy - np.mean(cv_scores):>15.4f}")

    print(f"\nWhy OOB works:")
    print(f"  - Each tree leaves out ~36.8% of data (out-of-bag)")
    print(f"  - For any observation, ~36.8% of trees never trained on it")
    print(f"  - OOB prediction uses only those trees → honest estimate")
    print(f"  - It's like automatic cross-validation, but FREE!")
    print(f"\nOOB is especially useful when:")
    print(f"  - Dataset is large (CV would be slow)")
    print(f"  - You want quick hyperparameter feedback")


def demo_gradient_boosting_steps():
    """
    Show gradient boosting learning step by step with residuals.
    """
    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING STEP BY STEP")
    print("=" * 60)

    np.random.seed(42)

    # Simple regression problem for clarity
    n_samples = 100
    X = np.random.uniform(
        low=0,
        high=10,
        size=(n_samples, 1),
    )
    # True function: y = 2x + 3 + noise
    y = 2 * X.ravel() + 3 + np.random.normal(
        loc=0,
        scale=2,
        size=n_samples,
    )

    print(f"\nTrue relationship: y = 2x + 3 + noise")
    print(f"Training gradient boosting with learning_rate=0.3, max_depth=1")

    # Manual gradient boosting steps
    learning_rate = 0.3
    max_depth = 1
    n_rounds = 5

    # Initial prediction: mean of y
    current_prediction = np.full(n_samples, np.mean(y))

    print(f"\n{'Round':>6} {'Mean Residual':>15} {'Residual Std':>15} {'MSE':>12}")
    print("-" * 55)

    # Initial state
    residuals = y - current_prediction
    mse = np.mean(residuals ** 2)
    print(f"{'Init':>6} {np.mean(residuals):>15.4f} {np.std(residuals):>15.4f} {mse:>12.4f}")

    for round_num in range(1, n_rounds + 1):
        # Calculate residuals (negative gradient for MSE)
        residuals = y - current_prediction

        # Fit a tree to residuals
        tree = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=42,
        )
        tree.fit(X, residuals)

        # Update predictions
        tree_prediction = tree.predict(X)
        current_prediction = current_prediction + learning_rate * tree_prediction

        # Calculate new residuals and MSE
        new_residuals = y - current_prediction
        mse = np.mean(new_residuals ** 2)

        print(f"{round_num:>6} {np.mean(new_residuals):>15.4f} {np.std(new_residuals):>15.4f} {mse:>12.4f}")

    print(f"\nWhat's happening at each step:")
    print(f"  1. Calculate residuals: (actual - predicted)")
    print(f"  2. Fit a shallow tree to predict those residuals")
    print(f"  3. Add learning_rate × tree_prediction to current prediction")
    print(f"  4. Repeat—each tree corrects remaining errors")
    print(f"\nThe residual std decreases as boosting 'learns' the pattern!")


def demo_learning_rate_effects():
    """
    Compare different learning rates with corresponding n_estimators.
    Shows the tradeoff: lower rate + more trees = better but slower.
    """
    print("\n" + "=" * 60)
    print("LEARNING RATE EFFECTS ON BOOSTING")
    print("=" * 60)

    np.random.seed(42)

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
    )

    # Different learning rate / n_estimators combinations
    configs = [
        (0.3, 50, "High rate, few trees"),
        (0.1, 150, "Medium rate, medium trees"),
        (0.03, 500, "Low rate, many trees"),
    ]

    print(f"\nComparing learning rate / n_estimators combinations:")
    print(f"{'Config':>30} {'LR':>8} {'Trees':>8} {'Train Acc':>12} {'Test Acc':>12}")
    print("-" * 75)

    for lr, n_est, label in configs:
        gb = GradientBoostingClassifier(
            learning_rate=lr,
            n_estimators=n_est,
            max_depth=3,
            random_state=42,
        )
        gb.fit(X_train, y_train)

        train_acc = gb.score(X_train, y_train)
        test_acc = gb.score(X_test, y_test)

        print(f"{label:>30} {lr:>8.2f} {n_est:>8} {train_acc:>12.4f} {test_acc:>12.4f}")

    print(f"\nKey insight:")
    print(f"  - Lower learning rate + more trees often gives best test accuracy")
    print(f"  - High learning rate can overshoot optimal solution")
    print(f"  - Tradeoff: lower rate = slower training, but better generalization")
    print(f"\nRule of thumb:")
    print(f"  - Start with learning_rate=0.1, n_estimators=100")
    print(f"  - If overfitting: lower the rate, increase trees")
    print(f"  - Always use early stopping to find optimal n_estimators")


def demo_early_stopping():
    """
    Demonstrate early stopping preventing overfitting in boosting.
    """
    print("\n" + "=" * 60)
    print("EARLY STOPPING IN ACTION")
    print("=" * 60)

    np.random.seed(42)

    # Create data where overfitting is likely
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=5,
        n_redundant=10,
        n_clusters_per_class=3,
        n_classes=2,
        random_state=42,
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.4,
        random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
    )

    # Train with many estimators and track performance
    max_estimators = 300
    gb = GradientBoostingClassifier(
        n_estimators=max_estimators,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
    )
    gb.fit(X_train, y_train)

    # Get staged predictions for train and validation
    train_scores = []
    val_scores = []

    for i, (train_pred, val_pred) in enumerate(zip(
        gb.staged_predict(X_train),
        gb.staged_predict(X_val)
    )):
        train_scores.append(np.mean(train_pred == y_train))
        val_scores.append(np.mean(val_pred == y_val))

    # Find best validation score
    best_n = np.argmax(val_scores) + 1
    best_val_score = val_scores[best_n - 1]

    print(f"\nTraining GradientBoosting with up to {max_estimators} trees...")
    print(f"\n{'Stage':>8} {'Train Acc':>12} {'Val Acc':>12}")
    print("-" * 35)

    checkpoints = [10, 25, 50, 100, 150, 200, 250, 300]
    for n in checkpoints:
        if n <= len(train_scores):
            print(f"{n:>8} {train_scores[n-1]:>12.4f} {val_scores[n-1]:>12.4f}")

    print(f"\nOptimal stopping point: {best_n} trees")
    print(f"Best validation accuracy: {best_val_score:.4f}")
    print(f"Final validation accuracy (300 trees): {val_scores[-1]:.4f}")
    print(f"Overfit penalty: {(best_val_score - val_scores[-1])*100:.1f} percentage points lost")

    # Test with optimal number
    gb_optimal = GradientBoostingClassifier(
        n_estimators=best_n,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
    )
    gb_optimal.fit(X_train, y_train)
    test_optimal = gb_optimal.score(X_test, y_test)

    gb_full = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
    )
    gb_full.fit(X_train, y_train)
    test_full = gb_full.score(X_test, y_test)

    print(f"\nTest set performance:")
    print(f"  With early stopping ({best_n} trees): {test_optimal:.4f}")
    print(f"  Without early stopping (300 trees):  {test_full:.4f}")

    print(f"\nWhy early stopping works:")
    print(f"  - Training accuracy keeps improving (memorization)")
    print(f"  - Validation accuracy peaks then declines (overfitting)")
    print(f"  - Stop when validation stops improving")
    print(f"  - XGBoost: early_stopping_rounds=10 does this automatically")


if __name__ == "__main__":
    demo_ensemble_variance_correlation()
    demo_bootstrap_sampling()
    demo_rf_vs_single_tree()
    demo_oob_vs_cv()
    demo_gradient_boosting_steps()
    demo_learning_rate_effects()
    demo_early_stopping()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
