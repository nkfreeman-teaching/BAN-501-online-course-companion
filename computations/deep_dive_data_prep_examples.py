"""
Deep Dive: Data Preparation and Feature Engineering - Numerical Examples

This script demonstrates key concepts from the Data Preparation Deep Dive
with concrete calculations.
Run with: pixi run python course-companion-computations/deep_dive_data_prep_examples.py

References in Course Companion:
- demo_feature_scaling_impact()  -> Deep Dive Data Prep, Feature Scaling section
- demo_data_leakage()            -> Deep Dive Data Prep, Data Leakage section

Last updated: 2026-02-22
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def demo_feature_scaling_impact():
    """
    Demonstrate how feature scaling affects k-NN performance
    when features have very different scales.
    """
    print("=" * 60)
    print("FEATURE SCALING IMPACT ON k-NN")
    print("=" * 60)

    # Create synthetic data with two features at very different scales
    np.random.seed(42)
    n_samples = 200

    # Feature 1: Age (20-70 years)
    age = np.random.uniform(
        low=20,
        high=70,
        size=n_samples,
    )

    # Feature 2: Income ($30,000-$200,000)
    income = np.random.uniform(
        low=30000,
        high=200000,
        size=n_samples,
    )

    # Target: Higher income AND middle age (35-55) -> class 1
    target = ((income > 80000) & (age > 35) & (age < 55)).astype(int)

    # Combine features
    X = np.column_stack([age, income])
    y = target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    # k-NN WITHOUT scaling
    knn_unscaled = KNeighborsClassifier(n_neighbors=5)
    knn_unscaled.fit(X_train, y_train)
    accuracy_unscaled = knn_unscaled.score(X_test, y_test)

    # k-NN WITH scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_scaled = KNeighborsClassifier(n_neighbors=5)
    knn_scaled.fit(X_train_scaled, y_train)
    accuracy_scaled = knn_scaled.score(X_test_scaled, y_test)

    print(f"\nDataset: {n_samples} samples with Age (20-70) and Income ($30k-$200k)")
    print(f"\nWithout scaling:")
    print(f"  Age range: {age.min():.0f} to {age.max():.0f} (range: {age.max()-age.min():.0f})")
    print(f"  Income range: ${income.min():,.0f} to ${income.max():,.0f} (range: ${income.max()-income.min():,.0f})")
    print(f"  k-NN accuracy: {accuracy_unscaled:.1%}")
    print(f"\nWith StandardScaler:")
    print(f"  Both features now have mean=0, std=1")
    print(f"  k-NN accuracy: {accuracy_scaled:.1%}")
    print(f"\nImprovement: {(accuracy_scaled - accuracy_unscaled):.1%}")
    print("\nExplanation: Without scaling, income differences dominate the distance")
    print("calculation, effectively ignoring age. Scaling gives both features equal weight.")


def demo_data_leakage():
    """
    Demonstrate how scaling before splitting causes data leakage
    and leads to overly optimistic performance estimates.

    Uses a small dataset where leakage effects are more pronounced.
    """
    print("\n" + "=" * 60)
    print("DATA LEAKAGE DEMONSTRATION")
    print("=" * 60)

    # Use a SMALL dataset where leakage effects are more visible
    # Leakage is worse with small data because test statistics
    # have more influence on overall statistics
    np.random.seed(42)

    wrong_accuracies = []
    right_accuracies = []

    # Run multiple trials to show the pattern
    for trial in range(20):
        X, y = make_classification(
            n_samples=50,  # Small dataset - leakage matters more
            n_features=15,
            n_informative=5,
            n_redundant=3,
            random_state=trial,
        )

        # WRONG: Scale ALL data, then split
        scaler_wrong = StandardScaler()
        X_scaled_wrong = scaler_wrong.fit_transform(X)
        X_train_wrong, X_test_wrong, y_train, y_test = train_test_split(
            X_scaled_wrong,
            y,
            test_size=0.3,
            random_state=trial,
        )

        model_wrong = LogisticRegression(random_state=42, max_iter=1000)
        model_wrong.fit(X_train_wrong, y_train)
        wrong_accuracies.append(model_wrong.score(X_test_wrong, y_test))

        # RIGHT: Split first, then scale training only
        X_train_right, X_test_right, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=trial,
        )

        scaler_right = StandardScaler()
        X_train_scaled_right = scaler_right.fit_transform(X_train_right)
        X_test_scaled_right = scaler_right.transform(X_test_right)

        model_right = LogisticRegression(random_state=42, max_iter=1000)
        model_right.fit(X_train_scaled_right, y_train)
        right_accuracies.append(model_right.score(X_test_scaled_right, y_test))

    wrong_mean = np.mean(wrong_accuracies)
    right_mean = np.mean(right_accuracies)

    print(f"\nDataset: 50 samples, 15 features (small - leakage matters more)")
    print(f"Average over 20 random datasets:")
    print(f"\nWRONG (scale all, then split):")
    print(f"  Mean test accuracy: {wrong_mean:.1%}")
    print(f"\nRIGHT (split first, scale train only):")
    print(f"  Mean test accuracy: {right_mean:.1%}")
    print(f"\nLeakage inflation: {(wrong_mean - right_mean):.1%}")
    print("\nThe 'wrong' approach reports higher accuracy because the model")
    print("indirectly sees information about test examples through the scaling")
    print("parameters. This gap widens with smaller data and more preprocessing.")


if __name__ == "__main__":
    demo_feature_scaling_impact()
    demo_data_leakage()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
