"""
Module 3: Classification - Numerical Examples

This script demonstrates key concepts from Module 3 with concrete calculations.
Run with: pixi run python course-companion-computations/module3_examples.py

References in Course Companion:
- demo_sigmoid_function()            -> Module 3, Section 3.1 (Sigmoid Function)
- demo_cost_based_threshold()        -> Module 3, Section 3.1 (Decision Thresholds)
- demo_roc_curve_construction()      -> Module 3, Section 3.1 (ROC Curves)
- demo_gini_impurity_calculation()   -> Module 3, Section 3.2 (Splitting Criteria)
- demo_tree_overfitting()            -> Module 3, Section 3.2 (Overfitting)
- demo_feature_importance()          -> Module 3, Section 3.2 (Feature Importance)

Last updated: 2026-02-22
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def demo_sigmoid_function():
    """
    Show how the sigmoid function maps any real number to a probability.
    Demonstrates the S-curve behavior at different z values.
    """
    print("=" * 60)
    print("SIGMOID FUNCTION IN ACTION")
    print("=" * 60)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    z_values = [-6, -4, -2, -1, 0, 1, 2, 4, 6]

    print(f"\nThe sigmoid function: σ(z) = 1 / (1 + e^(-z))")
    print(f"\n{'z':>6} {'σ(z)':>10} {'Interpretation':>30}")
    print("-" * 50)

    for z in z_values:
        prob = sigmoid(z)
        if prob < 0.01:
            interp = "Very confident: NOT positive"
        elif prob < 0.10:
            interp = "Likely not positive"
        elif prob < 0.40:
            interp = "Leaning negative"
        elif prob < 0.60:
            interp = "Uncertain (near 50-50)"
        elif prob < 0.90:
            interp = "Leaning positive"
        elif prob < 0.99:
            interp = "Likely positive"
        else:
            interp = "Very confident: positive"

        print(f"{z:>6} {prob:>10.4f} {interp:>30}")

    print(f"\nKey insights:")
    print(f"  - At z=0, probability is exactly 0.5 (maximum uncertainty)")
    print(f"  - Large positive z → probability approaches 1 (but never reaches it)")
    print(f"  - Large negative z → probability approaches 0 (but never reaches it)")
    print(f"  - The 'action zone' is roughly z ∈ [-4, 4] where probability changes meaningfully")


def demo_cost_based_threshold():
    """
    Show how to calculate optimal threshold from business costs.
    Fraud detection example with asymmetric costs.
    """
    print("\n" + "=" * 60)
    print("COST-BASED THRESHOLD SELECTION")
    print("=" * 60)

    # Business scenario
    cost_fp = 50  # Cost of investigating a false alarm
    cost_fn = 500  # Cost of missing actual fraud

    # Calculate optimal threshold
    optimal_threshold = cost_fp / (cost_fp + cost_fn)

    print(f"\nFraud detection scenario:")
    print(f"  - Cost of false positive (investigate non-fraud): ${cost_fp}")
    print(f"  - Cost of false negative (miss actual fraud): ${cost_fn}")

    print(f"\nOptimal threshold formula: t* = C_FP / (C_FP + C_FN)")
    print(f"  t* = {cost_fp} / ({cost_fp} + {cost_fn})")
    print(f"  t* = {cost_fp} / {cost_fp + cost_fn}")
    print(f"  t* = {optimal_threshold:.3f}")

    print(f"\nInterpretation: Predict fraud for any transaction with P(fraud) > {optimal_threshold:.1%}")

    # Show effect on predictions
    print(f"\nExample predictions at different thresholds:")
    print(f"{'Predicted P(fraud)':>20} {'Default (0.5)':>15} {'Optimal ({:.2f})':>15}".format(optimal_threshold))
    print("-" * 55)

    probabilities = [0.05, 0.10, 0.20, 0.40, 0.60]
    for p in probabilities:
        default_pred = "Flag" if p >= 0.5 else "Clear"
        optimal_pred = "Flag" if p >= optimal_threshold else "Clear"
        print(f"{p:>20.2f} {default_pred:>15} {optimal_pred:>15}")

    print(f"\nWhen fraud costs 10x more than investigation, we're much more aggressive")
    print(f"in flagging transactions. A 10% fraud probability is worth investigating!")


def demo_roc_curve_construction():
    """
    Show step-by-step construction of an ROC curve.
    Uses a small dataset to make the mechanics clear.
    """
    print("\n" + "=" * 60)
    print("BUILDING AN ROC CURVE STEP BY STEP")
    print("=" * 60)

    # Small example dataset
    np.random.seed(42)
    true_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    # Predicted probabilities (model output)
    pred_proba = np.array([0.95, 0.85, 0.70, 0.45, 0.60, 0.40, 0.35, 0.20, 0.10, 0.05])

    print(f"\nDataset: 10 examples (4 positive, 6 negative)")
    print(f"\n{'Example':>8} {'True Label':>12} {'P(positive)':>12}")
    print("-" * 35)
    for i, (label, prob) in enumerate(zip(true_labels, pred_proba)):
        print(f"{i+1:>8} {label:>12} {prob:>12.2f}")

    # Calculate ROC points at different thresholds
    thresholds = [1.0, 0.95, 0.85, 0.70, 0.60, 0.45, 0.40, 0.35, 0.20, 0.10, 0.05, 0.0]

    print(f"\nROC curve construction (varying threshold):")
    print(f"{'Threshold':>10} {'TP':>5} {'FP':>5} {'TPR':>8} {'FPR':>8}")
    print("-" * 40)

    for thresh in thresholds:
        predictions = (pred_proba >= thresh).astype(int)
        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        tn = np.sum((predictions == 0) & (true_labels == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"{thresh:>10.2f} {tp:>5} {fp:>5} {tpr:>8.2f} {fpr:>8.2f}")

    print(f"\nThe ROC curve plots TPR vs FPR as threshold varies:")
    print(f"  - Start at (0,0): threshold=1.0, predict nothing positive")
    print(f"  - End at (1,1): threshold=0.0, predict everything positive")
    print(f"  - Perfect model: goes up left side then across top")
    print(f"  - Random model: diagonal line from (0,0) to (1,1)")


def demo_gini_impurity_calculation():
    """
    Hand-calculate Gini impurity for different node compositions.
    Shows how purity affects impurity scores.
    """
    print("\n" + "=" * 60)
    print("CALCULATING GINI IMPURITY")
    print("=" * 60)

    def gini(class_proportions):
        """Calculate Gini impurity from class proportions."""
        return 1 - sum(p**2 for p in class_proportions)

    scenarios = [
        ("Pure node (all class A)", [1.0, 0.0]),
        ("Pure node (all class B)", [0.0, 1.0]),
        ("60-40 split", [0.6, 0.4]),
        ("50-50 split (maximum impurity)", [0.5, 0.5]),
        ("80-20 split", [0.8, 0.2]),
        ("90-10 split", [0.9, 0.1]),
    ]

    print(f"\nGini = 1 - Σ(p_i)²")
    print(f"\n{'Scenario':>35} {'p_A':>6} {'p_B':>6} {'Gini':>8}")
    print("-" * 60)

    for name, props in scenarios:
        g = gini(props)
        print(f"{name:>35} {props[0]:>6.2f} {props[1]:>6.2f} {g:>8.4f}")

    # Show a split calculation
    print(f"\n" + "-" * 60)
    print(f"Example: Evaluating a split")
    print(f"-" * 60)

    # Parent node
    parent_props = [0.5, 0.5]
    parent_gini = gini(parent_props)
    parent_samples = 100

    # After split
    left_props = [0.8, 0.2]
    left_gini = gini(left_props)
    left_samples = 60

    right_props = [0.1, 0.9]
    right_gini = gini(right_props)
    right_samples = 40

    # Weighted average
    weighted_child_gini = (left_samples * left_gini + right_samples * right_gini) / parent_samples
    information_gain = parent_gini - weighted_child_gini

    print(f"\nParent node: 100 samples, 50% class A, 50% class B")
    print(f"  Gini = 1 - (0.5² + 0.5²) = 1 - 0.50 = {parent_gini:.4f}")

    print(f"\nAfter split on feature X > 5:")
    print(f"  Left child: 60 samples, 80% A, 20% B → Gini = {left_gini:.4f}")
    print(f"  Right child: 40 samples, 10% A, 90% B → Gini = {right_gini:.4f}")

    print(f"\nWeighted child Gini: (60×{left_gini:.2f} + 40×{right_gini:.2f}) / 100 = {weighted_child_gini:.4f}")
    print(f"Information gain: {parent_gini:.4f} - {weighted_child_gini:.4f} = {information_gain:.4f}")
    print(f"\nThis split reduces impurity by {information_gain:.4f}. The tree picks the split with highest gain.")


def demo_tree_overfitting():
    """
    Compare deep vs shallow decision trees on train/test accuracy.
    Shows how unlimited depth leads to memorization.
    """
    print("\n" + "=" * 60)
    print("DECISION TREE OVERFITTING")
    print("=" * 60)

    # Create dataset with some noise
    np.random.seed(42)
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=2,
        flip_y=0.1,  # Add 10% label noise
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    print(f"\nDataset: 300 samples, 10 features (5 informative)")
    print(f"Train set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    print(f"Label noise: 10% (some labels are flipped)")

    depths = [2, 3, 5, 10, None]
    results = []

    print(f"\n{'Max Depth':>12} {'Train Acc':>12} {'Test Acc':>12} {'Gap':>10} {'Leaves':>10}")
    print("-" * 60)

    for depth in depths:
        tree = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42,
        )
        tree.fit(X_train, y_train)

        train_acc = tree.score(X_train, y_train)
        test_acc = tree.score(X_test, y_test)
        gap = train_acc - test_acc
        n_leaves = tree.get_n_leaves()

        depth_str = str(depth) if depth else "None"
        print(f"{depth_str:>12} {train_acc:>12.1%} {test_acc:>12.1%} {gap:>10.1%} {n_leaves:>10}")

        results.append({
            'depth': depth,
            'train': train_acc,
            'test': test_acc,
            'leaves': n_leaves,
        })

    # Find best test accuracy
    best = max(results, key=lambda x: x['test'])
    unlimited = [r for r in results if r['depth'] is None][0]

    print(f"\nKey observations:")
    print(f"  - Unlimited depth achieves {unlimited['train']:.1%} train accuracy (memorization!)")
    print(f"  - But test accuracy is only {unlimited['test']:.1%}")
    print(f"  - Best test performance: depth={best['depth']} with {best['test']:.1%}")
    print(f"  - The unlimited tree has {unlimited['leaves']} leaves for {len(y_train)} training samples")
    print(f"\n100% training accuracy with 10% label noise = memorizing the noise!")


def demo_feature_importance():
    """
    Show how feature importance is calculated in decision trees.
    Demonstrates importance from a fitted tree.
    """
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE CALCULATION")
    print("=" * 60)

    np.random.seed(42)

    # Create features with known relationships
    n_samples = 500
    x_strong = np.random.randn(n_samples)  # Strong predictor
    x_medium = np.random.randn(n_samples)  # Medium predictor
    x_weak = np.random.randn(n_samples)    # Weak predictor
    x_noise = np.random.randn(n_samples)   # Pure noise

    # Create target based on features with different strengths
    y = (2.0 * x_strong + 1.0 * x_medium + 0.3 * x_weak + np.random.randn(n_samples) * 0.5 > 0).astype(int)

    X = np.column_stack([x_strong, x_medium, x_weak, x_noise])
    feature_names = ['strong_predictor', 'medium_predictor', 'weak_predictor', 'noise']

    # Fit tree
    tree = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    )
    tree.fit(X, y)

    print(f"\nDataset design:")
    print(f"  - strong_predictor: coefficient 2.0 in true model")
    print(f"  - medium_predictor: coefficient 1.0 in true model")
    print(f"  - weak_predictor: coefficient 0.3 in true model")
    print(f"  - noise: not in true model (pure random)")

    print(f"\nFeature importance from fitted tree:")
    print(f"{'Feature':>20} {'Importance':>12}")
    print("-" * 35)

    importances = tree.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    for idx in sorted_idx:
        print(f"{feature_names[idx]:>20} {importances[idx]:>12.3f}")

    print(f"\nInterpretation:")
    print(f"  - Importance sums to 1.0 (relative measure)")
    print(f"  - Reflects total impurity reduction × samples at each split")
    print(f"  - Strong predictor dominates (used early, affects most samples)")
    print(f"  - Noise has low/zero importance (doesn't improve predictions)")

    # Show caveat about correlated features
    print(f"\nCaveat: Correlated features split importance between them!")
    print(f"If two features are correlated, the tree may use either one.")
    print(f"Importance doesn't indicate causation—just predictive value.")


if __name__ == "__main__":
    demo_sigmoid_function()
    demo_cost_based_threshold()
    demo_roc_curve_construction()
    demo_gini_impurity_calculation()
    demo_tree_overfitting()
    demo_feature_importance()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
