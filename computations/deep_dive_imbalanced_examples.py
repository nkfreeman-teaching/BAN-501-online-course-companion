"""
Deep Dive: Handling Imbalanced Data - Numerical Examples

This script demonstrates key concepts from the Imbalanced Data Deep Dive
with concrete calculations.
Run with: pixi run python course-companion-computations/deep_dive_imbalanced_examples.py

References in Course Companion:
- demo_precision_recall_threshold()  -> Deep Dive Imbalanced, Precision-Recall section
- demo_class_weights()               -> Deep Dive Imbalanced, Class Weights section

Last updated: 2026-02-22
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


def demo_precision_recall_threshold():
    """
    Show how precision and recall change as threshold varies.
    Same model, different business outcomes.
    """
    print("=" * 60)
    print("PRECISION-RECALL AT DIFFERENT THRESHOLDS")
    print("=" * 60)

    # Create imbalanced dataset
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        weights=[0.9, 0.1],  # 90% negative, 10% positive
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    # Fit model
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )
    model.fit(X_train, y_train)

    # Get probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\nImbalanced dataset: ~90% negative, ~10% positive")
    print(f"Test set: {len(y_test)} samples, {sum(y_test)} positive ({sum(y_test)/len(y_test):.1%})")

    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"\n{'Threshold':>10} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Predicted +':>14}")
    print("-" * 65)

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        n_pred_pos = sum(y_pred)

        print(f"{thresh:>10.1f} {prec:>12.1%} {rec:>12.1%} {f1:>12.1%} {n_pred_pos:>14}")

    print(f"\nThe trade-off:")
    print(f"  - Low threshold (0.1): High recall ({recall_score(y_test, (y_proba >= 0.1).astype(int)):.0%}) but more false alarms")
    print(f"  - High threshold (0.9): High precision but miss many positives")
    print(f"  - Choice depends on business: medical screening wants high recall,")
    print(f"    email marketing wants high precision")


def demo_class_weights():
    """
    Show how class weights affect predictions on imbalanced data.
    Without weights: model ignores minority class.
    """
    print("\n" + "=" * 60)
    print("EFFECT OF CLASS WEIGHTS")
    print("=" * 60)

    # Create very imbalanced dataset
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        weights=[0.95, 0.05],  # 95% negative, 5% positive
        flip_y=0,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    print(f"\nSeverely imbalanced: ~95% negative, ~5% positive")
    print(f"Train: {sum(y_train)} positive out of {len(y_train)} ({sum(y_train)/len(y_train):.1%})")
    print(f"Test: {sum(y_test)} positive out of {len(y_test)} ({sum(y_test)/len(y_test):.1%})")

    # Model without class weights
    model_unweighted = LogisticRegression(
        class_weight=None,
        random_state=42,
        max_iter=1000,
    )
    model_unweighted.fit(X_train, y_train)
    pred_unweighted = model_unweighted.predict(X_test)

    # Model with balanced class weights
    model_weighted = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
    )
    model_weighted.fit(X_train, y_train)
    pred_weighted = model_weighted.predict(X_test)

    print(f"\n{'Metric':>15} {'No Weights':>15} {'Balanced Weights':>18}")
    print("-" * 52)

    # Calculate metrics
    acc_uw = (pred_unweighted == y_test).mean()
    acc_w = (pred_weighted == y_test).mean()

    prec_uw = precision_score(y_test, pred_unweighted, zero_division=0)
    prec_w = precision_score(y_test, pred_weighted, zero_division=0)

    rec_uw = recall_score(y_test, pred_unweighted, zero_division=0)
    rec_w = recall_score(y_test, pred_weighted, zero_division=0)

    f1_uw = f1_score(y_test, pred_unweighted, zero_division=0)
    f1_w = f1_score(y_test, pred_weighted, zero_division=0)

    pred_pos_uw = sum(pred_unweighted)
    pred_pos_w = sum(pred_weighted)

    print(f"{'Accuracy':>15} {acc_uw:>15.1%} {acc_w:>18.1%}")
    print(f"{'Precision':>15} {prec_uw:>15.1%} {prec_w:>18.1%}")
    print(f"{'Recall':>15} {rec_uw:>15.1%} {rec_w:>18.1%}")
    print(f"{'F1 Score':>15} {f1_uw:>15.1%} {f1_w:>18.1%}")
    print(f"{'Predicted +':>15} {pred_pos_uw:>15} {pred_pos_w:>18}")

    print(f"\nKey insight:")
    print(f"  - Without weights: Model predicts mostly negative (lazy but 'accurate')")
    print(f"  - With balanced weights: Model tries to find positives (better recall)")
    print(f"  - Accuracy drops because we make more false positives")
    print(f"  - But F1 improves because we actually detect the minority class!")
    print(f"\nclass_weight='balanced' automatically scales: weight = n_samples / (n_classes × n_class_samples)")


if __name__ == "__main__":
    demo_precision_recall_threshold()
    demo_class_weights()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
