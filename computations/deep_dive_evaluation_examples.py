"""
Deep Dive: Model Evaluation and Validation - Numerical Examples

This script demonstrates key concepts from the Model Evaluation Deep Dive
with concrete calculations.
Run with: pixi run python course-companion-computations/deep_dive_evaluation_examples.py

References in Course Companion:
- demo_confusion_matrix_metrics() -> Deep Dive Evaluation, Classification Metrics section
- demo_cv_variance_reduction()    -> Deep Dive Evaluation, Cross-Validation section

Last updated: 2026-02-22
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression


def demo_confusion_matrix_metrics():
    """
    Calculate precision, recall, F1 from confusion matrix values.
    """
    print("=" * 60)
    print("CONFUSION MATRIX METRICS CALCULATION")
    print("=" * 60)

    # Using the values from the diagram
    TP = 85
    FN = 15
    FP = 10
    TN = 90

    total = TP + FN + FP + TN
    actual_positive = TP + FN
    actual_negative = FP + TN
    predicted_positive = TP + FP
    predicted_negative = FN + TN

    accuracy = (TP + TN) / total
    precision = TP / predicted_positive
    recall = TP / actual_positive
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f"\nConfusion Matrix (from course diagram):")
    print(f"                 Predicted")
    print(f"                 Pos   Neg")
    print(f"  Actual Pos  |  {TP:3d}   {FN:3d}  | = {actual_positive} actual fraud")
    print(f"  Actual Neg  |  {FP:3d}   {TN:3d}  | = {actual_negative} actual legitimate")
    print(f"               ─────────────")
    print(f"                 {predicted_positive:3d}   {predicted_negative:3d}    Total: {total}")

    print(f"\nMetrics:")
    print(f"  Accuracy  = (TP + TN) / Total = ({TP} + {TN}) / {total} = {accuracy:.1%}")
    print(f"  Precision = TP / (TP + FP) = {TP} / {predicted_positive} = {precision:.1%}")
    print(f"  Recall    = TP / (TP + FN) = {TP} / {actual_positive} = {recall:.1%}")
    print(f"  F1 Score  = 2 * P * R / (P + R) = {f1:.1%}")

    print(f"\nBusiness interpretation (fraud detection):")
    print(f"  - We correctly identify {recall:.0%} of actual fraud cases (recall)")
    print(f"  - When we flag something as fraud, we're right {precision:.0%} of the time (precision)")
    print(f"  - We miss {FN} fraud cases (false negatives) - these cost us money")
    print(f"  - We waste resources investigating {FP} false alarms (false positives)")


def demo_cv_variance_reduction():
    """
    Demonstrate how cross-validation reduces variance compared to single splits.
    """
    from sklearn.model_selection import StratifiedKFold

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION VARIANCE REDUCTION")
    print("=" * 60)

    # Create dataset
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )

    # Run many single splits to show variance
    single_split_scores = []
    for seed in range(50):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
        )
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        single_split_scores.append(model.score(X_test, y_test))

    single_split_scores = np.array(single_split_scores)

    # Show the 5 individual fold scores from one CV run
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(random_state=42, max_iter=1000)
    cv_scores = cross_val_score(model, X, y, cv=cv)

    print(f"\nDataset: 200 samples, 10 features")

    print(f"\n50 different random single 80/20 train/test splits:")
    print(f"  Mean accuracy: {single_split_scores.mean():.1%}")
    print(f"  Std deviation: {single_split_scores.std():.1%}")
    print(f"  Range: {single_split_scores.min():.1%} to {single_split_scores.max():.1%}")
    print(f"  (Any single split could report anywhere in this range)")

    print(f"\n5-fold cross-validation (one run):")
    print(f"  Individual fold scores: {', '.join(f'{s:.1%}' for s in cv_scores)}")
    print(f"  Mean: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    print(f"\nThe CV mean ({cv_scores.mean():.1%}) is close to the true average ({single_split_scores.mean():.1%})")
    print(f"A single split could have reported {single_split_scores.max():.1%} (lucky)")
    print(f"or {single_split_scores.min():.1%} (unlucky) - a {(single_split_scores.max() - single_split_scores.min()):.0%} range!")
    print("\nCV gives you confidence intervals (mean ± std) instead of")
    print("a single point estimate that might be lucky or unlucky.")


if __name__ == "__main__":
    demo_confusion_matrix_metrics()
    demo_cv_variance_reduction()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
