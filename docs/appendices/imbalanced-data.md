# Deep Dive: Handling Imbalanced Data

*Extends Module 3: Classification*

!!! note "Supplemental reading"

    Optional unless explicitly assigned in your section. Quiz and assignment content draws from the parent module, not from Deep Dives.


---

## Introduction

In many real-world classification problems, one class vastly outnumbers the other, which requires special attention to metrics, training procedures, and decision thresholds.

### Why Accuracy is Misleading

Consider fraud detection, where 99.9% of transactions are legitimate and only 0.1% are fraudulent. A model that predicts "legitimate" for every transaction achieves 99.9% accuracy while catching zero fraud. Accuracy is useless for imbalanced classes.

#### When Is It Imbalanced

A 60/40 split is typically fine; 70/30 is mild; 80/20 starts requiring attention; 90/10 likely needs specialized techniques; and 95/5 needs SMOTE, class weights, or threshold adjustment. But it's not just about ratio—absolute numbers matter (90/10 with 10,000 minority samples is fine; with 100 is problematic). The practical test: does your model learn anything about the minority class? If accuracy comes from ignoring the minority entirely, you have a problem.

### Better Metrics

Precision measures, of those we flagged as positive, how many actually were:

$$Precision = \frac{TP}{TP + FP}$$

Recall measures, of actual positives, how many we caught:

$$Recall = \frac{TP}{TP + FN}$$

The F1 Score is the harmonic mean that balances both:

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

The harmonic mean punishes extreme imbalance. For instance, Precision = 100% and Recall = 1% yields F1 = 2%, while Precision = 50% and Recall = 50% yields F1 = 50%.

The harmonic mean (as opposed to arithmetic mean) has a special property: it's dominated by the smaller value. If you have P=100% and R=1%, the arithmetic mean would be 50.5%—making it look decent. But the harmonic mean is 2%—revealing that one metric is terrible. This is exactly the behavior we want. A model that achieves high precision by predicting almost nothing (low recall) shouldn't score well. The harmonic mean enforces balance: both metrics must be reasonable for F1 to be high.

#### The Fishing Net Analogy

Imagine you're fishing for a specific type of fish (positives) in a lake (your data). Precision asks: "Of the fish in your net, what fraction are the type you wanted?" Recall asks: "Of all the target fish in the lake, what fraction did you catch?" A tight net (high threshold) catches few fish but mostly the right kind—high precision, low recall. A wide net (low threshold) catches more target fish but also lots of other fish—high recall, low precision. You can't optimize both without a better model (or more target fish).

!!! example "Numerical Example: Precision-Recall at Different Thresholds"

    ```python
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score, f1_score

    # Imbalanced dataset: 90% negative, 10% positive
    X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    for thresh in [0.1, 0.3, 0.5, 0.7]:
        y_pred = (y_proba >= thresh).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        print(f"Threshold {thresh}: Precision={prec:.0%}, Recall={rec:.0%}")
    ```

    **Output:**

    ```
    Threshold 0.1: Precision=29%, Recall=69%
    Threshold 0.3: Precision=70%, Recall=44%
    Threshold 0.5: Precision=86%, Recall=19%
    Threshold 0.7: Precision=100%, Recall=9%
    ```

    **Interpretation:** As threshold increases, precision rises (fewer false alarms) but recall falls (more missed positives). At threshold 0.1, we catch 69% of positives but 71% of our "positive" predictions are wrong. At 0.7, we're always right when we predict positive, but we miss 91% of actual positives. Choose based on business costs!

    *Source: `computations/deep_dive_imbalanced_examples.py` — `demo_precision_recall_threshold()`*


---

### The Precision-Recall Trade-off

Usually you can't maximize both. High precision means few false alarms but missing some positives, while high recall catches most positives at the cost of more false alarms. Business context determines the priority: in email marketing, high precision avoids wasting budget on unlikely responders, while in medical screening, high recall ensures you don't miss sick patients.

---

### Resampling: SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic minority examples by interpolating between existing minority points. This approach is better than simple duplication because it generates new, plausible data points rather than exact copies.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### How SMOTE Fills in the Gaps

SMOTE does not just copy existing minority examples (which would teach the model to memorize specific points). Instead, it creates synthetic examples by (i) picking a minority example, (ii) finding its k nearest minority neighbors, (iii) drawing a line to one neighbor, and (iv) placing a new point somewhere along that line. This "fills in" the feature space between existing minority examples, helping the model learn the general region where minority examples live rather than just memorizing specific cases. Think of it as sketching between the dots to reveal the underlying shape.

Only apply SMOTE to training data, never to test data. The test set must reflect real-world conditions—your deployed model will face the true class distribution. SMOTE is a training trick to help the model learn about the minority class, not a data transformation. The correct workflow is to (i) split data first, (ii) apply SMOTE only to the training set, (iii) evaluate on the original, imbalanced test set, and (iv) use appropriate metrics (F1, precision, recall) that work for imbalanced data.

---

### Class Weights

Many algorithms have built-in support for class weighting, which increases the penalty for misclassifying the minority class and is often simpler than resampling.

```python
model = LogisticRegression(class_weight='balanced')
model = DecisionTreeClassifier(class_weight='balanced')
```

!!! example "Numerical Example: Effect of Class Weights"

    ```python
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import recall_score, f1_score

    # Severely imbalanced: 95% negative, 5% positive
    X, y = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Without class weights
    model_uw = LogisticRegression(class_weight=None, random_state=42, max_iter=1000)
    model_uw.fit(X_train, y_train)
    pred_uw = model_uw.predict(X_test)

    # With balanced class weights
    model_w = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    model_w.fit(X_train, y_train)
    pred_w = model_w.predict(X_test)

    print(f"No weights:  Recall={recall_score(y_test, pred_uw):.0%}, F1={f1_score(y_test, pred_uw):.0%}")
    print(f"Balanced:    Recall={recall_score(y_test, pred_w):.0%}, F1={f1_score(y_test, pred_w):.0%}")
    ```

    **Output:**

    ```
    No weights:  Recall=0%, F1=0%
    Balanced:    Recall=78%, F1=19%
    ```

    **Interpretation:** Without weights, the model learns to predict "negative" for everything—achieving 96% accuracy by ignoring the minority class entirely (0% recall). With balanced weights, the model actually tries to find positives, achieving 78% recall. Accuracy drops because of more false positives, but F1 improves because we're actually solving the problem!

    *Source: `computations/deep_dive_imbalanced_examples.py` — `demo_class_weights()`*


---

### Threshold Adjustment

Rather than accepting the default 0.5 threshold, you can set it to any value that suits your business needs.

```python
y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.3  # Instead of 0.5
y_pred = (y_proba >= threshold).astype(int)
```

A lower threshold causes the model to predict positive more often, which increases recall at the cost of lower precision. The optimal threshold depends on the relative costs of false positives and false negatives in your particular application.

---

### Business Context Examples

The appropriate balance between precision and recall depends on the specific business application.

| Domain | Priority | Reason |
|--------|----------|--------|
| Fraud Detection | High recall | Cost of fraud >> investigation cost |
| Medical Diagnosis | High recall | Don't miss sick patients |
| Churn Prediction | Balance | Retention cost vs customer value |
| Manufacturing QC | Depends | Defect severity vs discard cost |

---

## Common Misconceptions

Several common misunderstandings about handling imbalanced data can undermine model development.

| Misconception | Reality |
|--------------|---------|
| "Always balance classes to 50-50" | Optimal ratio depends on the problem. Original distribution may be meaningful. |
| "SMOTE is always better than oversampling" | SMOTE can create unrealistic synthetic examples. Test both. |
| "Class weights and resampling do the same thing" | Similar effect but different mechanisms. Results can differ. |
| "Imbalanced data is always a problem" | If minority class is well-separated, imbalance may not hurt. Always check metrics. |
