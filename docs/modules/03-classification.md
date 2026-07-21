# Module 3: Classification Methods

## Introduction

We've covered a lot of ground—foundations in Module 1, regression in Module 2. Now we move to classification, which is arguably even more prevalent in business applications.

Think about the decisions businesses make every day: Should we approve this loan? Is this transaction fraudulent? Will this customer cancel their subscription? Is this email spam? These are all classification problems—predicting a category, not a number.

The concepts extend to **multiclass classification** (covered in Module 6): logistic regression uses softmax instead of sigmoid; decision trees handle it naturally; evaluation uses per-class precision/recall and $C \times C$ confusion matrices. The fundamentals transfer directly—the mechanics get more complex but the reasoning stays the same.

This module covers three major areas: logistic regression (extending regression concepts to classification), decision trees (intuitive classifiers that set us up for ensemble methods), and hyperparameter optimization. Along the way we cover evaluation metrics—confusion matrices, precision, recall, F1—and how class imbalance changes which metrics are meaningful.

---

## Learning Objectives

By the end of this module, you should be able to:

1. **Explain** the mechanics and interpretation of logistic regression, including the probability → odds → log odds → linear model chain
2. **Select** appropriate evaluation metrics (accuracy, precision, recall, F1, AUC) based on business context, and explain why accuracy alone is misleading when classes are imbalanced
3. **Build** and interpret decision tree classifiers, understanding their tendency to overfit and how pruning addresses it
4. **Use** hyperparameter optimization techniques—grid search, random search, and Bayesian optimization—to improve model performance without overfitting to the validation set

---

## 3.1 Logistic Regression

Logistic regression extends the regression framework from Module 2 to predict probabilities of class membership rather than continuous values.

### Three Components: Logistic Regression

Logistic regression connects to the three-component framework from Module 2.

| Component | Logistic Regression |
|-----------|---------------------|
| **Decision Model** | $P(Y=1) = \sigma(\beta_0 + \beta_1 x_1 + ...)$ — sigmoid of linear combination |
| **Quality Measure** | Cross-entropy (log loss) — penalizes confident wrong predictions |
| **Update Method** | Gradient descent on log-likelihood |

The decision model changes from a line to a sigmoid curve, and the quality measure changes from SSE to cross-entropy—but the overall structure is identical to linear regression.

### Why Linear Regression Fails for Classification

Binary outcomes are coded as 0 or 1. If we fit a line, predictions can be less than 0 or greater than 1. "There's a -15% chance of churn" is meaningless.

The solution is to transform the output so it always falls between 0 and 1. Other functions map to (0,1)—probit, scaled tanh—but the sigmoid has unique advantages: its derivative is expressible in terms of the output (efficient gradients), its inverse is the logit (clean coefficient interpretation as log-odds), and it arises from maximum entropy principles. Tools and practices are standardized around it.

### The Sigmoid Function

The sigmoid function maps any real number to the interval (0, 1):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

where $z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ...$

#### Key Properties

The sigmoid output is always between 0 and 1, making it a valid probability. The curve is S-shaped, so small changes in x have the biggest effect near 0.5. At z=0, the output is exactly 0.5.

The math follows directly from the formula. When z is large and positive, $e^{-z} \to 0$, so $\sigma(z) \to 1$. When z is large and negative, $e^{-z} \to \infty$, so $\sigma(z) \to 0$. When z = 0, $e^{0} = 1$, so $\sigma(0) = 0.5$.

#### Why This Particular S-Shape

The sigmoid is not arbitrary—it is one of very few functions that (i) maps any real number to (0,1), (ii) is symmetric around 0.5, and (iii) has a derivative expressible in terms of itself, making gradient descent efficient. Think of z as a "confidence score": strongly negative z means the model is confident in class 0, strongly positive means confident in class 1, and z near 0 means the model is uncertain. The sigmoid converts this confidence into a probability. The "action zone" where probability changes meaningfully is roughly z ∈ [-4, 4]—outside this range, the model is essentially certain.

!!! example "Numerical Example: Sigmoid Function in Action"

    ```python
    import numpy as np

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    z_values = [-6, -4, -2, 0, 2, 4, 6]
    for z in z_values:
        prob = sigmoid(z)
        print(f"z = {z:>3}: σ(z) = {prob:.4f}")
    ```

    **Output:**

    ```
    z =  -6: σ(z) = 0.0025
    z =  -4: σ(z) = 0.0180
    z =  -2: σ(z) = 0.1192
    z =   0: σ(z) = 0.5000
    z =   2: σ(z) = 0.8808
    z =   4: σ(z) = 0.9820
    z =   6: σ(z) = 0.9975
    ```

    **Interpretation:** At z=0, probability is exactly 0.5 (maximum uncertainty). Moving to z=±4 brings probability within 2% of the extremes (0 or 1). At z=±6, the model is 99.75% confident. The "interesting range" where probability changes meaningfully is roughly z ∈ [-4, 4].

    *Source: `computations/module3_examples.py` — `demo_sigmoid_function()`*


### Understanding Odds and Log Odds

Interpreting logistic regression coefficients requires understanding odds and their logarithmic transformation.

#### Odds

$$Odds = \frac{P(Y=1)}{P(Y=0)} = \frac{p}{1-p}$$

If P(churn) = 0.75, odds = 0.75/0.25 = 3, which reads as "3 to 1 odds of churning." Odds greater than 1 indicate the event is more likely than not, while odds less than 1 indicate it is less likely.

#### Log Odds (Logit)

$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + ...$$

The log odds are linear in the predictors, which is where the "linear" in logistic regression comes from. The full probability → odds → log odds → linear model chain works as follows: start with a probability $p$, convert to odds $p/(1-p)$, take the log to get $\log(p/(1-p))$, and this quantity equals the linear predictor $\beta_0 + \beta_1 x_1 + \ldots$. Inverting the chain — applying the sigmoid to the linear predictor — recovers the probability.

For example, consider the model $\log(odds) = -2 + 0.5 \times age$:

| Age | Log Odds | Odds | Probability |
|-----|----------|------|-------------|
| 0 | -2 | $e^{-2}$ ≈ 0.14 | 12% |
| 4 | 0 | $e^{0}$ = 1 | 50% |
| 20 | 8 | $e^8$ ≈ 2981 | 99.97% |

The coefficient $\beta_1 = 0.5$ is steep: each additional year of age increases log odds by 0.5, which multiplies the odds by $e^{0.5} \approx 1.65$ (a 65% increase). The table makes the non-linearity concrete — a jump from age 0 to age 4 moves probability from 12% to 50%, while the same 4-year jump from age 4 to 8 moves from 50% to 88%. Equal steps in $x$ produce equal steps in log odds but very unequal steps in probability.

As the table shows, log odds change linearly with age, but probabilities do not—they follow the S-shaped sigmoid curve. This is the central insight of the logit transform: it provides a linear space for modeling while keeping predictions bounded as probabilities.

#### Why Log Odds

Log odds provide interpretable coefficients (each β is "change in log-odds per unit"), unbounded range (the linear predictor can take any value while output stays bounded 0-1), and additive effects (effects of multiple variables sum in log-odds space, unlike in probability space). The transformation connects linear models to probability naturally.

### Coefficient Interpretation

The coefficient $\beta_1$ represents the change in log odds for a one-unit increase in $x_1$. The odds ratio $e^{\beta_1}$ gives the multiplicative change in odds. For example, if $\beta_1 = 0.5$, then $e^{0.5} \approx 1.65$, meaning each unit increase in X increases the odds by 65%.

To convert to probability, first calculate the log-odds $z = \beta_0 + \beta_1 x_1 + ...$, then apply the sigmoid: $P(Y=1) = \frac{1}{1 + e^{-z}}$.

### Is It Regression or Classification?

The following table clarifies the dual nature of logistic regression.

| Aspect | Answer |
|--------|--------|
| Name | "Regression" (historical reasons) |
| What it models | Probability (continuous 0-1) |
| What we use it for | Classification (discrete classes) |
| How | Apply a threshold to the probability |

Logistic regression is a regression model at its core (it predicts continuous probability), but we use it for classification by applying a threshold.

Probabilities give crucial flexibility over hard class predictions: threshold flexibility (adjust without retraining when costs change), ranking and prioritization ("which 100 customers are most likely to churn?"), confidence communication (P=0.95 vs P=0.55 both classify as positive but represent different confidence), and risk quantification (expected value calculations require probabilities). In business, you almost always benefit from probabilities.

### Evaluation Metrics

Before discussing threshold selection, we need precise language for the four types of outcomes a binary classifier produces. These are organized in the **confusion matrix**:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

A **true positive** is a correctly identified positive (fraud correctly flagged). A **false negative** is a positive case the model missed (fraud that slipped through). A **false positive** is a negative case incorrectly labeled positive (a legitimate transaction flagged). A **true negative** is a correctly identified negative.

From these four counts, we compute three metrics that matter more than raw accuracy when classes are imbalanced or costs are asymmetric:

**Precision** answers "of all the cases I flagged as positive, what fraction actually were?"

$$\text{Precision} = \frac{TP}{TP + FP}$$

High precision means few false alarms. This matters when false positives are costly — flagging innocent customers, or blocking legitimate transactions.

**Recall** (also called sensitivity or true positive rate) answers "of all the actual positives, what fraction did I catch?"

$$\text{Recall} = \frac{TP}{TP + FN}$$

High recall means few missed cases. This matters when false negatives are costly — missing fraud, failing to detect a disease, letting a defaulting borrower through.

**F1 score** is the harmonic mean of precision and recall:

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

F1 ranges from 0 to 1 and gives a single number that balances both concerns. It is more informative than accuracy when classes are imbalanced because accuracy is dominated by the majority class — a model that predicts "not fraud" for every transaction achieves 99.9% accuracy on a dataset where only 0.1% of transactions are fraudulent, but has F1 = 0 for the fraud class.

The choice of metric is a business decision. Use precision when false positives are expensive. Use recall when false negatives are expensive. Use F1 when both matter and you want a balanced summary. Use AUC when you want to evaluate performance across all thresholds simultaneously.

### Decision Thresholds

The default threshold of 0.5 is often suboptimal. The threshold is a business decision, not just a statistical one. Different contexts have different costs for the two types of errors a classifier can make:

- A **false positive** (predicting positive when the true label is negative) causes one kind of harm — investigating a legitimate transaction, denying credit to a good borrower, or sending an unnecessary follow-up call.
- A **false negative** (predicting negative when the true label is positive) causes another — missing actual fraud, approving a defaulting borrower, or failing to catch a disease.

The threshold converts the model's probability output into a binary decision. Changing it does not retrain the model — it changes only which probabilities get labeled as positive. Consider fraud detection, where the cost of missing fraud (false negative) is $10,000 while the cost of investigating non-fraud (false positive) is only $100. With such asymmetric costs, lowering the threshold catches more fraud at the expense of more false alarms.

#### Cost-Based Threshold Formula

The cost-based threshold is $t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$. For fraud costing $10,000 (FN) and investigation costing $100 (FP), the threshold is approximately 100/(100+10000) ≈ 0.01, meaning you would predict fraud for anyone above 1% probability. This formula gives the cost-optimal threshold for a *calibrated posterior probability* $\hat{P}(\text{positive} \mid x)$. Class prevalence is already absorbed into a calibrated posterior, so no extra base-rate adjustment is needed — what matters is whether the model's predicted probabilities reflect true frequencies. Verify calibration first (reliability plot, Brier score), and apply Platt scaling or isotonic regression if they don't. Alternative: Youden's J statistic (maximize TPR-FPR) when costs are unknown.

#### Why This Formula Works

The formula comes from minimizing expected cost at the decision threshold for a single instance with predicted probability $t$. At the threshold, the expected cost of a false positive equals the expected cost of a false negative:

$$P(\text{actually positive}) \times C_{FN} = P(\text{actually negative}) \times C_{FP}$$

Solving for the probability threshold where you're indifferent between predicting positive or negative gives the formula. When $C_{FN}$ is much larger than $C_{FP}$ (missing fraud is expensive), the threshold drops close to zero—you flag almost anything suspicious. When costs are equal, the threshold is 0.5 (the default).

!!! example "Numerical Example: Cost-Based Threshold Selection"

    ```python
    # Fraud detection costs
    cost_fp = 50   # Investigation cost
    cost_fn = 500  # Missed fraud cost

    optimal_threshold = cost_fp / (cost_fp + cost_fn)
    print(f"Optimal threshold: {optimal_threshold:.3f}")

    # Effect on predictions
    probabilities = [0.05, 0.10, 0.20, 0.40, 0.60]
    for p in probabilities:
        default = "Flag" if p >= 0.5 else "Clear"
        optimal = "Flag" if p >= optimal_threshold else "Clear"
        print(f"P={p:.2f}: Default={default}, Optimal={optimal}")
    ```

    **Output:**

    ```
    Optimal threshold: 0.091
    P=0.05: Default=Clear, Optimal=Clear
    P=0.10: Default=Clear, Optimal=Flag
    P=0.20: Default=Clear, Optimal=Flag
    P=0.40: Default=Clear, Optimal=Flag
    P=0.60: Default=Flag, Optimal=Flag
    ```

    **Interpretation:** When fraud costs 10x more than investigation, we flag any transaction with P(fraud) > 9.1%. A transaction with 10% fraud probability gets flagged—it's worth investigating because the expected loss from missing fraud ($500 × 0.1 = $50) equals the investigation cost ($50).

    *Source: `computations/module3_examples.py` — `demo_cost_based_threshold()`*


The following table summarizes how adjusting the threshold affects model behavior.

| Lower Threshold | Higher Threshold |
|-----------------|------------------|
| More positive predictions | Fewer positive predictions |
| Higher recall | Higher precision |
| Lower precision | Lower recall |
| Fewer false negatives | Fewer false positives |

### ROC Curves and AUC

For each possible threshold, calculate the True Positive Rate $TPR = \frac{TP}{TP + FN}$ and the False Positive Rate $FPR = \frac{FP}{FP + TN}$, then plot the resulting point on the curve.

An AUC of 0.5 corresponds to random guessing, while 1.0 indicates perfect separation. An AUC of 0.8 means there is an 80% chance that a randomly chosen positive ranks higher than a randomly chosen negative. Note that AUC is not the same as accuracy—AUC measures ranking ability across all thresholds.

#### Ranking Ability

Ranking ability means correctly ordering examples by likelihood—higher-risk items get higher scores—even if actual probability values are wrong. This matters for resource allocation ("call top 100 highest-risk customers"), campaign targeting (top decile by response rate), and prioritization (fraud investigators review by score). A model with AUC=0.9 and poor calibration is often more useful than AUC=0.6 with perfect calibration—you can recalibrate using Platt scaling or isotonic regression; you can't easily fix ranking ability.

#### The Threshold Dial Intuition

Imagine a dial you can turn from 0 to 1. As you turn it up (higher threshold), you become more selective—fewer positive predictions, but the ones you make are more confident. As you turn it down (lower threshold), you cast a wider net—catching more true positives but also more false alarms. The ROC curve shows you every position of this dial simultaneously. A good model gives you attractive options along the curve; a poor model forces you to choose between bad options (high FPR or low TPR).

!!! example "Numerical Example: Building an ROC Curve Step by Step"

    ```python
    import numpy as np

    # Small dataset: 4 positive, 6 negative
    true_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    pred_proba  = np.array([0.95, 0.85, 0.70, 0.45, 0.60, 0.40, 0.35, 0.20, 0.10, 0.05])

    thresholds = [1.0, 0.70, 0.45, 0.0]
    for thresh in thresholds:
        pred = (pred_proba >= thresh).astype(int)
        tp = np.sum((pred == 1) & (true_labels == 1))
        fp = np.sum((pred == 1) & (true_labels == 0))
        tpr = tp / 4  # 4 actual positives
        fpr = fp / 6  # 6 actual negatives
        print(f"Threshold {thresh:.2f}: TPR={tpr:.2f}, FPR={fpr:.2f}")
    ```

    **Output:**

    ```
    Threshold 1.00: TPR=0.00, FPR=0.00
    Threshold 0.70: TPR=0.75, FPR=0.00
    Threshold 0.45: TPR=1.00, FPR=0.17
    Threshold 0.00: TPR=1.00, FPR=1.00
    ```

    **Interpretation:** The ROC curve plots these (FPR, TPR) points as threshold varies. At threshold 0.70, we achieve TPR=75% with zero false positives—an excellent operating point. Lowering to 0.45 catches all positives but introduces one false positive (FPR=17%). The AUC measures the area under this curve; higher is better.

    *Source: `computations/module3_examples.py` — `demo_roc_curve_construction()`*


There are several approaches to choosing an optimal threshold. Youden's J statistic maximizes (TPR - FPR). A cost-based approach minimizes expected cost given specific FP/FN costs. For imbalanced data, the precision-recall trade-off can be evaluated using a PR curve.

### Case Study: Credit Scoring

Credit scoring is one of the most consequential classification problems in industry. Lenders use logistic regression models to predict whether a loan applicant will default, and these predictions determine who gets credit and at what interest rate.

#### Problem Setup

A bank wants to predict loan default (binary: default vs. no default) using applicant features such as income, debt-to-income ratio, credit history length, number of open accounts, and recent delinquencies. The model outputs a probability of default, which feeds into the approval decision and pricing.

#### Why Logistic Regression Fits

Logistic regression is the standard choice for credit scoring because of regulatory requirements. The Equal Credit Opportunity Act (ECOA) and similar regulations require lenders to explain why an applicant was denied. Logistic regression coefficients translate directly into "adverse action reasons"—the specific factors that most contributed to denial. A coefficient of -0.8 on "years of credit history" means shorter credit history increases default risk, and the magnitude indicates how much.

#### The Business Decision

The model outputs P(default). The lender must choose a threshold:

| Decision | Label | Consequence |
|----------|-------|-------------|
| Approve a good borrower (no default predicted, no default) | TN | Earn interest income |
| Deny a good borrower (default predicted, no actual default) | FP | Lost revenue + potential regulatory scrutiny |
| Deny a risky borrower (default predicted, actual default) | TP | Avoided loss |
| Approve a risky borrower (no default predicted, actual default) | FN | Loan loss (potentially $10K-$100K+) |

Because loan losses far exceed lost revenue from denied applicants, lenders typically lower the threshold — accepting more false positives (good borrowers incorrectly denied) to reduce false negatives (bad borrowers incorrectly approved). In the confusion matrix language: this trades lower precision of the default-prediction class for higher recall of actual defaults. The exact threshold depends on the lender's risk appetite and the loan product.

#### Lessons for Classification

This case study illustrates several key themes from this section: (i) probability outputs matter more than hard classifications because they enable flexible threshold setting and risk-based pricing, (ii) coefficient interpretability is a feature, not a limitation—it enables regulatory compliance and business understanding, (iii) the cost asymmetry between false positives and false negatives drives threshold selection, and (iv) AUC measures the model's overall ability to rank applicants by risk, which is the core task.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# The parameter C controls regularization strength: C = 1/λ, so higher C means
# less regularization. The default C=1.0 provides moderate regularization.
log_reg = LogisticRegression(penalty='l2', C=1.0, random_state=42)
log_reg.fit(X_train, y_train)

# Get probabilities
y_proba = log_reg.predict_proba(X_test)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Optimal threshold (Youden's J)
optimal_idx = (tpr - fpr).argmax()
optimal_threshold = thresholds[optimal_idx]

# Interpret coefficients as odds ratios
for feature, coef in zip(feature_names, log_reg.coef_[0]):
    odds_ratio = np.exp(coef)
    print(f"{feature}: odds ratio = {odds_ratio:.3f}")
```

### Precision-Recall Curves

For imbalanced datasets, the ROC curve can be overly optimistic because True Negative Rate (specificity) is inflated by the large number of negatives. Precision-Recall (PR) curves plot precision (y-axis) against recall (x-axis) and focus entirely on the positive class.

PR curves are most useful when (i) the positive class is rare, such as in fraud detection or disease screening, (ii) you care more about finding positives than correctly classifying negatives, and (iii) you want to understand the precision cost of increasing recall.

Two key differences distinguish PR curves from ROC curves. A "no-skill" classifier traces a horizontal line at y = prevalence rather than a diagonal, and the area under the PR curve (Average Precision) is more informative than AUC-ROC for imbalanced problems.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(recall, precision, label=f"AP = {ap:.3f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend()
plt.show()
```

### Handling Class Imbalance

Class imbalance — where one class greatly outnumbers the other — is the rule, not the exception in business classification problems. Fraud might be 0.1% of transactions. Disease might be 2% of a screened population. Churn might be 5% of customers. Standard training treats every misclassification equally, which means the algorithm implicitly optimizes for the majority class.

The core strategies for addressing imbalance are:

**Adjust the decision threshold.** The simplest and often most effective approach is adjusting the threshold as described above. If the model predicts probabilities well, lowering the threshold increases recall at the cost of precision. No retraining needed.

**Class weighting.** During training, increase the penalty for misclassifying the minority class. In scikit-learn, pass `class_weight='balanced'` to automatically set weights inversely proportional to class frequency. This changes what the model learns, not just how you interpret its output.

```python
from sklearn.linear_model import LogisticRegression

# balanced weights: class i gets weight n / (n_classes * count_i)
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
```

**Resampling.** Oversampling the minority class (e.g., SMOTE — Synthetic Minority Oversampling Technique, which creates synthetic minority examples by interpolating between existing ones) or undersampling the majority class changes the training set distribution. SMOTE is popular but adds complexity; always evaluate whether it actually improves the metric you care about via cross-validation.

The right strategy depends on the problem. Threshold adjustment is a good first step because it is cheap, reversible, and separates the modeling decision from the business decision. Class weighting is useful when you want the model's learned probabilities to reflect the resampled distribution. SMOTE is most useful when the minority class is so rare that the model has very few examples to learn from.

Whatever approach you use, report precision, recall, and F1 — not just accuracy. With 1% positive rate, a model that predicts all-negative achieves 99% accuracy and is completely useless.

### Common Misconceptions

Several common misunderstandings about logistic regression can lead practitioners astray.

| Misconception | Reality |
|--------------|---------|
| "Logistic regression outputs are well-calibrated probabilities" | Outputs may need calibration (Platt scaling, isotonic regression) for reliable probability estimates. |
| "Higher AUC always means better model" | AUC ignores calibration and threshold choice. A model with lower AUC but better calibration might be preferable. |
| "Logistic regression requires linear relationships" | It requires linearity in LOG ODDS, not probability. Add polynomial terms for non-linear relationships. |
| "Logistic regression can't handle multiple classes" | Multinomial logistic regression extends to multiple classes (one-vs-rest or softmax). |

---

## 3.2 Decision Trees (CART)

Decision trees partition the feature space into regions using a sequence of if-then rules, producing models that are intuitive and easy to explain.

### Three Components: Decision Trees

Decision trees fit the same three-component framework used for regression in Module 2.

| Component | Decision Trees |
|-----------|----------------|
| **Decision Model** | Tree of if-then rules — follow branches based on feature thresholds |
| **Quality Measure** | Gini impurity or entropy — measures class mixture in nodes |
| **Update Method** | Greedy recursive splitting — find best split at each node |

The key difference from logistic regression is that trees do not use gradient descent. Instead, they rely on a greedy algorithm that builds one split at a time.

### Decision Tree Intuition

Imagine you're a loan officer. You first ask whether income exceeds $50,000. If yes, you check the debt-to-income ratio; if no, you look at employment history. Decision trees formalize this intuitive process by automatically learning which questions to ask, in what order, and what thresholds to use.

The tree picks the feature and threshold that best separates classes (maximally reduces impurity). This is a greedy algorithm—locally best splits without looking ahead. The first feature is often important but not always "most important": a feature might matter most after controlling for another, or correlated features might be interchanged. Feature importance scores (aggregating across all nodes) are more reliable than just the root split.

### Splitting Criteria

Decision trees evaluate potential splits using a measure of class mixture in the resulting nodes.

#### Gini Impurity

Gini impurity is the scikit-learn default splitting criterion:

$$Gini = 1 - \sum_{i=1}^{C} p_i^2$$

Where $p_i$ is the proportion of class $i$ in the node. A Gini value of 0 indicates a pure node where all samples belong to the same class, while a Gini value of 0.5 represents maximum impurity for binary classification (a 50-50 split). For C classes, the maximum is 1 − 1/C.

#### Gini as Certainty of a Random Guess

If you randomly pick an example from a node and randomly assign it a class based on the node's distribution, Gini measures how often you'd be wrong. For a pure node (100% class A), you'd always guess A and always be right—Gini = 0. For a 50-50 node, you'd be wrong half the time on average—Gini = 0.5 (maximum uncertainty). The tree seeks splits that create nodes where a random guess would more often be correct.

As an example of hand-calculating Gini, consider a node with 60% class A and 40% class B:

$$Gini = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 1 - 0.52 = 0.48$$

This is close to maximum impurity (0.5), indicating the node is nearly evenly split.

!!! example "Numerical Example: Evaluating a Split with Gini"

    ```python
    def gini(class_proportions):
        return 1 - sum(p**2 for p in class_proportions)

    # Parent node: 100 samples, 50-50 split
    parent_gini = gini([0.5, 0.5])
    print(f"Parent Gini: {parent_gini:.4f}")

    # After split: Left (60 samples, 80-20), Right (40 samples, 10-90)
    left_gini = gini([0.8, 0.2])
    right_gini = gini([0.1, 0.9])
    weighted_gini = (60 * left_gini + 40 * right_gini) / 100
    info_gain = parent_gini - weighted_gini

    print(f"Left child Gini: {left_gini:.4f}")
    print(f"Right child Gini: {right_gini:.4f}")
    print(f"Weighted child Gini: {weighted_gini:.4f}")
    print(f"Information gain: {info_gain:.4f}")
    ```

    **Output:**

    ```
    Parent Gini: 0.5000
    Left child Gini: 0.3200
    Right child Gini: 0.1800
    Weighted child Gini: 0.2640
    Information gain: 0.2360
    ```

    **Interpretation:** The parent node has maximum impurity (0.5). After the split, both children are more "pure"—the left child is 80% one class (Gini=0.32), the right is 90% the other (Gini=0.18). The weighted average (0.264) is much lower than the parent (0.5), yielding high information gain (0.236). The tree picks the split that maximizes this gain.

    *Source: `computations/module3_examples.py` — `demo_gini_impurity_calculation()`*


#### Entropy

An alternative splitting criterion is entropy:

$$Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

Entropy usually produces similar results to Gini impurity, and Gini is slightly faster because it avoids logarithm computations. In practice, Gini vs. entropy rarely matters. Entropy penalizes near-equal splits slightly more; with many classes, Gini can favor isolating one class while entropy prefers balanced information gain. Default to Gini (slightly faster); if hyperparameter tuning, include criterion and let cross-validation decide.

### The scikit-learn API Pattern

This pattern is consistent across nearly every scikit-learn model:

```python
# 1. Instantiate
model = DecisionTreeClassifier(max_depth=5, random_state=42)

# 2. Fit
model.fit(X_train, y_train)

# 3. Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Decision Boundaries

Trees create rectangular decision regions because each split creates a horizontal or vertical line in the feature space. Deep trees create many small rectangles, producing a very different boundary from logistic regression's smooth curve.

### Demonstrating Overfitting

A deep tree with no depth limit achieves 100% training accuracy but only 75% test accuracy, producing hundreds of nodes. A shallow tree (depth=3) achieves 85% training accuracy and 82% test accuracy with roughly 15 nodes. The takeaway is that deep trees memorize training data including noise, and 100% training accuracy almost certainly means overfitting.

!!! example "Numerical Example: Decision Tree Overfitting"

    ```python
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    # Data with 10% label noise
    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=5,
        flip_y=0.1, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for depth in [3, 5, 10, None]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        print(f"Depth {str(depth):>4}: Train={tree.score(X_train, y_train):.1%}, "
              f"Test={tree.score(X_test, y_test):.1%}, Leaves={tree.get_n_leaves()}")
    ```

    **Output:**

    ```
    Depth    3: Train=81.9%, Test=71.1%, Leaves=7
    Depth    5: Train=92.4%, Test=77.8%, Leaves=16
    Depth   10: Train=100.0%, Test=80.0%, Leaves=29
    Depth None: Train=100.0%, Test=80.0%, Leaves=29
    ```

    **Interpretation:** The unlimited tree achieves 100% training accuracy—but the data has 10% label noise, so perfect training accuracy means it memorized the noise! The train-test gap (20%) signals overfitting. Shallow trees (depth 3-5) have lower training accuracy but smaller gaps, indicating better generalization.

    *Source: `computations/module3_examples.py` — `demo_tree_overfitting()`*


100% training accuracy is occasionally okay: perfectly separable data (predicting even/odd from last digit), very small clean datasets, or memorization tasks. Verify by checking test accuracy (also very high?), the train-test gap (small vs large?), complexity (10 leaves for 10,000 samples = simple rules; 5,000 leaves = memorized), and cross-validation consistency. The heuristic remains useful: 100% training accuracy should trigger suspicion.

### Pruning Strategies

Pruning controls tree complexity to prevent overfitting, and there are two main approaches.

#### Pre-Pruning (Early Stopping)

Pre-pruning constrains the tree during construction using parameters such as `max_depth` (maximum tree depth), `min_samples_split` (minimum samples required to split a node), and `min_samples_leaf` (minimum samples required in a leaf).

#### Post-Pruning

Post-pruning grows the full tree first, then prunes it back. The `ccp_alpha` parameter controls the extent of pruning, with higher alpha values producing more aggressive pruning.

As a general recommendation, start with pre-pruning. Set `max_depth=5` as a starting point and use cross-validation to optimize.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score

tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
tree.fit(X_train, y_train)

# Check for overfitting
train_acc = tree.score(X_train, y_train)
test_acc = tree.score(X_test, y_test)
print(f"Train: {train_acc:.3f}, Test: {test_acc:.3f}")

# Cross-validation for depth selection
for depth in range(1, 15):
    tree_cv = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(tree_cv, X_train, y_train, cv=5)
    print(f"Depth {depth}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Feature Importance

Decision trees provide a natural measure of feature importance based on how much each feature reduces impurity across all nodes where it is used:

$$Importance = \sum_{nodes} (impurity\ reduction \times samples)$$

There are several caveats to keep in mind: (i) importance is relative and sums to 1, (ii) correlated features split importance between them, and (iii) importance does not indicate the direction of effect or causation.

To understand importance with correlated features: use domain knowledge (which is more causal?), remove one and retrain (does importance transfer?), or use permutation importance (shuffles independently). For prediction, keeping both adds complexity without benefit. For interpretation, report both but note correlation. Consider reporting "this cluster of correlated features is important" rather than attributing to one.

#### The Weighted Vote Intuition

Think of feature importance as a weighted vote. Every time a feature is used to split a node, it gets "votes" equal to (impurity reduction × number of samples affected). A feature used at the root affects all samples—many votes. A feature used deep in the tree affects few samples—fewer votes. The final importance is each feature's total votes divided by all votes. This explains why the root feature often has high importance even if it's not the "most important" conceptually—it affects the most samples.

!!! example "Numerical Example: Feature Importance"

    ```python
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    np.random.seed(42)
    n = 500
    x_strong = np.random.randn(n)  # Strong predictor (coef 2.0)
    x_medium = np.random.randn(n)  # Medium predictor (coef 1.0)
    x_weak = np.random.randn(n)    # Weak predictor (coef 0.3)
    x_noise = np.random.randn(n)   # Pure noise

    y = (2.0*x_strong + 1.0*x_medium + 0.3*x_weak + np.random.randn(n)*0.5 > 0).astype(int)
    X = np.column_stack([x_strong, x_medium, x_weak, x_noise])

    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X, y)

    names = ['strong', 'medium', 'weak', 'noise']
    for name, imp in zip(names, tree.feature_importances_):
        print(f"{name:>8}: {imp:.3f}")
    ```

    **Output:**

    ```
      strong: 0.661
      medium: 0.280
        weak: 0.046
       noise: 0.013
    ```

    **Interpretation:** The tree correctly identifies the strong predictor as most important (66%), followed by medium (28%). The weak predictor and noise have minimal importance. Note: importance is relative (sums to 1) and doesn't indicate causation—just predictive value in this tree.

    *Source: `computations/module3_examples.py` — `demo_feature_importance()`*


### Why Decision Trees Are Popular

Decision trees are popular for several reasons. They are explainable, allowing you to show decision rules directly to stakeholders. They are scale-invariant — feature magnitudes do not affect splits, so standardization is unnecessary. They capture non-linear relationships automatically. Tree diagrams are visual and intuitive for non-technical audiences.

One important caveat about input handling: as of scikit-learn 1.3, `DecisionTreeClassifier` and `RandomForestClassifier` accept `NaN` in feature columns and learn a default direction at each split, so explicit imputation is no longer required. Raw categorical variables, however, still must be encoded — scikit-learn's tree implementations expect numeric features, so use ordinal or one-hot encoding before fitting. XGBoost and LightGBM (covered in Module 4) handle missing values via similar default-direction logic and additionally accept categorical features natively when configured to do so, which simplifies messy real-world workflows.

### Common Misconceptions

Several common misunderstandings about decision trees deserve clarification.

| Misconception | Reality |
|--------------|---------|
| "Deeper trees are always better" | Deeper trees overfit. Find the sweet spot via cross-validation. |
| "Decision trees require feature scaling" | Trees are scale-invariant. One of their advantages. |
| "sklearn trees can't handle missing values" | They can, since scikit-learn 1.3 — `DecisionTreeClassifier` learns a default split direction for `NaN`. Categorical features still need encoding. |
| "Feature importance = causal importance" | Importance only shows predictive power, not causation. |
| "Trees can't capture interactions" | Trees naturally capture interactions through hierarchical structure. |

---

## 3.3 Hyperparameter Optimization

Model performance depends not only on the algorithm chosen but also on the settings that govern how that algorithm learns from data.

### Parameters vs Hyperparameters

Understanding the distinction between parameters and hyperparameters is fundamental to model tuning.

| Parameters | Hyperparameters |
|------------|-----------------|
| Learned during training | Set before training |
| Model learns via .fit() | You choose before .fit() |
| Example: Coefficients | Example: Regularization strength |
| Example: Split points | Example: Max tree depth |

Hyperparameters control how the model learns, and finding the right values is a central part of model development. You can discover available hyperparameters through official documentation (search "sklearn DecisionTreeClassifier"), in-code exploration (`model.get_params()`, `help(DecisionTreeClassifier)`), or IDE autocomplete. Not all hyperparameters matter equally—most algorithms have 3-5 "important" ones: for decision trees, focus on `max_depth`, `min_samples_split`, `min_samples_leaf`; for Random Forests add `n_estimators`, `max_features`; for XGBoost: `learning_rate`, `max_depth`, `n_estimators`, `subsample`.

### Grid Search

Grid search tries every combination of hyperparameters from a predefined set of values.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
}
# Total: 4 × 3 = 12 combinations

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1'
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

Grid search is exhaustive and reproducible; however, it suffers from exponential growth in the number of combinations and wastes time evaluating bad regions of the hyperparameter space.

### Random Search

Random search samples combinations from specified distributions rather than exhaustively evaluating a grid.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distributions = {
    'max_depth': randint(2, 20),
    'min_samples_split': randint(2, 50),
}

random_search = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='f1',
    random_state=42
)
random_search.fit(X_train, y_train)
```

### Why Random Often Beats Grid

The key insight from Bergstra and Bengio (2012) is that not all hyperparameters are equally important. Grid search wastes trials on unimportant parameters, while random search explores more values of the parameters that matter. In practice, random search often beats grid search with the same computational budget.

#### The Geometric Intuition

Imagine a 2D hyperparameter space where only one dimension matters (common in practice). Grid search with 9 trials might try 3 values per dimension, giving you only 3 unique values of the important parameter. Random search with 9 trials gives you 9 unique values of the important parameter. When you don't know which parameters matter most (and you usually don't), random search automatically allocates more trials to exploring variation in every dimension. Grid search wastes trials exploring combinations of unimportant parameters.

#### Standard Ranges for Common Hyperparameters

Typical ranges include `max_depth`: 2-20 for trees; `n_estimators`: 50-500 for forests/boosting; `learning_rate`: 0.001-0.3 for boosting; `min_samples_split`: 2-50; and `C` (regularization): 0.001-100 (log scale). If the best value is at the edge of your range, extend that direction. Start with wide, log-spaced ranges, do a coarse search (10 values), then refine in the promising region.

### Bayesian Optimization (optuna)

Bayesian optimization uses past trial results to guide future trials, focusing the search on promising regions of the hyperparameter space.

```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
    }
    model = DecisionTreeClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(f"Best params: {study.best_params}")
```

This approach is more efficient than random search because it learns from previous trials, concentrating effort on configurations that are more likely to perform well.

### Cross-Validation During Tuning

Cross-validation (CV) is not just a way to estimate performance — during hyperparameter tuning it is the mechanism that prevents you from overfitting to your evaluation data. Without it, the tuning process itself becomes a source of overfitting.

Here is why this matters. Suppose you have a training set and a single fixed validation set. You try 100 hyperparameter combinations, measure validation accuracy for each, and pick the best. Even if the model never saw the validation set during training, you have effectively used that validation set 100 times to make decisions. With enough trials, you will find a combination that happens to perform well on that particular validation set — not because it is genuinely better, but because of chance. This is called **overfitting to the validation set**.

Cross-validation breaks this by using multiple validation folds. Each hyperparameter combination is evaluated on $K$ different held-out subsets. The average score across folds is a much more stable estimate of true generalization performance. Any individual fold can be misleading by chance, but the average across 5 or 10 folds is much harder to game. The more trials you run, the more important CV becomes — this is why Bayesian optimization with 100 trials still uses 5-fold CV inside each trial, giving 500 model fits per tuning run.

### Never Use Test Set for Tuning

The correct workflow begins with splitting data into train and test sets. Next, use cross-validation on the training set for tuning and select the best hyperparameters via the CV score. Then retrain on the full training set and evaluate once on the test set. If you tune on the test set, your performance estimate is no longer unbiased.

After tuning, retrain on all training data with the best hyperparameters. Cross-validation models were trained on only $(K-1)/K$ of your data, so retraining on 100% gives the model more examples. `GridSearchCV` does this automatically — `grid_search.best_estimator_` is already retrained on the full training set.

The test set is reserved for one purpose: a final, unbiased estimate of deployed performance. Treating it as a validation set during tuning turns it into another training signal, and your reported test accuracy will be optimistic — sometimes by several percentage points.

### Common Misconceptions

Several common misunderstandings about hyperparameter tuning can waste effort or produce misleading results.

| Misconception | Reality |
|--------------|---------|
| "More hyperparameter tuning always helps" | Diminishing returns. 50-100 trials often enough. Risk overfitting to validation data. |
| "Grid search is more thorough" | Grid is exhaustive only for values you specify. Random can find values between grid points. |
| "Best hyperparameters are universal" | Optimal hyperparameters depend on your specific dataset. |
| "Use test set to choose hyperparameters" | Never! Use cross-validation on training data. |

---

## Deep Dives

For supplementary material that extends this module:

- **[Deep Dive: Handling Imbalanced Data](../appendices/imbalanced-data.md)** — resampling, class weighting, threshold tuning, and metrics that remain meaningful under class skew.

---

## Reflection Questions

1. A model predicts P(churn) = 0.6 for a customer. What does this actually mean? How confident should we be?

2. Why might you choose a threshold other than 0.5? Give scenarios for very low and very high thresholds.

3. A logistic regression coefficient for 'number of support tickets' is 0.3. How would you explain this to a stakeholder?

4. You build a decision tree with 100% training accuracy. Is this good or bad? What would you do next?

5. In fraud detection with 0.1% fraud rate, a model achieves 99.9% accuracy. What's wrong with celebrating this?

6. When would you prefer high precision over high recall? Give a business example.

7. Why might random search find better hyperparameters than grid search with the same budget?

8. You run 200 hyperparameter trials using a single fixed validation set and achieve 92% validation accuracy. You report this as your expected test performance. What went wrong, and how would cross-validation have helped?

---

## Practice Problems

1. Calculate odds and log-odds for P = 0.8

2. Given coefficients β₀ = -2, β₁ = 0.5, β₂ = -0.3, calculate P(Y=1) when x₁ = 4, x₂ = 2

3. A classifier produces this confusion matrix (positive = spam, negative = legitimate):
   - TP = 80, FP = 20, FN = 10, TN = 890
   - Calculate: accuracy, precision, recall, and F1 score.

4. Draw what a decision tree boundary would look like for 2D data with 2 splits

5. Given a 95% legitimate / 5% fraud dataset: if we predict all legitimate, what's accuracy? Precision for fraud? Recall for fraud?

6. Choose between precision and recall priority for: (a) spam filter, (b) cancer screening, (c) loan approval

---

## Chapter Summary

This module introduced the core tools for classification. Logistic regression converts a linear predictor to a probability via the sigmoid function through the probability → odds → log odds chain; coefficients can be exponentiated into odds ratios for business-friendly interpretation, and the decision threshold is a business choice driven by the relative costs of false positives and false negatives. The confusion matrix — and the precision, recall, and F1 metrics derived from it — provides the vocabulary for making threshold decisions precisely, and is essential whenever classes are imbalanced because raw accuracy is dominated by the majority class. Decision trees offer intuitive if-then rule structures but overfit easily; pruning and cross-validated depth selection are the primary remedies. Modern scikit-learn decision trees (version 1.3 and later) accept missing values (`NaN`) directly and learn a default split direction, so explicit imputation is no longer required, though categorical features still need numeric encoding. Finally, hyperparameter tuning should always be conducted through cross-validation on the training set: CV prevents overfitting to the validation set by averaging performance across multiple folds, and the test set is reserved for a single final evaluation.

---

## What's Next

In Module 4, we tackle **Ensemble Methods**, covering Random Forests (ensembles of decision trees), Gradient Boosting (XGBoost, LightGBM), and the broader question of why combining weak learners creates strong models. Understanding decision trees is essential—Random Forests take everything we learned about trees and combine many of them for better performance.
