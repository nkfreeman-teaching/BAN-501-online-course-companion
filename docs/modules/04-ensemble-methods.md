# Module 4: Ensemble Methods

## Introduction

In Module 3, we learned about decision trees—intuitive classifiers that are easy to interpret but prone to overfitting. Deep trees memorize training data; shallow trees underfit.

This module answers a natural question: **What if we could get the benefits of deep trees without the overfitting?** The answer is ensemble methods. Instead of training one model, we train many models and combine their predictions. This simple idea—the wisdom of crowds—turns out to be one of the most powerful techniques in machine learning.

By the end of this module, you'll understand two major ensemble paradigms: **bagging** (where Random Forests come from) and **boosting** (where XGBoost comes from). These methods dominate tabular data competitions and are workhorses in industry.

---

## Learning Objectives

By the end of this module, you should be able to:

1. **Explain** the intuition behind ensemble methods and why combining models outperforms individuals
2. **Implement** bagging (Random Forests) and boosting (XGBoost)
3. **Interpret** feature importance for business stakeholders
4. **Select** appropriate ensemble strategies based on problem characteristics

---

## 4.1 Ensemble Learning Concepts

This section introduces the foundational ideas behind ensemble methods: why combining multiple models works and how diversity among models is the critical ingredient.

### The Wisdom of Crowds

#### Galton's Ox Experiment (1907)

At a county fair, 787 people tried to guess the weight of an ox. Individual guesses varied wildly—some way too high, some way too low. The mean of all guesses was 1,197 lbs, within 1 lb of the actual weight of 1,198 lbs (less than 1% error). The median came in at 1,207 lbs. How can a crowd of non-experts outperform individuals?

Errors cancel out when they are uncorrelated. Some people guessed too high, some too low. The errors went in different directions. When you average, errors cancel and the true signal remains. This is exactly the principle behind ensemble machine learning.

Correlation matters as well. Ensembles work best with uncorrelated errors, but help even with partially correlated errors. If individual models have variance σ² and correlation ρ between errors, ensemble variance is ρσ² + (1-ρ)σ²/n. With perfect independence (ρ=0), variance drops as 1/n. With perfect correlation (ρ=1), averaging does not help. In practice, even 50% correlation provides substantial benefit.

As a concrete example, imagine 5 models, each with 70% accuracy on a binary prediction. If each model makes independent errors, the probability all 5 are wrong on the same example is 0.3⁵ = 0.24%. The majority vote is wrong only when 3+ models are wrong, so the ensemble achieves roughly 84% accuracy—significantly better than any individual. But if all models make the *same* mistakes (ρ=1), the ensemble is still just 70% accurate. Diversity is the key ingredient.

!!! example "Numerical Example: Ensemble Variance and Correlation"

    ```python
    import numpy as np

    # Parameters: 10 models, each with variance σ² = 100
    individual_variance = 100
    n_models = 10

    # Ensemble variance formula: Var = ρσ² + (1-ρ)σ²/n
    for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ensemble_var = (
            rho * individual_variance
            + (1 - rho) * individual_variance / n_models
        )
        reduction = (1 - ensemble_var / individual_variance) * 100
        print(f"ρ={rho:.2f}: Var={ensemble_var:.1f}, Reduction={reduction:.0f}%")
    ```

    **Output:**

    ```
    ρ=0.00: Var=10.0, Reduction=90%
    ρ=0.25: Var=32.5, Reduction=68%
    ρ=0.50: Var=55.0, Reduction=45%
    ρ=0.75: Var=77.5, Reduction=22%
    ρ=1.00: Var=100.0, Reduction=0%
    ```

    **Interpretation:** With 10 independent models (ρ=0), variance drops by 90%. Even with moderate correlation (ρ=0.5), you still get 45% reduction. This is why Random Forest's feature sampling matters—it reduces ρ between trees.

    *Source: `computations/module4_examples.py` — `demo_ensemble_variance_correlation()`*


### How Ensembles Improve Predictions

Bagging reduces variance. Single decision trees are high-variance estimators, and small changes in training data can produce very different trees. The mechanism is averaging: each tree overfits to its particular bootstrap sample, but the directions of overfitting differ across trees. When predictions are averaged, the idiosyncratic errors cancel while the shared signal—patterns present across all bootstrap samples—reinforces. Mathematically, $Var(\text{average}) = Var(\text{individual}) / n$ when predictions are uncorrelated, so variance shrinks in proportion to the number of trees.

Boosting reduces bias. Each new model focuses on the errors of previous models, so the ensemble gradually learns patterns that individual weak learners missed. The mechanism is residual fitting: the first weak learner captures the easiest patterns; subsequent learners are trained on the remaining errors, forcing them to find structure that the ensemble has not yet explained. After many rounds, the combined model covers patterns that no single shallow tree could represent, reducing the systematic underfit (bias) that a single weak learner would exhibit.

### Model Diversity is Critical

Ensembles only help if the models are different. If all models make the same mistakes, averaging provides no benefit. Think: if you ask 787 people the same leading question and they all guess the same wrong answer, the median is still wrong.

Ensemble methods create diversity in several ways. Random Forests use random sampling of both data and features, boosting applies sequential focus on different examples, and different algorithms bring different inductive biases (heterogeneous ensembles).

Heterogeneous ensembles take diversity further by combining completely different algorithms, such as a neural network, a decision tree, and logistic regression. Because different algorithms carry different inductive biases, they are unlikely to make the same mistakes. The Netflix Prize winning solution combined 107 different models, illustrating the power of this approach.

### When Ensembles Help vs. Hurt

Ensembles are not universally beneficial. Understanding the conditions under which they add value—and where they impose costs without commensurate gain—is part of choosing the right tool for a problem.

Ensembles help most when (i) individual models have meaningful predictive power but differ in the patterns they capture (diversity), (ii) the dataset is large enough that training multiple models does not create a memory or time bottleneck, and (iii) the prediction task is complex enough that no single model can capture all relevant structure.

Ensembles can hurt or fail to help when the data is highly noisy and the base models are powerful enough to memorize noise—boosting in particular will amplify noisy labels round after round. They also add limited value when the base models are too similar (identical architectures on identical data produce correlated errors that do not cancel). In settings with a strict computational budget, the overhead of training, storing, and serving hundreds of models may not be justified by a 1-2% accuracy gain over a single well-tuned model. Finally, when a stakeholder requires a model whose predictions can be traced to a single decision path, an ensemble is the wrong architecture regardless of its accuracy.

### Common Misconceptions

Several misconceptions about ensemble learning persist in practice.

| Misconception | Reality |
|--------------|---------|
| "More models always means better results" | Diminishing returns kick in quickly. 1 → 10 trees helps a lot; 100 → 1000 helps little. |
| "Ensembles are always better than single models" | For simple problems or when interpretability is paramount, single models may be preferable. |
| "You need sophisticated models in your ensemble" | Ensembles of simple models (shallow trees, stumps) can be remarkably effective. |

---

## 4.2 Bagging Methods

Bagging methods reduce variance by training multiple models on different bootstrap samples of the data and combining their predictions.

### Three Components: Random Forest

Random Forest fits the three-component framework as follows.

| Component | Random Forest |
|-----------|---------------|
| **Decision Model** | Ensemble of decision trees — each tree votes, majority wins |
| **Quality Measure** | Gini/entropy for individual trees; OOB error for ensemble |
| **Update Method** | Independent parallel training — no iteration between trees |

The key insight is that Random Forest does not "update" traditionally. Each tree trains independently on a bootstrap sample. Learning happens through aggregation—the wisdom of crowds.

### Bootstrap Aggregating (Bagging)

The bagging algorithm proceeds in three steps. First, create B bootstrap samples by sampling with replacement; each sample is the same size as the original data, with some observations appearing multiple times, some not at all, and roughly 63.2% unique observations per sample. Next, train a separate model on each sample. Finally, aggregate predictions by averaging for regression or majority vote for classification.

#### Why ~63.2%?

Each bootstrap draw selects one of $n$ observations uniformly at random with replacement. The probability that a specific observation $i$ is not selected on any single draw is $(1 - 1/n)$. Because we make $n$ draws independently, the probability that observation $i$ is never selected is:

$$(1 - \frac{1}{n})^n$$

As $n \to \infty$, this converges to $e^{-1}$. This limit follows directly from the standard definition of $e$: $\lim_{n \to \infty}(1 + 1/n)^n = e$, so $\lim_{n \to \infty}(1 - 1/n)^n = e^{-1} \approx 0.368$.

Therefore, roughly 36.8% of observations are left out of any bootstrap sample, and the expected fraction of unique observations included is:

$$1 - e^{-1} \approx 0.632$$

This convergence is fast: with $n = 100$, the fraction is already 0.634; with $n = 1000$, it is 0.632. The 36.8% left out are called out-of-bag samples and provide free validation without any additional training.

!!! example "Numerical Example: Bootstrap Sampling in Action"

    ```python
    import numpy as np

    np.random.seed(42)
    n_samples = 1000
    n_bootstrap_samples = 100

    unique_fractions = []
    for _ in range(n_bootstrap_samples):
        bootstrap_indices = np.random.choice(
            n_samples,
            size=n_samples,
            replace=True,
        )
        unique_count = len(np.unique(bootstrap_indices))
        unique_fractions.append(unique_count / n_samples)

    print(f"Mean unique fraction: {np.mean(unique_fractions):.3f}")
    print(f"Theoretical (1 - e⁻¹): {1 - np.exp(-1):.3f}")
    ```

    **Output:**

    ```
    Mean unique fraction: 0.632
    Theoretical (1 - e⁻¹): 0.632
    ```

    **Interpretation:** Across 100 bootstrap samples, exactly 63.2% of observations appear on average—matching the theoretical prediction. The remaining 36.8% are "out-of-bag" and can be used for free validation.

    *Source: `computations/module4_examples.py` — `demo_bootstrap_sampling()`*


Sampling with replacement is essential. If you sampled without replacement at the same size as the original dataset, you would recover the original dataset exactly—every bootstrap sample would be identical, every tree would train on the same data, and no diversity would be created. With replacement, some observations appear multiple times (emphasized by that sample), some do not appear at all (~36.8%, providing OOB validation), and different trees see different emphases—creating the diversity that makes averaging beneficial. Bootstrap sampling with replacement approximates drawing fresh samples from the true population.

### Random Forests: Double Randomness

Random Forests extend bagging with **two sources of randomness**: (i) row sampling from bagging, where each tree gets a bootstrap sample, and (ii) feature sampling unique to RF, where each split considers only a random subset of features (by default, $\sqrt{d}$ features for classification, where d is the total number of features).

#### Why Feature Sampling Matters

Imagine one highly predictive feature (credit score for loan default). Without feature sampling, every tree uses it as the root split. All trees become highly correlated.

With feature sampling, each split considers a random subset. Sometimes credit score is not available. The tree finds other splits. This creates diversity.

The tradeoff is that ignoring the best feature sometimes hurts individual trees (higher bias), but trees become more diverse (lower correlation). The ensemble variance formula shows that reducing correlation (ρ) often helps more than the slight increase in individual variance (σ²). Random Forests typically outperform bagged trees precisely because of this tradeoff. The `max_features` hyperparameter controls this—default √d is a good starting point.

### Why Bagging Reduces Overfitting

A single deep tree overfits to specific patterns. Each tree in the forest also overfits, but to different patterns. When we average, idiosyncratic overfitting cancels out, and the true signal remains because all trees agree on it.

The ensemble variance formula formalizes this intuition:

$$Var(ensemble) = \rho\sigma^2 + \frac{(1-\rho)\sigma^2}{n}$$

Here, $\sigma^2$ is the variance of individual tree predictions, $\rho$ is the average correlation between trees (0 = independent, 1 = identical), and $n$ is the number of trees.

Reading this formula, the first term ($\rho\sigma^2$) represents irreducible variance from correlation, and the second term shrinks as you add trees. The key insight is that lower correlation between trees produces a better ensemble, and feature sampling specifically reduces $\rho$.

The following table illustrates the formula in action with 10 trees and σ²=100:

| Correlation (ρ) | Ensemble Variance | Reduction |
|-----------------|-------------------|-----------|
| 0.0 (independent) | 10 | 90% |
| 0.5 (moderate) | 55 | 45% |
| 1.0 (identical) | 100 | 0% |

Even with ρ=0.5, you still get 45% variance reduction. This explains why Random Forests work well in practice—trees do not need to be perfectly independent, just somewhat different. The diminishing returns are also visible: going from ρ=1.0 to ρ=0.5 saves 45 points of variance, but going from ρ=0.5 to ρ=0.0 saves another 45, illustrating why any reduction in correlation helps.

!!! example "Numerical Example: Random Forest vs Single Tree"

    ```python
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    np.random.seed(42)
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=10,
        n_redundant=5, n_classes=2, random_state=42,
    )

    # Run 20 different train/test splits
    tree_scores, rf_scores = [], []
    for trial in range(20):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=trial,
        )
        tree = DecisionTreeClassifier(max_depth=None, random_state=42)
        tree.fit(X_train, y_train)
        tree_scores.append(tree.score(X_test, y_test))

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_scores.append(rf.score(X_test, y_test))

    print(f"Single Tree: Mean={np.mean(tree_scores):.3f}, Std={np.std(tree_scores):.3f}")
    print(f"RF (100):    Mean={np.mean(rf_scores):.3f}, Std={np.std(rf_scores):.3f}")
    ```

    **Output:**

    ```
    Single Tree: Mean=0.809, Std=0.032
    RF (100):    Mean=0.891, Std=0.025
    ```

    **Interpretation:** Across 20 different data splits, Random Forest achieves 8 percentage points higher accuracy AND 22% lower variance. The ensemble is both more accurate and more stable than any single tree.

    *Source: `computations/module4_examples.py` — `demo_rf_vs_single_tree()`*


In practice, 100-500 trees are usually sufficient. Plot OOB error vs. n_estimators—it decreases rapidly then flattens. Unlike boosting, more RF trees never hurt performance; they just stop helping. More trees mean more memory and slower inference, so balance accuracy against cost.

### Feature Importance

Random Forests provide two main approaches for measuring how much each feature contributes to predictions.

#### Mean Decrease in Impurity (MDI)

Mean Decrease in Impurity sums the impurity decreases from splits using each feature, averaged across trees. It is fast to compute but can favor high-cardinality features.

#### Permutation Importance

Permutation importance works by shuffling each feature and measuring the resulting accuracy decrease. It is more reliable than MDI, though slower, and is preferred for stakeholder communication. An important caveat is that importance is not the same as direction of effect: importance tells you which features the model relies on, not how they affect predictions.

Both MDI and permutation importance answer "how much does this feature matter overall?" but neither tells you whether a feature pushes predictions up or down, or whether its effect is linear, threshold-based, or interaction-driven. For those questions, Module 9 covers SHAP (SHapley Additive exPlanations), which decomposes each individual prediction into per-feature contributions with sign. SHAP values apply to any model—including Random Forests and XGBoost—and produce both global summaries (similar to importance rankings) and local explanations (why this specific prediction was made). The feature importance intuition you build here transfers directly to interpreting SHAP outputs.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    oob_score=True,
    random_state=42
)
rf.fit(X_train, y_train)

# OOB score (free validation!)
print(f"OOB Accuracy: {rf.oob_score_:.3f}")

# MDI importance (fast)
importance_mdi = rf.feature_importances_

# Permutation importance (more reliable)
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10)
```

### Out-of-Bag (OOB) Error

Each bootstrap sample leaves out ~36.8% of observations. These "out-of-bag" samples provide free validation. For each observation, you predict using only trees that did not train on it. The resulting OOB error approximates cross-validation error at no additional computational cost.

```python
rf = RandomForestClassifier(oob_score=True)
rf.fit(X_train, y_train)
print(f"OOB Accuracy: {rf.oob_score_}")
```

#### Why OOB Approximates Cross-Validation

For any single observation, about 36.8% of trees never saw it during training. When you predict that observation using only those trees, you get an honest estimate—those trees could not have memorized it. Aggregating these honest predictions across all observations gives you an error estimate very close to what k-fold cross-validation would produce, but without the computational cost of retraining k times.

!!! example "Numerical Example: OOB Error vs Cross-Validation"

    ```python
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_classes=2, random_state=42,
    )

    # OOB scoring
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
    rf.fit(X, y)

    # Cross-validation
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X, y, cv=5,
    )

    print(f"OOB Accuracy:    {rf.oob_score_:.4f}")
    print(f"5-Fold CV Mean:  {np.mean(cv_scores):.4f}")
    print(f"Difference:      {abs(rf.oob_score_ - np.mean(cv_scores)):.4f}")
    ```

    **Output:**

    ```
    OOB Accuracy:    0.9210
    5-Fold CV Mean:  0.9330
    Difference:      0.0120
    ```

    **Interpretation:** OOB and 5-fold CV produce nearly identical estimates (within 1.2 percentage points), but OOB comes free—no extra model training required. Use OOB for quick hyperparameter feedback during tuning.

    *Source: `computations/module4_examples.py` — `demo_oob_vs_cv()`*


### Common Misconceptions

The following misconceptions about Random Forests are worth addressing directly.

| Misconception | Reality |
|--------------|---------|
| "Random Forest can't overfit" | It can. Deep trees with too few estimators still overfit. Tuning on the test set causes overfitting to that set. |
| "More trees is always better" | Diminishing returns. 100-500 usually sufficient. |
| "Random Forest is a black box" | Feature importance and SHAP make it reasonably interpretable. |
| "Feature importance = feature effect" | Importance shows reliance, not direction of effect. |

---

## 4.3 Boosting Methods

Boosting methods build ensembles sequentially, with each new model correcting the errors of its predecessors.

### Three Components: Gradient Boosting

Gradient boosting can be understood through the same three-component framework used for other models.

| Component | Gradient Boosting (XGBoost) |
|-----------|----------------------------|
| **Decision Model** | Sequential ensemble — sum of many shallow trees |
| **Quality Measure** | Any differentiable loss + regularization |
| **Update Method** | Gradient descent in function space — each tree corrects previous errors |

The update method is notable: instead of updating parameters, we add new functions (trees). Each tree predicts the negative gradient (residuals).

### The Boosting Philosophy

The boosting philosophy is to build models sequentially, where each new model focuses on the mistakes of previous ones.

| Bagging | Boosting |
|---------|----------|
| Parallel (independent trees) | Sequential (dependent trees) |
| Reduces variance | Primarily reduces bias; can increase variance if over-iterated |
| Deep trees | Shallow trees typical |

Two visual metaphors help distinguish them: bagging is like a committee of experts who work independently and vote, while boosting is like a relay team where each runner covers for previous weaknesses.

### AdaBoost: Adaptive Boosting

AdaBoost begins by assigning equal weights to all training examples (weight $w_i = 1/n$ for each observation). It then trains a weak learner (often a "stump"—a tree with one split) and evaluates its predictions. Misclassified examples receive a higher weight; correctly classified examples receive a lower weight. Concretely, the weight of a misclassified example is multiplied by $e^{\alpha_t}$, where $\alpha_t = \frac{1}{2}\ln\frac{1 - \epsilon_t}{\epsilon_t}$ and $\epsilon_t$ is the weighted error rate of the current learner. A better learner (lower $\epsilon_t$) produces a larger $\alpha_t$, amplifying the weight shift on its mistakes. The next weak learner is trained on the reweighted dataset, so it is forced to pay attention to the examples that stumped previous learners.

The key insight is that each subsequent learner specializes in hard examples that previous learners got wrong. The final prediction is a weighted vote across all stumps, where each stump's vote is weighted by its own $\alpha_t$—more accurate stumps get more say. Over many rounds, the ensemble builds up a collection of specialists, each contributing where others struggled.

Boosting can, however, obsess over mislabeled or impossible-to-fit examples. Mitigation strategies include (i) setting `subsample` to 0.8 so outliers do not appear every round, (ii) lowering the learning rate to limit per-iteration damage, (iii) applying regularization (`reg_alpha`, `reg_lambda`) to prevent extreme predictions, and (iv) using early stopping before overfitting to noise. Random Forests are more robust because outliers only affect ~63% of trees and no tree specifically focuses on them.

### Gradient Boosting Machines

The core innovation of gradient boosting is fitting each new tree to the residuals (errors). The algorithm starts by making an initial prediction (often the mean), then calculates residuals as $actual - predicted$. Next, it fits a tree to predict those residuals and adds the tree's predictions (scaled by a learning rate) to the current predictions. It then calculates new residuals and repeats the process.

The name "gradient" comes from the connection to MSE loss:

$$\frac{\partial L}{\partial \hat{y}} = -(y - \hat{y}) = -\text{residual}$$

The residual is the negative gradient of the loss. When we fit trees to residuals, we are following the gradient in function space.

#### Gradient in Function Space

Normal gradient descent optimizes parameters (adjust $\theta$). Gradient boosting optimizes functions (add a new tree). For squared error, the negative gradient is simply the residual. Fitting a tree to residuals answers: "what should I add to reduce error?"

The update at each boosting round is:

$$F_t(x) = F_{t-1}(x) + \alpha \cdot h_t(x)$$

where $F_{t-1}(x)$ is the ensemble after the previous round, $h_t(x)$ is the new tree trained on the current residuals, and $\alpha$ is the learning rate. The learning rate works exactly as in gradient descent: taking fractional steps prevents overshooting the minimum. Each tree is one step in function space toward lower loss.

On a simple regression problem (y = 2x + 3 + noise), you can watch boosting learn across rounds:

| Round | Residual Std | MSE |
|-------|--------------|-----|
| Init | 5.93 | 35.1 |
| 1 | 4.69 | 22.0 |
| 2 | 3.83 | 14.6 |
| 3 | 3.22 | 10.4 |
| 5 | 2.46 | 6.1 |

Each tree chips away at the remaining error. The residual standard deviation drops steadily as boosting "discovers" the linear relationship through many small corrections.

!!! example "Numerical Example: Gradient Boosting Step by Step"

    ```python
    from sklearn.tree import DecisionTreeRegressor
    import numpy as np

    np.random.seed(42)
    n_samples = 100
    X = np.random.uniform(low=0, high=10, size=(n_samples, 1))
    y = 2 * X.ravel() + 3 + np.random.normal(loc=0, scale=2, size=n_samples)

    # Manual gradient boosting
    learning_rate = 0.3
    prediction = np.full(n_samples, np.mean(y))  # Start with mean

    print(f"{'Round':>6} {'Residual Std':>14} {'MSE':>10}")
    for round_num in range(6):
        residuals = y - prediction
        mse = np.mean(residuals ** 2)
        print(f"{round_num:>6} {np.std(residuals):>14.2f} {mse:>10.2f}")
        if round_num < 5:
            tree = DecisionTreeRegressor(max_depth=1, random_state=42)
            tree.fit(X, residuals)
            prediction += learning_rate * tree.predict(X)
    ```

    **Output:**

    ```
    Round   Residual Std        MSE
         0           5.93      35.12
         1           4.69      21.99
         2           3.83      14.64
         3           3.22      10.35
         4           2.76       7.62
         5           2.46       6.06
    ```

    **Interpretation:** Each round, a shallow tree predicts the residuals (errors), and we add a fraction of its predictions. MSE drops from 35 to 6 in just 5 rounds as boosting learns the linear pattern y = 2x + 3.

    *Source: `computations/module4_examples.py` — `demo_gradient_boosting_steps()`*


### Key Boosting Hyperparameters

Three hyperparameters have the greatest impact on boosting performance.

| Parameter | Effect |
|-----------|--------|
| `n_estimators` | More → more capacity, but overfit risk |
| `learning_rate` | Smaller → need more trees, often better |
| `max_depth` | Usually 3-8 (much shallower than RF) |

The central tradeoff is that a lower learning rate paired with more trees often gives the best results but takes longer to train.

#### Practical Guidance for Learning Rate

Start with 0.1 as a good default that is fast enough to iterate. Try 0.01-0.05 if you observe overfitting (training accuracy far exceeding test accuracy). Use 0.3 only for quick prototyping or when the data is very large. In all cases, pair the learning rate with early stopping so the algorithm can find the optimal n_estimators on its own.

A lower learning rate makes each tree's contribution smaller, requiring more trees to reach the same capacity. This acts as implicit regularization—the model has more chances to "change its mind" and does not commit too heavily to early patterns.

!!! example "Numerical Example: Learning Rate Effects on Boosting"

    ```python
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_classes=2, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )

    configs = [(0.3, 50), (0.1, 150), (0.03, 500)]
    for lr, n_est in configs:
        gb = GradientBoostingClassifier(
            learning_rate=lr, n_estimators=n_est, max_depth=3, random_state=42,
        )
        gb.fit(X_train, y_train)
        print(f"LR={lr:.2f}, Trees={n_est:>3}: Test={gb.score(X_test, y_test):.4f}")
    ```

    **Output:**

    ```
    LR=0.30, Trees= 50: Test=0.9033
    LR=0.10, Trees=150: Test=0.9067
    LR=0.03, Trees=500: Test=0.9033
    ```

    **Interpretation:** All three configurations achieve similar test accuracy, but through different paths. Lower learning rate + more trees is slower to train but often more stable. The medium configuration (0.1, 150) slightly edges out the others here.

    *Source: `computations/module4_examples.py` — `demo_learning_rate_effects()`*


The two paradigms differ in tree depth. Random Forest uses deep, fully-grown trees (low bias, high variance) and relies on averaging to reduce variance. Boosting uses shallow trees (high bias) and relies on sequential correction to reduce bias.

### XGBoost: The Competition Champion

XGBoost adds several optimizations that make it dominant: (i) L1/L2 regularization penalties on leaf weights, (ii) parallel processing of split evaluation within trees, (iii) learned optimal direction for missing values, and (iv) histogram-based splitting that bins features for speed. It has won more Kaggle competitions than any other algorithm and is widely adopted in finance, insurance, and tech.

That said, Random Forest is sometimes the better choice. RF handles noisy labels more robustly because noise does not compound across trees. RF also works well with defaults when tuning time is limited, parallelizes naturally since trees train independently, and is less prone to overfitting on small datasets. A well-tuned XGBoost beats a well-tuned RF, but default RF often beats default XGBoost. In many real-world scenarios, the difference is 1-2%.

### XGBoost with Early Stopping

You should always use early stopping with boosting. Because boosting adds complexity with each round, setting a generous upper bound on n_estimators and letting a validation set determine when to stop produces better-generalizing models than picking a fixed number of rounds.

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)
# Second split: separate validation set from training
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42,
)

xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
    random_state=42,
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print(f"Best iteration: {xgb_model.best_iteration}")
```

Without early stopping, boosting will overfit as it continues to add trees that memorize noise. With early stopping enabled, training halts automatically when validation performance plateaus, preserving the model at its best generalization point.

#### Why Early Stopping Beats Fixed n_estimators

The optimal number of rounds depends on learning rate, tree depth, data complexity, and sample size—a fixed number cannot adapt. Set a large n_estimators as an upper limit, monitor validation loss, and stop when no improvement occurs for N consecutive rounds. The model finds its own stopping point, works with any learning rate, and prevents overfitting automatically. Always use a separate validation set for early stopping—not your final test set.

For comparison, scikit-learn provides its own gradient boosting implementation. While XGBoost is generally faster and more feature-rich, sklearn's `GradientBoostingClassifier` integrates seamlessly with sklearn pipelines. The following example uses sklearn's implementation to demonstrate how validation accuracy evolves across boosting rounds:

!!! example "Numerical Example: Early Stopping in Action"

    ```python
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    import numpy as np

    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=5,
        n_redundant=10, n_clusters_per_class=3, random_state=42,
    )
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=4, random_state=42,
    )
    gb.fit(X_train, y_train)

    # Track validation accuracy at each stage
    val_scores = [np.mean(pred == y_val) for pred in gb.staged_predict(X_val)]
    best_n = np.argmax(val_scores) + 1

    print(f"Best validation at {best_n} trees: {val_scores[best_n-1]:.4f}")
    print(f"Final (300 trees):                 {val_scores[-1]:.4f}")
    print(f"Overfit penalty: {(val_scores[best_n-1] - val_scores[-1])*100:.1f} points")
    ```

    **Output:**

    ```
    Best validation at 14 trees: 0.9000
    Final (300 trees):           0.8800
    Overfit penalty: 2.0 points
    ```

    **Interpretation:** Validation accuracy peaks at just 14 trees, then declines as the model overfits. Training to 300 trees costs 2 percentage points of accuracy. Early stopping would have stopped at 14 trees automatically.

    *Source: `computations/module4_examples.py` — `demo_early_stopping()`*


### Key XGBoost Hyperparameters

The following table summarizes the primary hyperparameters for tuning XGBoost.

| Parameter | Range | Effect |
|-----------|-------|--------|
| `n_estimators` | 100--1000 | Number of boosting rounds. More rounds = more complex model. Use early stopping to find the optimal value. |
| `max_depth` | 3--10 | Maximum tree depth. Deeper trees capture more complex interactions but are more prone to overfitting. Start with 6. |
| `learning_rate` | 0.01--0.3 | Shrinkage factor applied to each tree's contribution. Lower values need more rounds but often generalize better. |
| `subsample` | 0.5--1.0 | Fraction of training samples used per tree. Values below 1.0 add stochastic regularization. |
| `colsample_bytree` | 0.5--1.0 | Fraction of features considered per tree. Reduces correlation between trees, similar to Random Forest's `max_features`. |
| `min_child_weight` | 1--10 | Minimum sum of instance weights in a leaf. Higher values prevent trees from learning overly specific patterns. |
| `reg_alpha` (alpha) | 0--10 | L1 regularization on leaf weights. Encourages sparsity (some leaves get zero weight). |
| `reg_lambda` (lambda) | 0--10 | L2 regularization on leaf weights. Shrinks leaf weights toward zero. Default is 1. |

The following configuration provides a practical starting point for most problems, balancing model capacity with regularization:

```python
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    random_state=42,
)
```

### LightGBM and CatBoost

**LightGBM** (Microsoft, 2017) is a gradient boosting framework optimized for speed and memory efficiency. Its two key innovations over XGBoost are Gradient-based One-Side Sampling (GOSS), which retains instances with large gradients (the hard examples) while randomly dropping instances with small gradients, and Exclusive Feature Bundling (EFB), which bundles mutually exclusive sparse features to reduce the effective feature count. Together, these reduce both the number of data points and features examined at each split, giving LightGBM training speeds that are often 5–10× faster than XGBoost on large datasets with little accuracy loss. It also grows trees leaf-wise (best-leaf-first) rather than level-wise, which tends to find better splits but requires careful control of `num_leaves` and `min_data_in_leaf` to avoid overfitting. LightGBM is the method of choice when training time or data scale is the primary constraint.

**CatBoost** (Yandex, 2017) addresses a specific pain point: categorical features. Standard gradient boosting requires converting categorical columns to numbers before training—typically via one-hot encoding or target encoding, both of which can introduce leakage or create high-dimensional sparse inputs. CatBoost uses an ordered target statistics scheme that encodes categories using historical observations only, avoiding leakage while preserving the categorical information efficiently. Beyond categorical handling, CatBoost implements symmetric (oblivious) trees—at each level, the same split is applied to all nodes—which trades some expressiveness for dramatically faster prediction and better resistance to overfitting. CatBoost also tends to require less hyperparameter tuning than XGBoost or LightGBM, making it a strong default when a dataset has many categorical columns.

As a starting rule, use XGBoost as the baseline. Switch to LightGBM when training time or dataset size is a constraint, and switch to CatBoost when categorical features are numerous or when you want a framework that performs well with minimal tuning.

### Bagging vs Boosting: When to Use Each

The choice between bagging and boosting depends on the problem characteristics.

| Scenario | Recommendation |
|----------|----------------|
| High-variance (deep trees) | Bagging (RF) |
| High-bias (shallow trees) | Boosting |
| Fast training needed | Bagging (parallelizable) |
| Best accuracy needed | Boosting (often wins) |
| Noisy labels | Bagging (more robust) |
| Need built-in interpretability | Random Forest (OOB + MDI available by default; both support SHAP equally) |

### Common Misconceptions

Several misconceptions about boosting methods deserve attention.

| Misconception | Reality |
|--------------|---------|
| "XGBoost is always best" | No Free Lunch. Linear models beat it on linear data. Neural networks beat it on images/text. |
| "Boosting can't overfit" | It certainly can. Use early stopping. |
| "More boosting rounds = better" | Unlike RF, more rounds increases overfit risk. |
| "XGBoost, LightGBM, CatBoost are completely different" | All gradient boosting variants. Similar core ideas. |

---

## Reflection Questions

1. Why does the wisdom of crowds work? Under what conditions would it fail?

2. A colleague says Random Forest can never overfit. How would you respond?

3. Why sample features at each split rather than once per tree?

4. When might boosting overfit more easily than bagging? What would you adjust?

5. A data scientist says they always use XGBoost because "it wins Kaggle." What's your response?

6. You have 5 models with accuracies 82%, 81%, 79%, 78%, 75%. Would you ensemble all 5? Why or why not?

---

## Practice Problems

1. Show step-by-step why ~63.2% of observations appear in a bootstrap sample of size $n$ drawn from $n$ observations. Your derivation should start with the per-draw probability of exclusion and arrive at $1 - e^{-1}$ using the limit definition of $e$.

2. A classification dataset has $d = 100$ features. How many features does Random Forest consider at each split by default? If you switch to `max_features='log2'`, how does that change the answer?

3. You train a Random Forest on a dataset where one feature is a unique customer ID (high cardinality, no predictive value). MDI ranks it as the most important feature. Permutation importance ranks it near zero. Explain why, and which measure you would report to a stakeholder.

4. Trace through 3 rounds of AdaBoost on a toy dataset of 5 examples: $[+, +, -, -, +]$ where the first stump misclassifies the two negative examples. Show how weights change after round 1 and what that means for the second stump.

5. You train XGBoost without early stopping and see training accuracy at 99% but test accuracy at 75%. Diagnose the cause, identify two hyperparameters you would adjust, and explain why each adjustment should help.

---

## Chapter Summary

Ensemble methods improve predictions by combining diverse models whose errors cancel out. The two major paradigms address different problems: bagging, exemplified by Random Forests, reduces variance by averaging independent trees, while boosting, exemplified by XGBoost, reduces bias through sequential learning from errors. Both paradigms offer feature importance measures that reveal which features the model relies on, though these measures indicate predictive power rather than the direction of a feature's effect. For boosting in particular, early stopping is essential to prevent overfitting as additional rounds add complexity. When choosing between the two, Random Forests offer robustness and ease of use, while XGBoost tends to deliver the highest accuracy when carefully tuned.

---

## What's Next

In Module 5, we tackle **Unsupervised Learning**, covering clustering (K-Means, hierarchical), dimensionality reduction (PCA), and the broader challenge of finding structure without labels. So far, we have always had a target variable to predict; in unsupervised learning, there is no target, and the goal shifts to discovering hidden patterns in the data.
