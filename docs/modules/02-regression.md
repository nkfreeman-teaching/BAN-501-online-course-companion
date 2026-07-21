# Module 2: Classical Machine Learning - Regression

## Introduction

Module 1 established the foundation: what ML is, how to prepare data, and how to evaluate models. Now we put that foundation to work. Regression is the workhorse of predictive analytics. When a business wants to predict sales, estimate prices, or forecast demand, regression is often the first tool they reach for. But we're not just going to use regression as a black box—we're going to understand *how* it works, including implementing gradient descent from scratch.

Why learn gradient descent? Because gradient descent is the foundation for training neural networks. Every deep learning model you've heard of—GPT, image classifiers, everything—learns through gradient descent. Understanding it for linear regression means understanding it for neural networks.

The closed-form solution for linear regression requires matrix inversion—O(d³) complexity where d is the number of features—that becomes computationally infeasible for neural networks with millions of parameters. Neural networks also have non-convex loss surfaces with many critical points (local minima and points where the gradient is zero but no true minimum exists—called saddle points, covered in Module 6); there's no mathematical formula to jump to the optimal weights. Gradient descent works for any differentiable function and scales to billions of parameters—the linear regression closed-form is a special case where we can skip the search.

---

## Learning Objectives

By the end of this module, you should be able to:

1. **Explain** the mechanics of linear regression including the least squares method
2. **Implement** gradient descent from scratch and understand its trade-offs
3. **Interpret** regression coefficients for business insights
4. **Diagnose** model issues through residual analysis
5. **Apply** regularization techniques (L1, L2, Elastic Net) to prevent overfitting
6. **Communicate** regression findings to non-technical stakeholders

---

## 2.1 Simple Linear Regression

This section covers the building blocks of linear regression: the decision model, the quality measure, gradient descent optimization, and how to interpret and diagnose the results.

### The Three Components of Every ML Model

Before diving into linear regression, here's a framework that applies to *every* supervised learning algorithm:

| Component | Question It Answers | Linear Regression |
|-----------|---------------------|-------------------|
| **Decision Model** | How do we transform inputs into predictions? | $\hat{y} = \beta_0 + \beta_1 x$ |
| **Quality Measure** | How do we evaluate prediction quality? | Sum of Squared Errors (SSE) |
| **Update Method** | How do we improve the model? | Gradient descent (or closed-form) |

This same pattern applies to every algorithm: logistic regression, decision trees, random forests, neural networks. The decision model changes, the quality measure may change, but the structure is always the same.

### The Goal of Linear Regression

Given input features, we want to predict a continuous output. We assume the relationship can be approximated by a line:

$$\hat{y} = \beta_0 + \beta_1 x$$

Where:
- $\hat{y}$ (y-hat) is the **predicted value**
- $\beta_0$ (beta-zero) is the **intercept**—the baseline prediction when x = 0
- $\beta_1$ (beta-one) is the **slope**—how much y changes for a one-unit change in x
- $x$ is the **input feature**

The key assumption is that a straight line is a reasonable approximation of the true relationship. Check linearity visually: scatter plots should show points around an imaginary straight line. Use `sns.pairplot()` for multiple regression. Correlation measures only *linear* association—a perfect U-shaped relationship has r=0. Always check residual plots after fitting; curved patterns reveal non-linearity. If non-linear, try transforms (log, square root), polynomial terms (x², x³), or inherently non-linear models (trees, neural networks).

### The Least Squares Method

How do we find the *best* line? We find the coefficients that minimize squared prediction errors:

$$\text{minimize } \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

For each data point, calculate the error (actual minus predicted), square it, and add them all up. The best line is the one that makes this sum as small as possible.

#### Why Squared Errors

Squared errors penalize large errors more than small ones—an error of 10 contributes 100 while an error of 1 contributes only 1—so the algorithm prioritizes avoiding big mistakes. The squared error function is also mathematically tractable: it is differentiable and convex, so we can find the minimum using calculus. Finally, under certain assumptions (detailed below), least squares gives the Best Linear Unbiased Estimator (BLUE), providing strong statistical properties.

#### When Squared Errors Are Not the Right Choice

MSE assumes symmetric, quadratic costs—each unit of over-prediction is penalized equally to each unit of under-prediction. That assumption doesn't always hold. In demand forecasting, under-predicting (stockouts that lose sales) might cost more than over-predicting (excess inventory). For outlier-heavy data, MAE or Huber loss are more robust because they don't square large errors. For classification, cross-entropy is more appropriate than MSE. The right quality measure encodes your actual business costs. When asymmetric costs apply, use **weighted least squares** (assign higher weights to observations where errors are more costly) or **quantile regression** (which systematically over- or under-predicts at a chosen quantile—useful for safety stock planning). In deep learning, you can define arbitrary custom loss functions. Start with squared errors and only add complexity when you have clear business justification for asymmetric costs.

The closed-form solution gives exact values for the coefficients:

$$\beta_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$$

$$\beta_0 = \bar{y} - \beta_1\bar{x}$$

### Linear Regression Assumptions

For statistical inference to be valid, five assumptions must hold. The relationship between X and Y must be linear (linearity), and the observations must be independent of each other (independence). The variance of residuals should remain constant across all levels of X (homoscedasticity), and the residuals should be normally distributed (normality). Finally, for multiple regression, the predictors should not be too highly correlated with one another (no multicollinearity).

#### When Assumptions Are Violated

| Violation | Symptom | Solution |
|-----------|---------|----------|
| Non-linearity | Curved pattern in residuals | Transform variables, add polynomial terms |
| Heteroscedasticity | Fan-shaped residual plot | Transform Y, use robust standard errors |
| Non-normality | Q-Q plot deviates from line | Transform Y, use bootstrap |
| Autocorrelation | Patterns in time-ordered residuals | Time series methods |
| Multicollinearity | Unstable coefficients, high VIF | Remove one predictor, combine features, or use Ridge (see Section 2.2) |

#### Common Transformations

| Transform | Formula | Best for |
|-----------|---------|----------|
| Log | $\log(x)$ | Right-skewed data, multiplicative relationships |
| Square root | $\sqrt{x}$ | Count data, mild right skew |
| Box-Cox | $(x^\lambda - 1)/\lambda$ | Automated selection—finds optimal λ. For λ = 0, the transform reduces to log(y). Requires y > 0. |

The log transform is the workhorse—it handles right-skewed distributions (common in business data like income, prices, counts) and converts multiplicative relationships to additive ones. With a log-transformed target, coefficients represent **percentage changes**: a one-unit increase in x multiplies y by $e^{\beta_1}$. For small β (roughly |β| < 0.2), this approximates β × 100% change. If both x and y are logged, coefficients represent **elasticities**: a 1% change in x → β₁% change in y. Always back-transform predictions before evaluating metrics, and document which scale coefficients are interpreted on.

### Gradient Descent

We could use the closed-form solution, but gradient descent is worth learning because it's the foundation for all neural network training.

#### The Algorithm

1. Start with random values for $\beta_0$ and $\beta_1$
2. Calculate the gradient—the direction of steepest error increase
3. Update parameters in the **opposite** direction—downhill toward lower error
4. Repeat until **convergence**—when parameters have stabilized and further iterations produce no meaningful improvement

#### The Landscape Intuition

Imagine a hilly terrain where your position is determined by your parameter values ($\beta_0$, $\beta_1$) and the elevation is your error (MSE). You're dropped somewhere on this terrain—probably high up—and want to reach the lowest valley. You can't see the whole landscape, but you can feel which way is steepest *right where you're standing*. The gradient tells you that direction. Each step, you walk downhill proportional to the steepness. For linear regression, this landscape is bowl-shaped (convex) with exactly one lowest point—you'll always reach it eventually. Neural networks have more complex terrain with multiple valleys; you'll find *a* valley, but maybe not the deepest one.

#### The Gradients

To derive the gradient, apply the chain rule to $MSE = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$:

$$\frac{\partial MSE}{\partial \beta_0} = \frac{1}{n}\sum 2(y_i - \hat{y}_i) \cdot \frac{\partial (y_i - \hat{y}_i)}{\partial \beta_0} = \frac{1}{n}\sum 2(y_i - \hat{y}_i) \cdot (-1) = -\frac{2}{n}\sum(y_i - \hat{y}_i)$$

$$\frac{\partial MSE}{\partial \beta_1} = \frac{1}{n}\sum 2(y_i - \hat{y}_i) \cdot \frac{\partial (y_i - \hat{y}_i)}{\partial \beta_1} = \frac{1}{n}\sum 2(y_i - \hat{y}_i) \cdot (-x_i) = -\frac{2}{n}\sum(y_i - \hat{y}_i) \cdot x_i$$

The $-x_i$ term in the $\beta_1$ gradient reflects the chain rule: changing $\beta_1$ changes $\hat{y}_i = \beta_0 + \beta_1 x_i$ by $x_i$. In compact form:

$$\frac{\partial MSE}{\partial \beta_0} = -\frac{2}{n}\sum(y_i - \hat{y}_i)$$

$$\frac{\partial MSE}{\partial \beta_1} = -\frac{2}{n}\sum(y_i - \hat{y}_i) \cdot x_i$$

#### Update Rules

$$\beta_0 \leftarrow \beta_0 - \alpha \cdot \frac{\partial MSE}{\partial \beta_0}$$

$$\beta_1 \leftarrow \beta_1 - \alpha \cdot \frac{\partial MSE}{\partial \beta_1}$$

Here $\alpha$ is the **learning rate**—how big a step we take each iteration. Common stopping criteria for convergence: loss change below a threshold (1e-6), small gradient magnitude, or a maximum iteration count—use a combination in practice. For linear regression, the loss surface is convex with one global minimum; for neural networks, you'll find a local minimum (usually good enough). Monitor the loss curve—oscillating or increasing loss suggests the learning rate is too high.

Here is a complete implementation of gradient descent for linear regression.

```python
def gradient_descent_linear_regression(
    X, y,
    learning_rate=0.01,
    n_iterations=1000,
    tolerance=1e-6
):
    n = len(X)

    # Initialize parameters randomly
    beta_0 = np.random.randn()
    beta_1 = np.random.randn()

    history = []

    for i in range(n_iterations):
        # Predictions
        y_pred = beta_0 + beta_1 * X

        # Compute gradients
        d_beta_0 = -2/n * np.sum(y - y_pred)
        d_beta_1 = -2/n * np.sum((y - y_pred) * X)

        # Update parameters
        beta_0 = beta_0 - learning_rate * d_beta_0
        beta_1 = beta_1 - learning_rate * d_beta_1

        # Track loss
        mse = np.mean((y - y_pred)**2)
        history.append(mse)

        # Check convergence
        if i > 0 and abs(history[-1] - history[-2]) < tolerance:
            break

    return beta_0, beta_1, history
```

!!! example "Numerical Example: Watching Gradient Descent Converge"

    ```python
    import numpy as np

    # Generate data: y = 3 + 2x + noise
    np.random.seed(42)
    X = np.random.uniform(low=0, high=10, size=100)
    y = 3.0 + 2.0 * X + np.random.normal(loc=0, scale=1.5, size=100)

    # Gradient descent with learning_rate=0.02
    # Note: initialized at zero here for reproducible output.
    # The function above uses random initialization (np.random.randn()),
    # which is more typical in practice but produces different starting values each run.
    beta_0, beta_1 = 0.0, 0.0
    for iteration in range(501):
        y_pred = beta_0 + beta_1 * X
        mse = np.mean((y - y_pred)**2)
        if iteration in [0, 10, 50, 100, 200, 500]:
            print(f"Iter {iteration:3d}: β₀={beta_0:.4f}, β₁={beta_1:.4f}, MSE={mse:.4f}")
        d_beta_0 = -2/100 * np.sum(y - y_pred)
        d_beta_1 = -2/100 * np.sum((y - y_pred) * X)
        beta_0 = beta_0 - 0.02 * d_beta_0
        beta_1 = beta_1 - 0.02 * d_beta_1
    ```

    **Output:**

    ```
    Iter   0: β₀=0.0000, β₁=0.0000, MSE=188.2950
    Iter  10: β₀=0.6787, β₁=2.3374, MSE=3.7994
    Iter  50: β₀=1.6304, β₁=2.1911, MSE=2.6278
    Iter 100: β₀=2.3539, β₁=2.0799, MSE=2.0813
    Iter 200: β₀=3.0051, β₁=1.9798, MSE=1.8434
    Iter 500: β₀=3.3115, β₁=1.9328, MSE=1.8149
    ```

    **Interpretation:** Starting from zeros (iteration 0: β₀=0, β₁=0, MSE=188), gradient descent iteratively improves toward the true parameters (β₀=3, β₁=2). MSE drops rapidly at first (188 to 3.8 in just 10 iterations), then refines more slowly. The betas and MSE shown at each iteration are consistent—both reflect the model's state before that iteration's update. The final estimates (3.31, 1.93) are close to truth—the remaining gap is due to noise in the data, not algorithm failure.

    *Source: `computations/module2_examples.py` – `demo_gradient_descent_convergence()`*


### Learning Rate Trade-offs

The learning rate $\alpha$ controls how aggressively gradient descent updates the parameters, and choosing it well is crucial.

| Too Small | Just Right | Too Large |
|-----------|------------|-----------|
| Very slow convergence | Converges in reasonable time | Overshoots the minimum |
| Safe but inefficient | Reaches good solution | Can diverge (loss increases!) |

#### What the Loss Curve Tells You

Plot MSE vs. iteration number to diagnose learning rate issues. When the learning rate is too small (e.g., α = 0.0001), the curve creeps downward slowly—you might need 10,000+ iterations to converge, with a shallow but always-decreasing slope. When the learning rate is well-chosen (e.g., α = 0.01), the curve drops quickly at first and then flattens as you approach the minimum, converging in hundreds rather than thousands of iterations. When the learning rate is too large (e.g., α = 0.1), the curve may oscillate wildly or explode upward; if loss increases iteration-over-iteration, your learning rate is too high. As a rule of thumb, if your loss keeps *increasing*, reduce the learning rate by a factor of 10.

Adaptive learning rate methods automatically adjust during training—learning rate schedules decrease the rate over time, and adaptive optimizers adjust per parameter based on gradient history. These methods (including Adam, the standard optimizer for deep learning) are covered in detail in Module 6. For scikit-learn's linear regression, optimization is handled automatically—you don't need to set a learning rate.

!!! example "Numerical Example: Learning Rate Effects"

    Same dataset, three different learning rates:

    | Learning Rate | Status | Iterations | Final MSE |
    |---------------|--------|------------|-----------|
    | 0.0001 | Not converged | 1000 | 4.04 |
    | 0.01 | Converged | 920 | 1.81 |
    | 0.1 | **DIVERGED** | 6 | ∞ |

    **Interpretation:** With α=0.0001 (too small), the algorithm made progress but didn't converge in 1000 iterations—MSE is still far from optimal. With α=0.01 (just right), it converged in 920 iterations to the best solution (MSE=1.81). With α=0.1 (too large), the algorithm diverged after just 6 iterations—the loss exploded to infinity. When you see increasing loss, immediately reduce the learning rate by 10x.

    *Source: `computations/module2_examples.py` – `demo_learning_rate_effects()`*


### Using statsmodels for Regression

In practice, you will use libraries rather than hand-coding gradient descent. When you need rich statistical output—p-values, confidence intervals, and diagnostic tests—statsmodels is the standard choice.

!!! note "Note on pandas in a polars-first course"

    Statsmodels' formula API (`smf.ols`) parses R-style formulas against pandas-style DataFrames; it does not understand polars `DataFrame` objects directly. The example below imports pandas for that reason. The rest of the course remains polars-first — when you have a polars source, pass `df.to_pandas()` to the formula API and continue working in polars elsewhere.

```python
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

# Create synthetic data (or use your own DataFrame)
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'advertising': np.random.uniform(low=1000, high=10000, size=n),
    'price': np.random.uniform(low=10, high=50, size=n),
})
df['sales'] = 50000 + 2.5 * df['advertising'] + 1200 * df['price'] + np.random.normal(loc=0, scale=5000, size=n)

# R-style formula interface
model = smf.ols(
    formula='sales ~ advertising + price',
    data=df
)
results = model.fit()
print(results.summary())
```

The statsmodels summary contains several statistics worth understanding.

The R² (Coefficient of Determination) measures the proportion of variance your model explains:

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

An R² of 0.75, for instance, means 75% of the variance is explained and 25% remains unexplained.

### Adjusted R²

When comparing models with different numbers of predictors, R² alone can be misleading because adding any predictor (even noise) will never decrease R². Adjusted R² penalizes model complexity:

$$R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - d - 1}$$

where $n$ is the number of observations and $d$ is the number of features. Unlike R², adjusted R² can decrease when a predictor adds more noise than signal. This makes it a better metric for comparing models with different numbers of features. Statsmodels reports adjusted R² automatically in the summary output—look for "Adj. R-squared."

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
n = len(y_test)
d = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - d - 1)
```

The p-value for each coefficient tests whether that coefficient differs from zero. A p-value below 0.05 is the conventional threshold for "statistically significant," though p-values do not tell you the effect *size*. The 0.05 threshold is a historical convention from R.A. Fisher, not a magic number. Problems: with enough data, trivial effects become "significant"; with little data, real effects may not be. Modern practice: report exact p-values, consider effect sizes and confidence intervals, and remember that practical significance matters more than statistical significance in business—a statistically significant 0.1% improvement might not be worth implementing.

Confidence intervals quantify the uncertainty around each estimate. A 95% CI of [1.8, 3.2] means you can be 95% confident the true effect is between $1.80 and $3.20. If the CI includes zero, the effect is not statistically significant. The F-statistic tests whether the model as a whole is useful—whether it predicts better than simply using the mean. While statsmodels excels at providing these diagnostic statistics, you should use scikit-learn instead when building ML pipelines, when prediction is the main goal, or when you need cross-validation and hyperparameter tuning.

### Interpreting Coefficients

The standard interpretation of a regression coefficient is: "A one-unit increase in X is associated with a $\beta_1$ change in Y, *holding all else constant*."

For example, consider this fitted model:
```
sales = 50,000 + 2.5 × advertising + 1,200 × sales_staff
```

The intercept tells you that baseline sales are $50,000 (when advertising = 0 and sales_staff = 0). The advertising coefficient means each $1 in advertising is associated with $2.50 more in sales (250% ROI), and the sales_staff coefficient means each additional staff member is associated with $1,200 more in sales.

#### Correlation Does Not Imply Causation

Regression shows *association*, not *causation*. When we say "Each $1 in advertising → $2.50 more in sales," we mean they're associated. We haven't proven advertising *causes* sales. A classic example: ice cream sales and drowning deaths are positively correlated, but both are driven by summer heat—a confounding variable.

**Going Deeper:** Establishing causation requires experimental design or careful causal inference methods. Randomized experiments such as A/B tests are the gold standard, while natural experiments (policy changes affecting some regions) create quasi-random groups. Instrumental variables find factors that affect treatment but not outcome directly, and causal inference frameworks like propensity score matching and difference-in-differences try to estimate effects from observational data. With observational regression alone, you have association—to claim causation, you need a convincing argument for why confounders are controlled. These methods are beyond the scope of this module, but they matter whenever your analysis is expected to support a business decision about an intervention.

### Residual Analysis

Residuals—the differences between actual and predicted values—are your primary tool for diagnosing model problems.

$$e_i = y_i - \hat{y}_i$$

A positive residual means the model under-predicted (actual was higher than predicted); a negative residual means the model over-predicted. If the model is good and assumptions hold, residuals should look like random noise centered at zero, with no pattern related to fitted values or any predictor. Systematic patterns in the residuals signal that the model is missing something—structure you haven't yet captured.

#### Diagnostic Plots

The residuals vs. fitted values plot should show random scatter around zero; a curved pattern indicates non-linearity, while a funnel shape indicates heteroscedasticity. The Q-Q plot compares residuals against theoretical normal quantiles—a straight line means the normality assumption is satisfied, and curves at the ends indicate heavy or light tails.

```python
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

plt.subplot(1, 3, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')

plt.subplot(1, 3, 3)
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel('Residuals')
plt.title('Residual Distribution')

plt.tight_layout()
plt.show()
```

#### Reading Residual Plots — A Pattern Recognition Guide

| What You See | What It Means | Action |
|--------------|---------------|--------|
| Random scatter around zero | Good! Assumptions satisfied | Proceed with model |
| U-shape or inverted-U curve | Non-linear relationship missed | Add polynomial terms or transform variables |
| Funnel shape (wider on one side) | Heteroscedasticity—variance changes | Transform Y (log), use robust standard errors |
| Clusters or groups | Subgroups with different patterns | Add categorical variable, consider separate models |
| Pattern over time (if ordered) | Autocorrelation | Time series methods, lagged variables |

The residual plot is your diagnostic dashboard. Even if R² looks great, always check residuals—patterns reveal problems that summary statistics hide. A high R² paired with a patterned residual plot still indicates a bad model, because the pattern means you are missing structure in the data.

To fix a curved residual pattern, first identify which feature causes it by plotting residuals against each predictor. Next, try transforms—log for diminishing returns, square root for counts, polynomial terms (x², x³). If simple transforms are not enough, use `PolynomialFeatures(degree=2)` with regularization. If transforms still do not help, consider non-linear models (trees, GAMs, neural networks). Finally, verify the fix—residuals should show random scatter.

### Common Misconceptions

The following misconceptions frequently arise when interpreting simple linear regression results.

| Misconception | Reality |
|--------------|---------|
| "Correlation implies causation" | Regression shows association only. Causation requires experimental design or causal inference methods. |
| "High R² means the model is good" | R² can be high due to overfitting. Must check test set performance and residual plots. |
| "The intercept is always meaningful" | Often it's not (e.g., salary when experience = 0 years). Focus on slopes for interpretation. |
| "Larger coefficients mean more important features" | Only true if features are on the same scale. Use standardized coefficients to compare. |

---

## 2.2 Multiple Linear Regression

This section extends simple linear regression to multiple predictors, covering confounding, multicollinearity, and regularization techniques that prevent overfitting.

### Multiple Predictors

In the real world, we rarely have just one predictor:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_d x_d$$

#### Why Multiple Predictors

Multiple predictors matter because (i) a single predictor rarely captures the full story, (ii) additional predictors let you control for confounding variables, and (iii) including relevant predictors improves prediction accuracy. Each coefficient represents the effect of that variable *while controlling for the others*, which is different from running separate simple regressions.

Multiple regression coefficients are **partial effects**—the effect of one variable holding others constant. Simple regression confounds direct and indirect effects through correlated variables. If experience and education are correlated, simple regression conflates their effects; multiple regression "controls for" education when estimating experience's effect. Sometimes adding variables can even flip coefficient signs (Simpson's paradox)—the Berkeley admissions example showed apparent disadvantage for women overall that reversed within each department.

### Confounding Variables

Consider the difference between a simple regression of Salary on Experience ($\beta = \$5,000$ per year) and a multiple regression of Salary on Experience plus Education ($\beta_{exp} = \$3,500$ per year). The coefficient for experience dropped because education was **confounded** with experience. People with more experience often have more education. The simple regression was attributing some of education's effect to experience.

#### The Causal Diagram View

This type of arrow diagram is called a **Directed Acyclic Graph (DAG)**—each arrow represents a causal direction, and the graph has no cycles (no variable causes itself through any chain of arrows):

```
         Education
         ↙      ↘
    Experience → Salary
```

Education affects both experience (more education often means less early work experience) and salary. When we don't control for education, we're measuring a **back-door path**—a non-causal route between two variables that flows through a common cause. Here: Experience ← Education → Salary. This path is "back-door" because the arrow into Experience points backward, away from the variable we care about. Simple regression picks up this indirect association alongside the direct effect. Multiple regression "closes" this back-door path by holding education constant, isolating experience's direct effect.

#### Three Types of Variables to Distinguish

A **confounder** (like Education above) affects both X and Y, and you should include it to avoid bias. A **mediator** (e.g., Skills on the Experience-to-Salary path) lies on the causal chain between X and Y; include it only if you want indirect effects removed. A **collider** (e.g., "Got Promoted," affected by both Experience and Salary) is different: controlling for it *creates* spurious associations, so you should not include it.

To identify confounders, ask: "What could affect both X and Y?" Draw causal diagrams—confounders have arrows TO both predictor and outcome. Check correlations, but correlation alone isn't sufficient; you need domain reasoning. Don't throw every variable in—mediators (on the causal path) or colliders (affected by both) can introduce bias. Include variables that theory suggests are confounders and were measured before the treatment.

### Multicollinearity

Multicollinearity occurs when predictors are highly correlated with one another.

#### Symptoms

Common signs of multicollinearity include (i) coefficients that change dramatically when you add or remove features, (ii) high R² paired with few individually significant predictors, and (iii) coefficient signs that seem wrong given domain knowledge.

#### The See-Saw Analogy

Imagine predicting house price from both "total square feet" and "number of rooms." These are highly correlated—bigger houses have more rooms. In the model, their coefficients are like two kids on a see-saw that must balance to produce the right prediction. If one kid (coefficient) goes up, the other must go down. The model can balance them many different ways: (sqft=+100, rooms=-50), (sqft=+50, rooms=0), (sqft=+150, rooms=-100)—all giving similar predictions. The *total effect* is stable, but the *individual coefficients* are unstable. That's why coefficients jump around with small data changes when multicollinearity is present.

### Detecting Multicollinearity: VIF

The Variance Inflation Factor (VIF) quantifies how much a predictor's variance is inflated by collinearity:

$$VIF_j = \frac{1}{1 - R_j^2}$$

Where $R_j^2$ is R² from regressing feature j on all other features.

| VIF Value | Interpretation |
|-----------|----------------|
| VIF = 1 | No correlation with other features |
| VIF > 5 | Moderate multicollinearity—investigate |
| VIF > 10 | Serious multicollinearity—must address |

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
for i in range(X.shape[1]):
    vif = variance_inflation_factor(X.values, i)
    print(f"{feature_names[i]}: VIF = {vif:.2f}")
```

!!! example "Numerical Example: VIF Multicollinearity Detection"

    **Scenario 1:** Three independent features

    | Feature | VIF |
    |---------|-----|
    | x1 | 1.03 |
    | x2 | 1.01 |
    | x3 | 1.02 |

    **Scenario 2:** Added x4 which is highly correlated with x1 (r = 0.994)

    | Feature | VIF |
    |---------|-----|
    | x1 | 88.60 |
    | x2 | 1.02 |
    | x3 | 1.03 |
    | x4 (corr with x1) | 88.12 |

    **Interpretation:** Independent features have VIF ≈ 1, indicating no multicollinearity. When x4 (nearly identical to x1) is added, both x1 and x4 jump to VIF ≈ 88—severe multicollinearity. The x2 and x3 VIF values remain unaffected because they're not involved in the collinear relationship. VIF > 10 requires action: remove x4, combine with x1, or use Ridge.

    *Source: `computations/module2_examples.py` – `demo_vif_multicollinearity()`*


#### Solutions for Multicollinearity

You can address multicollinearity by (i) removing one of the correlated features, (ii) combining features through averaging or PCA, (iii) using regularization (Ridge handles this well), or (iv) simply accepting it if prediction is your only goal.

Multicollinearity does not affect prediction accuracy—but it creates problems for interpretation. Coefficients become unstable (jumping around with small data changes), standard errors inflate (true effects appear "not significant"), and signs may reverse. If your goal is purely prediction, ignore it. If you need interpretation, address it.

### Why Regularize?

Regularization prevents overfitting by penalizing large coefficients:

$$\text{minimize } \sum(y_i - \hat{y}_i)^2 + \lambda \cdot \text{penalty}(\beta)$$

The parameter $\lambda$ controls how strong the penalty is.

#### The Mechanism: Penalizing Noise-Fitting

Here is a concrete illustration. Suppose you fit OLS on 20 data points with 15 features and get:

$$\hat{y} = 3 + 847 x_1 - 612 x_2 + 503 x_3 - \ldots$$

Those large coefficients (847, -612, 503) are a warning sign—the model is likely latching onto noise in the small sample. A coefficient of 847 on a standardized feature means a one-standard-deviation change in $x_1$ shifts $\hat{y}$ by 847 units—almost certainly too large for a real effect. When you add a Ridge penalty with $\lambda = 10$, the optimizer must weigh the benefit of each unit of fit against the cost $\lambda \cdot \beta^2$. To keep $\beta_1 = 847$, it pays a penalty of $10 \times 847^2 \approx 7{,}174{,}090$—a steep price. The optimizer settles for a smaller coefficient (say, 42) that captures the genuine signal without that penalty. If the feature is pure noise, even a modest penalty makes zeroing it out the better choice.

Signs of overfitting: large coefficients with small samples, large standard errors, or a wide gap between training and test performance. Regularization forces the model to justify large coefficients—if a feature is fitting real patterns, the benefit outweighs the penalty; if fitting noise, it won't. Note: "large" depends on feature scale—always standardize before applying regularization.

### L1 Regularization (Lasso)

The Lasso penalty is $\lambda \sum|\beta_j|$. Lasso can shrink coefficients **exactly to zero**, which amounts to automatic feature selection.

#### The Geometry of Regularization

Picture a 2D space where each axis is a coefficient (β₁, β₂). The loss function forms elliptical contours—concentric ovals centered on the OLS solution. Regularization adds a constraint: "stay within this budget region."

For L1 (Lasso), the budget region is a diamond with corners touching the axes. As you shrink the budget, the loss contours first touch a corner—where one coefficient equals zero. The diamond's sharp corners make this likely. For L2 (Ridge), the budget region is a circle. Loss contours touch it tangentially, almost never on an axis. Coefficients shrink smoothly toward zero but rarely reach exactly zero.

This geometry matters for feature selection: L1's diamond naturally produces sparse solutions (some coefficients = 0), while L2's circle keeps all features, just smaller. Additionally, the L1 gradient is constant (±1) regardless of how small β gets—always pulling toward zero—while the L2 gradient (2β) weakens as β approaches zero. In higher dimensions (many features), the L1 constraint region is a hyperdiamond with many corners and edges, each corresponding to a subspace where one or more coefficients equal zero—so the sparsity-inducing property of L1 becomes more pronounced, not less, as the number of features grows.

#### When to Use Lasso

Lasso is a good choice when you suspect many features are irrelevant, when you want an interpretable sparse model, or when feature selection is an explicit goal.

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train_scaled, y_train)

# See which features were selected (non-zero coefficients)
selected_features = [f for f, c in zip(feature_names, lasso.coef_) if c != 0]
print(f"Selected {len(selected_features)} features")
```

!!! example "Numerical Example: Lasso Feature Selection"

    True model: y = 5·x1 + 3·x2 − 2·x3 (features x4–x10 are pure noise)

    *Note: scikit-learn's `alpha` parameter corresponds to course notation $\lambda$ (regularization strength). The columns below use $\lambda$ for consistency.*

    | Feature | $\lambda$=0.01 | $\lambda$=0.1 | $\lambda$=0.5 | $\lambda$=1.0 |
    |---------|--------|-------|-------|-------|
    | x1 | 4.50 | 4.37 | 3.89 | 3.29 |
    | x2 | 3.41 | 3.32 | 2.92 | 2.43 |
    | x3 | −2.07 | −1.94 | −1.49 | −0.93 |
    | x4 | 0.09 | **0** | **0** | **0** |
    | x5 | 0.03 | **0** | **0** | **0** |
    | x6 | −0.07 | **0** | **0** | **0** |
    | x7–x10 | small | **0** | **0** | **0** |
    | **Non-zero** | **10** | **3** | **3** | **3** |

    **Interpretation:** At low regularization ($\lambda$=0.01), all 10 features have non-zero coefficients—noise features show small spurious effects. As $\lambda$ increases, noise features (x4–x10) shrink exactly to zero, leaving only the three true predictors. By $\lambda$=0.1, Lasso has automatically identified which features matter. This is automatic feature selection in action.

    *Source: `computations/module2_examples.py` – `demo_lasso_feature_selection()`*


### L2 Regularization (Ridge)

The Ridge penalty is $\lambda \sum\beta_j^2$. It shrinks all coefficients toward zero but **never exactly zero**.

#### When to Use Ridge

Ridge is a good choice when multicollinearity is present, when all features are potentially relevant, or when prediction accuracy is the main goal.

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
```

### Elastic Net: Combining L1 and L2

Elastic Net combines both penalties: $\lambda_1 \sum|\beta_j| + \lambda_2 \sum\beta_j^2$.

#### Benefits of Elastic Net

Elastic Net inherits feature selection from L1 (it can zero out coefficients) and stability from L2 (it handles correlated features better), making it more flexible than either approach alone.

#### When to Use Elastic Net

Elastic Net is the right choice when you have many features, some of which are correlated with each other. Pure Lasso tends to arbitrarily pick one from a group of correlated predictors and zero out the rest—which can be unstable and misleading. Elastic Net keeps groups of correlated features together (the L2 part) while still zeroing out truly irrelevant features (the L1 part). A practical heuristic: start with Elastic Net when you have more than 20 features and you're not sure how many matter. If you have strong reason to believe the true model is sparse (few important features), lean toward Lasso. If you expect all features to contribute at least a little, lean toward Ridge.

#### Understanding l1_ratio

The `l1_ratio` parameter controls the mix between L1 and L2:
- `l1_ratio=1.0` → pure Lasso (all L1 penalty)
- `l1_ratio=0.0` → pure Ridge (all L2 penalty)
- `l1_ratio=0.5` → equal mix (the default starting point)

In practice, try values such as 0.1, 0.5, 0.7, 0.9, 0.95, 1.0 via cross-validation. Values above 0.5 lean toward sparsity; values below 0.5 lean toward shrinkage without selection.

```python
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(
    alpha=0.1,       # Overall regularization strength (λ in course notation)
    l1_ratio=0.5,    # 0 = pure Ridge, 1 = pure Lasso, 0.5 = equal mix
    random_state=42
)
elastic.fit(X_train_scaled, y_train)
```

!!! example "Numerical Example: Ridge vs Lasso Comparison"

    True model: y = 3·x1 + 2·x2 + 1.5·x3 (x4 is noise, x1 and x2 are correlated with r=0.89)

    | Feature | OLS | Ridge | Lasso |
    |---------|-----|-------|-------|
    | x1 | 2.88 | 2.86 | 2.84 |
    | x2 (corr) | 2.05 | 2.07 | 1.98 |
    | x3 | 1.49 | 1.48 | 1.38 |
    | x4 (noise) | 0.08 | 0.08 | **0.00** |

    **Interpretation:** All methods recover approximate true coefficients. The key difference is in handling x4 (noise): Ridge keeps a small non-zero coefficient (0.08); Lasso zeros it out completely. Both Ridge and Lasso shrink coefficients toward zero, but only Lasso performs selection. When x1 and x2 are correlated, Ridge distributes weight across both; Lasso might keep one and drop the other (not shown here, but common with stronger regularization).

    **When to choose which:**
    - **Ridge:** When all features likely contribute, especially with multicollinearity
    - **Lasso:** When you want automatic feature selection (sparse model)
    - **Elastic Net:** When you want both—feature selection plus stability

    *Source: `computations/module2_examples.py` – `demo_ridge_vs_lasso()`*


### Choosing Regularization Strength

Cross-validation is the standard way to select the regularization parameter.

```python
from sklearn.linear_model import LassoCV, RidgeCV

# Automatic alpha selection via CV
lasso_cv = LassoCV(
    alphas=np.logspace(-4, 1, 50),
    cv=5
)
lasso_cv.fit(X_train_scaled, y_train)
print(f"Best alpha: {lasso_cv.alpha_}")
```

Always scale your features before regularization, because regularization penalizes large coefficients and features on different scales are penalized unfairly.

#### Back-Transforming Coefficients to Original Units

After fitting on scaled data, you need to reverse the scaling to report coefficients that stakeholders can interpret. StandardScaler transforms each feature as $x_{j,scaled} = (x_j - \mu_j) / \sigma_j$, where $\mu_j$ and $\sigma_j$ are the training mean and standard deviation. Reversing this gives:

$$\beta_{j,original} = \frac{\beta_{j,scaled}}{\sigma_j}$$

This works because a one-unit change in the original feature corresponds to a $1/\sigma_j$ change in the scaled feature, so the original-scale coefficient is smaller by a factor of $\sigma_j$.

The intercept requires an additional correction for the mean shift:

$$\beta_{0,original} = \beta_{0,scaled} - \sum_j \frac{\beta_{j,scaled} \cdot \mu_j}{\sigma_j}$$

This corrects for the fact that standardization shifted each feature by its mean—the intercept must absorb that shift when you move back to original units. In code:

```python
original_coefs = model.coef_ / scaler.scale_
original_intercept = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)
```

Back-transform for business communication ("each $1,000 in advertising → X more sales"); keep scaled coefficients for comparing feature importance within the same model. Save your scaler object so you have access to `mean_` and `scale_` attributes.

### Linear Regression as a Neural Network

Linear regression can be viewed as the simplest possible neural network, an insight that will pay off in Module 6.

```
Input Layer          Output Layer

   x₁ ──── w₁ ────┐
                   ├──→ Σ + b ──→ ŷ
   x₂ ──── w₂ ────┘
```

Linear regression is a neural network with no hidden layers. The weights (w) correspond to the coefficients ($\beta$), the bias (b) corresponds to the intercept ($\beta_0$), and the operation is the same: sum the weighted inputs and add the bias. Module 6 adopts the $w$/$b$ notation standard in deep learning frameworks—so when you see $w$ and $b$ there, you're seeing the same quantities you learned here as $\beta_1$ and $\beta_0$. When we add hidden layers and non-linear activations, we get deep learning, but the foundation—weighted sums optimized by gradient descent—is exactly what we learned here.

### Common Misconceptions

These misconceptions arise frequently around regularization and model complexity.

| Misconception | Reality |
|--------------|---------|
| "Regularization always hurts training performance" | True, but that's the point! We sacrifice training fit for better generalization. |
| "Lasso always performs feature selection" | Only with sufficient regularization. Very small alpha may keep all features. |
| "More features always improve the model" | Only if they're informative. Irrelevant features add noise and overfitting risk. |
| "Ridge is inferior because it doesn't zero out coefficients" | Ridge is often better when all features matter. Lasso is for sparse solutions. |

---

## Reflection Questions

1. You implement gradient descent and the loss keeps increasing. What's likely wrong? How would you fix it?

2. A regression coefficient for 'ice cream sales' on 'drowning deaths' is positive and statistically significant. Should ice cream vendors be concerned about causing drownings?

3. Your model has R² = 0.95 but the residual plot shows a clear curved pattern. Is this a good model?

4. When would you prefer Lasso over Ridge regression? Give a business scenario.

5. You add more features to your model and R² on training data increases, but test set performance decreases. What's happening?

6. How would you explain regularization to a business stakeholder without using math?

---

## Practice Problems

1. A fitted model for monthly store revenue (in dollars) is:

   $$\hat{y} = 12{,}400 + 3.20 \cdot \text{advertising} + 870 \cdot \text{staff} - 150 \cdot \text{competitors}$$

   where advertising is in dollars, staff is headcount, and competitors is the number of competing stores within 5 miles. (a) Interpret each coefficient in a sentence a store manager could act on. (b) What is the predicted revenue for a store with $2,000 in advertising, 4 staff, and 3 nearby competitors? (c) Is the intercept meaningful here? Why or why not?

2. Diagnose issues from residual plots: given a residual-vs-fitted plot that fans out as fitted values increase, identify the violation, name the assumption it breaks, and propose one corrective action.

3. You compute VIF for a model with three features and get values of 1.1, 12.4, and 11.8. What do these values tell you? Which features are implicated, and what are three options for addressing the problem?

4. You have 500 observations, 40 features, and domain knowledge that roughly 5–10 of them drive most of the outcome. Which regularization method would you start with—Lasso, Ridge, or Elastic Net—and why?

5. Write a two-paragraph memo to a non-technical marketing director explaining (a) what the advertising coefficient of 3.20 means for budget decisions and (b) why you can't conclude that spending more on advertising *causes* higher revenue.

---

## Chapter Summary

This module covered regression from first principles through practical application. First, linear regression minimizes squared errors to find the best-fit line—the math is elegant, but the intuition is simple: find the line that makes predictions as close as possible to reality. Second, gradient descent iteratively optimizes parameters and serves as the foundation for all neural network training, where learning rate matters: too small is slow, too large is unstable. Third, coefficients show association rather than causation, and you must be careful how you communicate this distinction. Fourth, residual analysis reveals assumption violations, so you should always check your residual plots—a high R² with a patterned residual plot is still a bad model. Fifth, regularization prevents overfitting: Lasso selects features, Ridge handles multicollinearity, and Elastic Net does both. Finally, communication matters—translate statistics to business impact, include limitations, and make recommendations actionable.

---

## What's Next

In Module 3, we tackle classification methods, including logistic regression (which extends linear regression to classification), decision boundaries and probability estimation, classification metrics in depth, and handling imbalanced classes. You will apply everything from Module 2—the same data preparation workflow, gradient descent concepts, and regularization techniques—with one key difference: we are predicting categories instead of numbers.
