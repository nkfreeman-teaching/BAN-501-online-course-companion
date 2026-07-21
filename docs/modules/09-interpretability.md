# Module 9: Model Interpretability & Explainability

## Introduction

We've covered a wide range of modeling techniques: linear models, decision trees, random forests, XGBoost, neural networks, CNNs, and transformers. Some are simple—you can look at coefficients. Others are complex—millions of parameters that no human can comprehend directly. The challenge is that **a model that can't be explained often can't be deployed.**

A bank denies someone a loan. A hospital's AI recommends a treatment. An insurance company sets a premium. In all these cases, people deserve to know why. And in many cases, the law requires it.

This module bridges the gap between model performance and real-world deployment. You'll learn how to explain any model—black box or not—and how to communicate those explanations to stakeholders who don't know (or care) about gradient descent.

Modern tools narrow, but do not eliminate, the tradeoff between interpretability and performance. Train a complex model for maximum performance, then use post-hoc explanation methods (SHAP, LIME, PDP) to attribute predictions to features under stated assumptions — you get useful attributions, not a guarantee that the model has been "fully understood." Intrinsically interpretable models (linear regression, short decision trees) provide explanations directly when regulations require it, and a well-regularized linear model can often match tree-ensemble performance on tabular problems. Section 9.2 covers the specific tools and the assumptions each one rests on; Section 9.4 catalogs where post-hoc explanations can mislead.

---

## Learning Objectives

By the end of this module, you should be able to:

1. **Explain** why model interpretability matters for business and regulatory compliance
2. **Distinguish** between global and local interpretability
3. **Apply** SHAP and LIME to explain model predictions
4. **Create** effective visualizations of model behavior
5. **Communicate** model insights to non-technical stakeholders
6. **Document** models with model cards and limitations

---

## 9.1 Why Interpretability Matters

A model is **interpretable** when a human can understand the reasoning behind its predictions well enough to verify, trust, or challenge them. Interpretability is not just a technical nicety—it is a requirement for responsible deployment in regulated industries and a practical necessity for earning stakeholder trust.

### The Business Case

The best model in the world is worthless if no one trusts it. You could build a fraud detection system with 99% accuracy, but if the compliance team can't explain why it flagged a transaction, they can't defend that decision to regulators. If loan officers can't explain why an application was denied, they can't legally send that denial letter.

### Regulatory Requirements

Multiple regulatory frameworks now require that automated decisions be explainable.

#### GDPR (EU General Data Protection Regulation)

The GDPR grants citizens a "right to explanation" for automated decisions. If a machine makes a decision that significantly affects someone, they can demand to know why. This applies to credit scoring, hiring, insurance, and healthcare. The precise scope of this right is still debated in legal scholarship—some readings limit it to decisions made *solely* by automated means—but in practice, affected companies treat meaningful explanation as a compliance expectation.

#### Fair Lending Laws (US)

The Equal Credit Opportunity Act requires reasons for adverse actions. "Your application was denied because..." is legally required; "the algorithm said no" does not satisfy the law.

#### Healthcare Regulations

The FDA scrutinizes AI medical devices, and clinicians need to understand recommendations before acting on them. Liability concerns compound this: if something goes wrong, the organization must be able to explain why the AI recommended that course of action.

### Building Stakeholder Trust

Business stakeholders typically want to know (i) why the model made a particular prediction, (ii) which factors are most important, (iii) whether they can trust the prediction, and (iv) what would change it. That last question—"what would change it?"—is addressed by **counterfactual explanations**: the smallest change to the input that flips the prediction. For example, "your loan would have been approved if your debt-to-income ratio were 5 points lower." Counterfactual explanations are especially useful for adverse-action notices because they give the affected person a concrete, actionable path forward. The tools in Section 9.2 (SHAP, LIME) answer the "why" question; counterfactuals answer the "what if" question.

Without trust, models will not be adopted because people ignore recommendations, decisions get overridden (defeating the model's purpose), and the value of the ML investment is lost after months of work go unused.

### Debugging and Improving Models

Interpretability helps identify problems such as (i) spurious correlations where the model learned wrong patterns, (ii) data leakage where the model uses information it should not have, (iii) bias in training data where historical biases become encoded in predictions, and (iv) overfitting where the model memorized patterns that will not generalize.

#### The Pneumonia Example

Researchers trained a model to predict pneumonia severity from X-rays. The model performed well—too well. Investigation revealed that the model learned to associate "portable X-ray" equipment markers with low risk. Portable X-rays were used for patients well enough to not need a trip to the radiology department, so the model was predicting equipment type, not disease severity. Without interpretability tools, this would have been deployed and potentially harmed patients.

#### Catching Spurious Correlations

For consequential models, investigating what the model learned is a professional responsibility. Use the explanation tools introduced in Section 9.2 (feature importance, partial dependence, and attribution methods) in your standard workflow, and show top features to domain experts—a radiologist would immediately question equipment-type markers as a predictor of disease severity. Ask: "What shortcuts could the model have taken?" Test on out-of-distribution data. The investigation level should match the stakes—product recommendations warrant less scrutiny than medical diagnosis.

### Discovering Bias

ML models can encode and amplify biases through (i) historical bias in training data, (ii) proxy variables for protected attributes, and (iii) feedback loops. Interpretability reveals which features drive predictions for different groups, whether protected attributes have indirect influence, and whether unexpected correlations might indicate bias.

Consider a hiring model that heavily weights ZIP code. ZIP code correlates with race and income, so the model might be making discriminatory decisions even without explicit race features. This is proxy discrimination—often unethical and sometimes illegal.

The practical workflow for catching this is: compute SHAP values (introduced in Section 9.2), then disaggregate them by demographic group. If protected-group members systematically receive large negative SHAP contributions from a proxy variable like ZIP code, that is a signal worth investigating. Module 10 extends this analysis with formal fairness metrics and bias mitigation techniques.

### Communicating Model Insights

Interpretability tools are only valuable if their findings reach the people who make decisions. This subsection covers how to translate technical model explanations into language that business stakeholders, regulators, and customers can act on.

#### Executive Summaries

An effective executive summary has five parts: (i) the business question (what you were predicting and why), (ii) the key finding (the main takeaway), (iii) the top factors driving predictions (three to five factors at most), (iv) the confidence level (how reliable the model is and any limitations), and (v) the recommendation (what the organization should do). No code, no jargon—just business value.

The following example illustrates how to present a churn model's findings without technical jargon.

```
EXECUTIVE SUMMARY: Customer Churn Model

Business Question: Which customers are likely to cancel
their subscription in the next 90 days?

Key Finding: We can identify 75% of churning customers
before they leave, with 80% precision—meaning 4 out of 5
customers we flag will actually churn.

Top Factors Driving Churn Risk:
1. Support tickets in last 30 days (more tickets = higher risk)
2. Days since last login (longer gap = higher risk)
3. Contract type (monthly contracts 3x more likely to churn)

Confidence: Model validated on 6 months of holdout data.
Limitation: Works best for customers with 90+ days of history.

Recommendation: Prioritize retention outreach to customers
with churn probability > 70%. Expected ROI: $2.50 saved
per $1 spent on retention.
```

Notice that this summary makes no mention of random forests, SHAP, or cross-validation. Every sentence focuses on business-relevant insights that a non-technical reader can act on.

!!! note "Note"

    The example below references SHAP values—a method for attributing each feature's contribution to a prediction. SHAP is explained fully in Section 9.2. For now, treat each numeric contribution as "how many percentage points did this feature push the churn probability up or down?"

!!! example "Numerical Example: From SHAP to Business English"

    ```python
    # Raw SHAP output for a high-risk customer:
    shap_output = {
        'base_value': 0.25,      # Average churn rate (25%)
        'prediction': 0.78,       # This customer (78%)
        'contributions': {
            'support_tickets_30d': +0.22,
            'days_since_login': +0.18,
            'contract_type': +0.12,
            'tenure_months': +0.08,
            'satisfaction_score': -0.05,
            'total_spend': -0.02,
        }
    }
    ```

    **Translated to business language:**
    ```
    Customer Churn Risk Assessment
    --------------------------------
    Risk Level: HIGH (78% likelihood of churning)
    Baseline: Average customer has 25% churn risk

    Top factors INCREASING risk:
    1. SUPPORT ISSUES (+22 points)
       Filed 5 tickets in 30 days—indicates frustration

    2. ENGAGEMENT DROP (+18 points)
       Last login 45 days ago—stopped using product

    3. CONTRACT FLEXIBILITY (+12 points)
       Monthly contract—easy to cancel anytime

    Mitigating factors:
    - Satisfaction score 6/10 (better than churners)
    - Recent spending $250 (some investment)

    RECOMMENDED ACTIONS:
    1. Customer success outreach within 24 hours
    2. Resolve open support tickets immediately
    3. Offer annual contract incentive
    ```

    **Interpretation:** The translation removes all technical jargon (no "SHAP values," "base value," or decimals). It groups factors into "increasing risk" vs "mitigating," uses percentage points instead of raw values, and ends with actionable recommendations.

    *Source: `computations/module9_examples.py` — `demo_shap_to_business()`*


#### Model Cards

**Model cards** are documentation standards for ML models, introduced by Google. They provide a structured way to communicate a model's capabilities, limitations, and appropriate use cases to both technical and non-technical audiences.

A model card includes seven sections: (i) model details such as type, version, date, and owner; (ii) intended use, specifying what the model is and is not for; (iii) relevant factors such as demographics; (iv) metrics showing performance overall and by subgroup; (v) training data describing what data was used; (vi) limitations documenting when the model fails; and (vii) ethical considerations covering potential harms and biases.

A minimal example for the churn model used throughout this module:

```
MODEL CARD: Customer Churn Predictor v1.2
=========================================
Model details:    XGBoost classifier | Trained 2025-11 | Owner: Analytics Team
Intended use:     Flag customers for proactive retention outreach (internal only)
NOT intended for: Automated cancellation, pricing, or legal decisions

Metrics (holdout set, 2025-Q3):
  Overall AUC: 0.84 | Precision: 0.80 | Recall: 0.75
  Subgroup parity: AUC within 0.03 across age groups 18-65

Training data:    18 months of subscription events (2024-01 to 2025-06)
                  ~120,000 customers; excludes <90-day accounts

Known limitations:
  - Accuracy degrades for customers with <90 days of history
  - Does not account for seasonal churn patterns
  - Recalibrate quarterly as product offerings change

Ethical considerations:
  - ZIP code excluded after proxy-discrimination review
  - SHAP audit showed no disparate impact by income quintile
```

The model card is a living document—update it when the model is retrained or when new limitations are discovered.

#### Individual Prediction Explanations

For customer-facing explanations, use natural language, focus on the top two or three factors, avoid technical jargon, and provide actionable insights.

When someone is denied credit, they're legally entitled to reasons:

```
Your loan application was declined. The main factors were:

1. Your debt-to-income ratio is above our threshold
2. Your credit history is shorter than we typically require
3. Recent credit inquiries suggest high credit-seeking behavior

Steps you can take to improve your chances:
- Pay down existing debt to lower your debt-to-income ratio
- Wait 6 months to build more credit history
- Avoid applying for new credit in the near term
```

This notice is specific, actionable, and free of jargon. The customer knows what happened, why it happened, and what they can do about it.

---

## 9.2 Interpretation Techniques

This section introduces the primary tools for understanding what a model has learned and why it makes specific predictions.

### Global vs Local Interpretability

**Global interpretability** means understanding overall model behavior: which features are generally important and what patterns the model uses. **Local interpretability** means understanding individual predictions: why was this particular customer predicted to churn, and what would change that decision.

Both matter. Executives want global insights: "What drives churn?" Customer service needs local explanations: "Why was this specific customer flagged?"

Think of global interpretability as understanding what a disease does to a *population*—epidemiologists identify which risk factors most commonly drive hospitalizations across thousands of patients. Local interpretability is a clinician diagnosing one patient sitting in front of them: their specific lab values, their specific history. The epidemiologist's population-level findings inform clinical practice, but the clinician still needs a patient-specific explanation to act. Both perspectives are necessary, and they answer different questions.

### Permutation Importance

The idea is straightforward. First, train a model and measure its baseline performance. Next, shuffle one feature's values to break its signal, then measure the performance drop. A larger drop indicates a more important feature. This works because if a feature is important, breaking its signal hurts predictions.

An analogy helps here: imagine testing how much a basketball player relies on their vision. Blindfold them and see how much worse they play. If performance drops dramatically, vision was important. If they still play well (maybe they are great at listening for the ball), vision was not crucial. Permutation importance "blindfolds" each feature one at a time and measures how much the model's performance degrades.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)

# Sort by importance
for i in result.importances_mean.argsort()[::-1]:
    print(f"{feature_names[i]}: {result.importances_mean[i]:.3f}")
```

Permutation importance works with any model (it is model-agnostic) and uses held-out test data, making it reliable. However, it can be slow when there are many features and misleading with correlated features, since shuffling one correlated feature is compensated by another.

**Comparison with built-in `.feature_importances_`:** If you used random forests or gradient boosted trees in Module 4, you may have already accessed `model.feature_importances_`. That attribute reports *impurity-based* importance: how much each feature reduced Gini impurity (or MSE) across all splits in all trees, averaged over the ensemble. Permutation importance and impurity-based importance often agree on the most important features, but differ in meaningful ways:

| | Impurity-based (`.feature_importances_`) | Permutation importance |
|---|---|---|
| Data used | Training data only | Held-out test data |
| Bias toward | High-cardinality features (many split points) | None—purely based on prediction change |
| Works with | Tree models only | Any model |
| Speed | Instant (computed during training) | Slower (requires repeated model evaluation) |

For a final analysis, permutation importance on test data is generally preferred because it reflects held-out generalization, not in-sample fit. Use `.feature_importances_` for fast exploration during model development.

!!! example "Numerical Example: Permutation Importance Step by Step"

    ```python
    # Dataset: x1 (strong signal), x2 (correlated with x1, moderate true effect),
    #          x3 (pure noise)
    # Target depends on: 2*x1 + 0.5*x2, not on x3

    # Train Random Forest and measure baseline accuracy
    baseline_accuracy = 0.647  # 64.7%

    # Shuffle each feature and measure performance drop
    # x1 (strong): shuffle → accuracy drops to 40.0%  → importance = +24.7%
    # x2 (correlated): shuffle → accuracy rises to 70.7% → importance = -6.0%
    # x3 (noise):  shuffle → accuracy rises to 66.0% → importance = -1.3%

    importance_x1 = 0.647 - 0.400  # = +0.247 (large positive: x1 is important)
    importance_x2 = 0.647 - 0.707  # = -0.060 (negative: shuffling actually helped)
    importance_x3 = 0.647 - 0.660  # = -0.013 (near-zero: model ignores x3)
    ```

    **Output:**

    ```
    x1 (strong):      Importance = +24.7%  ← critical feature
    x2 (correlated):  Importance =  -6.0%  ← negative (see explanation)
    x3 (noise):       Importance =  -1.3%  ← near-zero as expected
    ```

    **Interpretation:** Shuffling x1 destroys the main signal, causing a 24.7% accuracy drop. x2's importance is *negative*—this happens because x2 is correlated with x1 (r ≈ 0.96). When the model was trained, it used both x1 and x2 as proxies for the same signal. On this finite test set, shuffling x2 sometimes *reduces* redundant confounding, giving the model a cleaner signal from x1 alone. Negative permutation importance means the feature was not helping (and may have been adding noise at the margin on this sample). Treat values near zero, whether slightly positive or slightly negative, as "not important." x3's near-zero value confirms the model correctly ignored it.

    *Source: `computations/module9_examples.py` — `demo_permutation_importance()`*


### Partial Dependence Plots (PDP)

PDPs show the average effect of a feature on predictions. The procedure works as follows. For each value of feature X (e.g., age from 20 to 80), set all samples to that value, average the predictions, and then plot the average prediction against the feature value.

Imagine a dashboard with a slider for each feature. When you drag the "age" slider from 20 to 80, the PDP shows how the *average* prediction changes. It answers the question: "If I could set everyone's age to 50, what would the average prediction be?" This isolates the marginal effect of that feature, averaging over all the other features in the data.

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features=['age', 'income']
)
```

When reading a PDP, an upward slope means higher feature values lead to higher predictions, a flat line indicates little average effect, and a non-linear shape reveals a complex relationship.

One limitation is that PDPs assume feature independence, which can produce impossible combinations (20-year-olds with $500K income). Individual Conditional Expectation (ICE) plots address this limitation by showing one line per observation, revealing heterogeneous effects that PDP averages away. Where a PDP shows one smooth "average" curve, an ICE plot shows a bundle of curves—one per row in the dataset. If those curves fan out or cross, the feature's effect differs across subgroups, and the PDP average is hiding meaningful variation.

```python
# ICE plot: kind="individual" shows one line per sample
# kind="both" overlays the PDP mean on top
PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features=['age'],
    kind='both',        # shows ICE lines + PDP average
    subsample=100,      # plot 100 random samples for clarity
    alpha=0.3,          # make individual lines semi-transparent
)
```

!!! example "Numerical Example: Building a Partial Dependence Plot"

    ```python
    # Churn prediction model with age, income, tenure
    # PDP for 'age': What happens to average churn as we vary age?

    # For each age value, set ALL customers to that age
    # and average the predictions
    age_values = [25, 35, 45, 55, 65]
    avg_churn_probs = []

    for age in age_values:
        X_modified = X.copy()
        X_modified[:, 0] = age  # Everyone is now this age (age is column 0)
        avg_prob = model.predict_proba(X_modified)[:, 1].mean()
        avg_churn_probs.append(avg_prob)
    ```

    **Output:**

    ```
    Age Value    Avg Churn Prob
    --------------------------------
          25              71.0%
          35              64.0%
          45              53.9%
          55              47.2%
          65              28.7%
    ```

    **Interpretation:** The PDP shows a clear downward trend—as age increases, average churn probability decreases. This is the "what-if slider": drag age from 25→65 and watch the average prediction drop from 71%→29%.

    *Source: `computations/module9_examples.py` — `demo_partial_dependence()`*


### SHAP (SHapley Additive exPlanations)

SHAP is founded on Shapley values from game theory, which fairly distribute "credit" among players. Applied to ML, the question becomes: how much did each feature contribute to pushing this prediction away from the average?

SHAP has three mathematically proven properties: (i) local accuracy, meaning SHAP values sum to the prediction minus the baseline; (ii) consistency, meaning that if a feature's marginal contribution increases in every possible coalition, its Shapley value does not decrease; and (iii) missingness, meaning unused features get zero attribution.

When interpreting SHAP values, a positive value means the feature pushed the prediction higher, a negative value means it pushed the prediction lower, and the magnitude indicates the strength of the effect.

Before the formula, consider a concrete example. Three data scientists (A, B, C) work on a project. Alone, A generates $50k, B generates $40k, C generates $20k. But together, A+B generate $120k (synergy), and all three generate $150k. How do you fairly split the $150k? Shapley values average each person's marginal contribution across all possible orderings they could have joined. Player A's Shapley value is $66.7k—they get more because they add value in every combination. This is the same math SHAP uses: features are "players" and the prediction is the "payoff."

#### The Shapley Formula

$$\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{j\}) - f(S)]$$

**In plain English:** Consider all possible subsets of features. For each subset, measure how much adding feature $j$ changes the prediction. Average these contributions with weights $\frac{|S|!(d - |S| - 1)!}{d!}$ where $|S|$ is the coalition size and $d$ is the total number of features, ensuring fairness.

Exact Shapley computation requires evaluating all $2^d$ feature subsets, which is exponential in $d$. For a model with 20 features that is already $2^{20} \approx 1{,}000{,}000$ subsets; with 30 features it is over a billion. Practical variants are therefore necessary. TreeSHAP exploits tree structure for polynomial-time exact values—use it for random forests, XGBoost, and LightGBM. DeepSHAP uses gradient approximations for neural networks. KernelSHAP handles arbitrary models but is slow. This often influences model choice: if both interpretability and speed are required, tree-based models with TreeSHAP become attractive.

!!! example "Numerical Example: Shapley Values in a Simple Game"

    ```python
    # Three data scientists (A, B, C) work on a project
    # Coalition payoffs (in $1000s):
    payoffs = {
        '∅': 0,      '{A}': 50,   '{B}': 40,   '{C}': 20,
        '{A,B}': 120,  # A+B have synergy!
        '{A,C}': 80,   '{B,C}': 70,
        '{A,B,C}': 150  # Grand coalition
    }

    # For each player, average marginal contribution across
    # all orderings they could join:

    # Player A joins: ∅→+50, {B}→+80, {C}→+60, {B,C}→+80
    # Weights follow the Shapley formula: |S|!(d - |S| - 1)!/d!
    # where |S| is the coalition size and d is the total number of features
    shapley_A = weighted_average([50, 80, 60, 80])  # = $66.7k

    # Player B joins: ∅→+40, {A}→+70, {C}→+50, {A,C}→+70
    shapley_B = weighted_average([40, 70, 50, 70])  # = $56.7k

    # Player C joins: ∅→+20, {A}→+30, {B}→+30, {A,B}→+30
    shapley_C = weighted_average([20, 30, 30, 30])  # = $26.7k
    ```

    **Output:**

    ```
    Final allocation:
      A: $66.7k (highest—adds value everywhere)
      B: $56.7k (good synergy with A)
      C: $26.7k (consistent but lower contribution)
      Total: $150.0k (= grand coalition value)
    ```

    **Interpretation:** Shapley values are the *only* allocation that is fair, efficient, and additive. In ML, features are "players" and the prediction is the "payoff"—SHAP tells us how much each feature contributed to pushing the prediction away from the baseline.

    *Source: `computations/module9_examples.py` — `demo_shapley_game()`*


### SHAP in Practice

The following code demonstrates how to compute and visualize SHAP values for tree-based models.

```python
import shap

# For tree-based models (fast!)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# NOTE: For multi-class classification, shap_values is a list of arrays—
# one array per class. To explain predictions for class k:
#   shap_values[k]  → shape (n_samples, d_features)
# For binary classification, some versions return a single array (for class 1)
# and others return a list of two. Always check: type(shap_values)

# Summary plot (global view)
shap.summary_plot(shap_values, X_test)

# Force plot (single prediction)
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0]
)

# Waterfall plot (detailed breakdown)
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=feature_names
))
```

!!! example "Numerical Example: SHAP Values Sum to Prediction"

    ```python
    # For a single test instance:
    baseline = 0.517  # Average training prediction
    prediction = 0.913  # This instance's prediction
    difference = prediction - baseline  # = +0.396 to explain

    # SHAP breaks down the difference by feature:
    shap_values = {
        'feature_1': +0.328,  # Pushed prediction UP
        'feature_2': -0.301,  # Pushed prediction DOWN
        'feature_3': -0.086,  # Pushed prediction DOWN
        'feature_4': +0.455,  # Pushed prediction UP
    }
    # Sum: 0.328 + (-0.301) + (-0.086) + 0.455 = +0.396
    ```

    **Output:**

    ```
    Component             Value
    ----------------------------------------
    Base value            0.517
    feature_1         +   0.328
    feature_2         -   0.301
    feature_3         -   0.086
    feature_4         +   0.455
    ----------------------------------------
    Prediction            0.913  ✓
    ```

    **Interpretation:** SHAP's *local accuracy* (efficiency) property guarantees that for the displayed instance, `base_value + Σ(SHAP values) = prediction` exactly — that's the arithmetic above. What this does *not* mean is that the explanation has captured the model's true reasoning: SHAP values depend on the choice of background dataset (what counts as a feature being "absent"), the output space (raw scores vs. probabilities), and how correlated features are handled when sampling coalitions. TreeSHAP returns exact Shapley values for tree models; KernelSHAP and many DeepSHAP variants return approximations whose quality depends on the number of samples and the background size. So the right reading here is: *given these design choices*, Feature 4 pushed the prediction up most (+0.455), while features 2 and 3 pushed it down. See the misconception table later in this module for more on where SHAP exactness applies.

    *Source: `computations/module9_examples.py` — `demo_shap_sum_to_prediction()`*


### SHAP Visualizations

The **summary plot** provides a global view of importance with direction: each dot represents one sample, the x-axis shows the SHAP value, and color indicates the feature value (red for high, blue for low). The **force plot** breaks down a single prediction by starting from the baseline, showing features pushing the prediction up and down, and ending at the actual prediction. The **waterfall plot** offers a step-by-step breakdown from baseline to prediction, where each bar represents one feature's contribution. The **dependence plot** resembles a PDP but shows actual data points, and you can color by another feature to reveal interactions.

#### How to Read a SHAP Summary Plot Step by Step

First, look at the feature order: features at the top are most important, indicated by the widest spread of dots. Next, find the red dots (high feature value) and blue dots (low feature value). Then observe where red clusters fall: if red dots cluster on the right, high values increase predictions. Similarly, if blue dots cluster on the right, low values increase predictions. Finally, check the spread: a wide horizontal spread indicates strong impact, while a tight cluster at zero indicates weak impact. For example, if "support_tickets" shows red dots clustered on the right, it means customers with many support tickets (high value = red) have higher churn predictions (positive SHAP = right).

!!! example "Numerical Example: Reading a SHAP Summary Plot"

    ```python
    # Churn prediction model - SHAP summary patterns:
    # (Each feature shows where high/low values cluster)

    feature_patterns = {
        'support_tickets': 'Red dots RIGHT → high tickets = higher churn',
        'months_customer': 'Red dots LEFT → long tenure = lower churn',
        'income':          'Red dots LEFT → higher income = lower churn',
        'age':             'Dots spread evenly → weak/noisy effect',
    }
    ```

    **Output:**

    ```
    Feature            High values (red)    Pattern
    ----------------------------------------------------------------
    support_tickets    → cluster RIGHT      More tickets = higher churn
    months_customer    → cluster LEFT       Longer tenure = lower churn
    income             → cluster LEFT       Higher income = lower churn
    age                → spread across      Age effect is noisy/weak
    ```

    **Interpretation:** The summary plot tells a complete story. Support tickets is the top driver (widest spread), with high values strongly increasing churn risk. Tenure is protective—long-term customers (red) have negative SHAP values (left). Age shows no clear pattern, suggesting it's not a reliable predictor.

    *Source: `computations/module9_examples.py` — `demo_shap_summary_interpretation()`*


### LIME (Local Interpretable Model-agnostic Explanations)

The core idea of LIME is to approximate a complex model locally with a simple one. A complex model's decision boundary might be wildly curved and twisted at the global level—impossible to describe simply. But if you zoom in to a tiny neighborhood around one point, even the most complex curve looks approximately straight. LIME zooms into that local neighborhood, fits a simple linear model that captures the local behavior, and interprets *that* simple model. The explanation is only valid in that neighborhood—move to a different point and you would get a different local approximation.

The procedure works as follows. First, generate perturbed samples around the instance. Next, obtain the complex model's predictions for those samples. Then fit a simple linear model weighted by distance, and interpret that simple model.

The weighting step uses a **kernel function**: perturbed samples that are close to the original instance are weighted heavily, while distant samples are weighted lightly. The **kernel width** controls how large the "neighborhood" is—a narrow kernel creates a very local explanation (valid only for points almost identical to the instance), while a wide kernel creates a broader, smoother explanation that may miss fine-grained local behavior. In practice, LIME's default kernel width works well for most tabular data; if explanations seem unstable across similar instances, try narrowing the kernel.

*"In the neighborhood of this prediction, what does the model behave like?"*

```python
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=['Not Churn', 'Churn'],
    mode='classification'
)

explanation = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)

explanation.show_in_notebook()
```

!!! example "Numerical Example: LIME Perturbation in Action"

    ```python
    # Original instance: [0.80, 0.90, 0.10] → P(class=1) = 100%
    # LIME generates nearby perturbed samples and gets predictions:

    perturbed_samples = [
        [0.95, 0.86, 0.29],  # → 94%
        [1.26, 0.83, 0.03],  # → 100%
        [0.63, 0.60, 0.19],  # → 98%
        [0.53, 0.48, 0.54],  # → 100%
        # ... more samples ...
    ]

    # Fit weighted linear model (closer samples get more weight)
    # Local linear approximation coefficients:
    local_coefs = {
        'feature_A': +0.041,  # increases prediction locally
        'feature_B': -0.057,  # decreases prediction locally
        'feature_C': +0.015,  # slight positive effect
    }
    ```

    **Output:**

    ```
    Local linear approximation (LIME):
      feature_A: +0.041 (increases prediction)
      feature_B: -0.057 (decreases prediction)
      feature_C: +0.015 (increases prediction)
      Intercept: 0.977
    ```

    **Interpretation:** In the *neighborhood* of this specific instance, the model behaves approximately linearly. Feature A has a positive local effect, while feature B has a negative effect. This explanation is only valid nearby—a different instance might have completely different local behavior.

    *Source: `computations/module9_examples.py` — `demo_lime_perturbation()`*


### SHAP vs LIME

The two methods differ along several dimensions.

| Aspect | SHAP | LIME |
|--------|------|------|
| Foundation | Game theory | Local approximation |
| Consistency | Mathematically guaranteed | Not guaranteed |
| Speed | Fast with TreeSHAP | Generally slower |
| Global view | Yes (aggregate) | Limited |
| Stability | High—same instance always gets same values | Lower—randomness in perturbation sampling can shift values across runs |
| Data modality | Tabular; image/text variants exist | Tabular, text, and image natively supported |

Both are valuable in practice. SHAP has stronger theoretical foundations and provides consistent global views, while LIME can be more intuitive for explaining individual predictions to non-technical audiences and handles text and image data more naturally out of the box.

!!! example "Numerical Example: SHAP vs Permutation Importance"

    ```python
    # Dataset with correlated features:
    # x1: True predictor (target depends on x1)
    # x2: Correlated with x1 (r ≈ 0.96) but not directly causal
    # x3: Independent noise

    # Correlation matrix:
    #         x1      x2      x3
    # x1    1.00    0.96   -0.06
    # x2    0.96    1.00   -0.03
    # x3   -0.06   -0.03    1.00
    ```

    **Output:**

    ```
    Feature              Permutation     SHAP (mean |val|)
    -------------------------------------------------------
    x1 (causal)                0.216              0.350
    x2 (correlated)            0.016              0.250
    x3 (noise)                 0.005              0.020
    ```

    **Interpretation:** Permutation importance shows x2 as unimportant (0.016) because when x2 is shuffled, x1 still carries the signal. SHAP distributes credit between correlated features, giving x2 a meaningful value (0.250). Neither is "wrong"—they answer different questions:
    - Permutation: "What if we removed this feature?"
    - SHAP: "How much did each feature contribute to predictions?"

    *Source: `computations/module9_examples.py` — `demo_shap_vs_permutation()`*


### Attention Visualization for Transformers

For transformer-based models (Module 8), attention weights are sometimes described as a built-in interpretability mechanism, but this characterization requires care. Attention weights show how the model *routes* information between tokens during computation—they do not directly measure which tokens *caused* the prediction. Think of attention as showing the model's information-flow pattern, not its explanation. With that framing, attention visualization is still useful: it can surface plausible linguistic relationships, reveal unexpected patterns worth investigating, and serve as a starting point for deeper analysis with SHAP or LIME. Each attention head produces a matrix showing how strongly each token attends to every other token when the model processes the sequence.

#### How It Works

Given an input sequence, extract the attention weights from one or more layers. Each weight $a_{ij}$ represents how much token $i$ attended to token $j$. High weights indicate that the model considered token $j$ important when processing token $i$.

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    output_attentions=True,
)

inputs = tokenizer("The cat sat on the mat", return_tensors="pt")
outputs = model(**inputs)

# outputs.attentions is a tuple of (batch, heads, seq_len, seq_len) per layer
# Layer 0, Head 0 attention matrix:
attention = outputs.attentions[0][0, 0].detach().numpy()
```

#### Interpreting Attention

Different heads specialize in different relationships. One head might capture syntactic dependencies (verbs attending to their subjects), another might handle positional patterns (each token attending to its neighbor), and another might focus on semantic relationships (pronouns attending to their referents). Averaging across all heads often washes out these patterns—examining individual heads is more informative.

#### Caveats

Attention weights are not the same as feature importance. A token can receive high attention without being causally important to the prediction. Attention shows where the model *looks*, not necessarily what *drives* the output. Use attention visualization as a hypothesis-generating tool alongside SHAP or LIME, not as a standalone explanation method.

### Important Caveats

Feature importance is not the same as causation. When SHAP says "age is the most important feature," it means age most influences predictions. It does not mean age *causes* the outcome. A model might use age as a strong predictor of churn, but that does not mean getting older causes churn. There might be a confounder.

A practical heuristic: ask yourself whether the proposed intervention makes sense. If a feature is "important," would *changing that feature* change the outcome? If yes, causation is plausible and intervention might work. If no—like banning umbrella sales to increase outdoor event attendance—the feature is a symptom, not a cause. Domain experts are your best check: show them the top features and ask "could someone deliberately change this to change the outcome?" If they say "no, that's just a proxy for X," trace the chain back to X.

#### The Umbrella Sales Example

A model predicting outdoor event attendance might show "umbrella sales" as the most important feature. But buying umbrellas does not *cause* low attendance—both are caused by rain (a confounder). If you tried to increase attendance by banning umbrella sales, you would fail. The model correctly learned that umbrella sales predict attendance, but the *causal* intervention point is weather, not umbrellas. This is why domain expertise matters: a meteorologist would immediately spot the spurious relationship. The takeaway: do not confuse prediction importance with causal importance.

### Common Misconceptions

The following table collects the most frequent misunderstandings that arise across both the business and technical dimensions of interpretability.

| Misconception | Reality |
|--------------|---------|
| "Accuracy is all that matters" | Without interpretability, you can't trust, debug, or deploy responsibly |
| "Deep learning can never be interpreted" | Many techniques exist (SHAP, attention, feature visualization) |
| "Simple models are always more interpretable" | A 100-feature linear model isn't necessarily interpretable |
| "Feature importance = causation" | Importance shows prediction influence, not causal effect |
| "SHAP values are always exact" | KernelSHAP is approximate; TreeSHAP is exact only for trees |
| "High attention = high importance" | Attention shows information routing, not prediction causation |
| "LIME explanations are stable" | Perturbation randomness means LIME can give different results for the same instance across runs |

---

## Reflection Questions

1. A bank's loan approval model has 95% accuracy but can't explain decisions. Why might regulators reject it?

2. You discover your hiring model relies heavily on ZIP code. Why is this concerning?

3. SHAP shows 'age' has highest importance, but PDP shows a flat relationship. How is this possible? *(Hint: SHAP and PDP measure different things. PDP shows the average marginal effect of age alone, averaging over all other features. SHAP measures each feature's contribution to each individual prediction, accounting for interaction effects with other features. Age can have large individual SHAP values—positive for some customers and negative for others—that cancel out when averaged, producing a flat PDP. Look at the spread of SHAP values, not just the mean.)*

4. You need to explain a loan denial to a customer. Would you use SHAP or LIME? Why?

5. A stakeholder asks "which feature is most important?" What clarifying questions should you ask?

6. Your model uses 50 features. How do you explain it to a CEO in 5 minutes?

7. A customer asks why their insurance premium increased. How do you respond without technical jargon?

---

## Practice Problems

1. Given SHAP values for a prediction, write the explanation in plain English

2. Identify potential problems from PDP shapes (non-monotonic, discontinuous)

3. Choose appropriate explanation technique for different scenarios

4. Write an adverse action notice from model output

5. Create a model card outline for a fraud detection system

---

## Chapter Summary

Model interpretability is not optional—it is required for regulatory compliance, stakeholder trust, and effective debugging. Interpretability operates at two levels: global techniques reveal overall patterns and feature importance, while local techniques explain individual predictions. Among the available tools, SHAP provides mathematically principled feature attribution grounded in game theory, and LIME approximates complex models locally with simple, interpretable surrogates. The value of these tools depends on communication: executive summaries translate technical findings into business language that stakeholders can act on, and model cards standardize documentation of a model's capabilities, limitations, and appropriate use cases.

---

## What's Next

In Module 10, we tackle **Ethics, Fairness & Deployment**, covering bias in ML systems and how it arises, fairness metrics and definitions, bias mitigation techniques, responsible AI practices, and model deployment considerations. Interpretability is the foundation for fairness analysis—you can't assess whether a model is fair if you can't understand what it's doing.
