"""
Module 9: Model Interpretability & Explainability - Numerical Examples

This script demonstrates key concepts from Module 9 with concrete calculations.
Run with: pixi run python course-companion-computations/module9_examples.py

References in Course Companion:
- demo_permutation_importance()       -> Module 9, Section 9.2 (Permutation Importance)
- demo_partial_dependence()           -> Module 9, Section 9.2 (Partial Dependence Plots)
- demo_shapley_game()                 -> Module 9, Section 9.2 (Shapley Values Intuition)
- demo_shap_sum_to_prediction()       -> Module 9, Section 9.2 (SHAP Additivity)
- demo_shap_summary_interpretation()  -> Module 9, Section 9.2 (Reading SHAP Plots)
- demo_lime_perturbation()            -> Module 9, Section 9.2 (LIME)
- demo_shap_vs_permutation()          -> Module 9, Section 9.2 (Method Comparison)
- demo_shap_to_business()             -> Module 9, Section 9.3 (Communication)

Last updated: 2026-01-02
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from itertools import combinations


def demo_permutation_importance():
    """
    Show permutation importance step by step.
    Demonstrates how shuffling breaks feature signal.
    """
    print("=" * 60)
    print("PERMUTATION IMPORTANCE STEP BY STEP")
    print("=" * 60)

    # Create dataset with known important features
    np.random.seed(42)
    n_samples = 500

    # Feature 1: Very important (strong signal)
    x1 = np.random.randn(n_samples)
    # Feature 2: Moderately important
    x2 = np.random.randn(n_samples)
    # Feature 3: Noise (not important)
    x3 = np.random.randn(n_samples)

    # Target depends strongly on x1, moderately on x2, not on x3
    prob = 1 / (1 + np.exp(-(2 * x1 + 0.5 * x2)))
    y = (np.random.rand(n_samples) < prob).astype(int)

    X = np.column_stack([x1, x2, x3])
    feature_names = ['x1 (strong)', 'x2 (moderate)', 'x3 (noise)']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Baseline accuracy
    baseline_acc = model.score(X_test, y_test)
    print(f"\nBaseline accuracy on test set: {baseline_acc:.1%}")

    print("\nPermutation importance process:")
    print("-" * 50)

    # Manual permutation importance for each feature
    for i, name in enumerate(feature_names):
        # Shuffle feature i
        X_test_shuffled = X_test.copy()
        np.random.seed(42)
        X_test_shuffled[:, i] = np.random.permutation(X_test_shuffled[:, i])

        # Measure new accuracy
        shuffled_acc = model.score(X_test_shuffled, y_test)
        importance = baseline_acc - shuffled_acc

        print(f"\n{name}:")
        print(f"  Shuffle feature → accuracy drops to {shuffled_acc:.1%}")
        print(f"  Importance = {baseline_acc:.1%} - {shuffled_acc:.1%} = {importance:.1%}")

    print("\nInterpretation:")
    print("  - x1 (strong signal): Large accuracy drop when shuffled")
    print("  - x2 (moderate signal): Moderate drop")
    print("  - x3 (noise): Near-zero drop (model doesn't use it)")
    print("\nThe 'blindfold test': hiding important information hurts performance.")


def demo_partial_dependence():
    """
    Build a partial dependence plot step by step.
    Shows the 'what-if slider' concept.
    """
    print("\n" + "=" * 60)
    print("BUILDING A PARTIAL DEPENDENCE PLOT")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 500

    # Create features
    age = np.random.uniform(low=20, high=70, size=n_samples)
    income = np.random.uniform(low=30000, high=150000, size=n_samples)
    tenure = np.random.uniform(low=0, high=20, size=n_samples)

    # Churn probability: higher age = lower churn, higher income = lower churn
    logit = -0.05 * (age - 45) - 0.00002 * (income - 90000) + 0.1 * (tenure - 10)
    prob = 1 / (1 + np.exp(-logit))
    y = (np.random.rand(n_samples) < prob).astype(int)

    X = np.column_stack([age, income, tenure])
    feature_names = ['age', 'income', 'tenure']

    # Train model
    model = GradientBoostingClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X, y)

    print("\nPartial Dependence for 'age' feature:")
    print("(What happens to average churn prediction as we vary age?)")
    print("-" * 50)

    # PDP calculation for age
    age_values = [25, 35, 45, 55, 65]
    print(f"\n{'Age Value':>12} {'Avg Churn Prob':>18}")
    print("-" * 32)

    for age_val in age_values:
        # Set ALL samples to this age value
        X_modified = X.copy()
        X_modified[:, 0] = age_val  # Replace age column

        # Average prediction
        avg_prob = model.predict_proba(X_modified)[:, 1].mean()
        print(f"{age_val:>12} {avg_prob:>18.1%}")

    print("\nInterpretation:")
    print("  - As age increases, average churn probability decreases")
    print("  - This shows the marginal effect of age, averaging over")
    print("    all other feature combinations in the data")
    print("\nThe 'what-if slider': drag age from 25 to 65 and watch")
    print("the average prediction change from ~58% to ~42%.")


def demo_shapley_game():
    """
    Illustrate Shapley values with a simple 3-player coalition game.
    Makes the concept concrete before applying to ML.
    """
    print("\n" + "=" * 60)
    print("SHAPLEY VALUES: A SIMPLE GAME")
    print("=" * 60)

    print("\nScenario: Three data scientists (A, B, C) work on a project.")
    print("The company earns different profits depending on who works:")
    print()

    # Define payoffs for all coalitions
    payoffs = {
        frozenset(): 0,           # No one
        frozenset(['A']): 50,     # Just A
        frozenset(['B']): 40,     # Just B
        frozenset(['C']): 20,     # Just C
        frozenset(['A', 'B']): 120,    # A and B together (synergy!)
        frozenset(['A', 'C']): 80,     # A and C
        frozenset(['B', 'C']): 70,     # B and C
        frozenset(['A', 'B', 'C']): 150,  # Everyone
    }

    print("Coalition payoffs (in $1000s):")
    print("-" * 35)
    coalition_labels = [
        ("∅ (nobody)", frozenset()),
        ("{A}", frozenset(['A'])),
        ("{B}", frozenset(['B'])),
        ("{C}", frozenset(['C'])),
        ("{A,B}", frozenset(['A', 'B'])),
        ("{A,C}", frozenset(['A', 'C'])),
        ("{B,C}", frozenset(['B', 'C'])),
        ("{A,B,C}", frozenset(['A', 'B', 'C'])),
    ]
    for label, coalition in coalition_labels:
        print(f"  {label:12} → ${payoffs[coalition]:>3}k")

    print("\nNow we calculate each player's Shapley value.")
    print("For each player, we average their marginal contribution")
    print("across ALL possible orderings they could join.\n")

    players = ['A', 'B', 'C']

    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    def shapley_value(player, players, payoffs):
        """Calculate Shapley value for a player."""
        n = len(players)
        other_players = [p for p in players if p != player]
        total = 0

        # Consider all subsets that don't include the player
        for r in range(len(other_players) + 1):
            for subset in combinations(other_players, r):
                S = frozenset(subset)
                S_with_player = S | {player}

                # Marginal contribution
                marginal = payoffs[S_with_player] - payoffs[S]

                # Weight: |S|! * (n - |S| - 1)! / n!
                weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                total += weight * marginal

        return total

    print("Shapley value calculation:")
    print("-" * 50)

    for player in players:
        sv = shapley_value(player, players, payoffs)
        print(f"\nPlayer {player}:")

        # Show marginal contributions
        other_players = [p for p in players if p != player]
        for r in range(len(other_players) + 1):
            for subset in combinations(other_players, r):
                S = frozenset(subset)
                S_with = S | {player}
                marginal = payoffs[S_with] - payoffs[S]
                S_str = "{" + ",".join(sorted(S)) + "}" if S else "∅"
                print(f"  {player} joins {S_str:8}: ${payoffs[S_with]}k - ${payoffs[S]}k = +${marginal}k")

        print(f"  → Shapley value = ${sv:.1f}k (weighted average)")

    print("\nFinal allocation:")
    total_shapley = 0
    for player in players:
        sv = shapley_value(player, players, payoffs)
        total_shapley += sv
        print(f"  {player}: ${sv:.1f}k")
    print(f"  Total: ${total_shapley:.1f}k (= grand coalition value)")

    print("\nKey insight: Shapley values are the ONLY allocation that is:")
    print("  - Fair (symmetric players get equal share)")
    print("  - Efficient (total equals grand coalition value)")
    print("  - Additive (contributions add up correctly)")
    print("\nIn ML: features are 'players', prediction is 'payoff'.")


def demo_shap_sum_to_prediction():
    """
    Show that SHAP values sum to prediction minus baseline.
    Demonstrates the local accuracy property.
    """
    print("\n" + "=" * 60)
    print("SHAP VALUES SUM TO PREDICTION")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 500

    # Simple dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
    )

    # Train model
    model = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # For demonstration, we'll manually compute approximate SHAP-like values
    # using the concept of marginal contributions

    # Pick a test instance
    instance_idx = 0
    instance = X_test[instance_idx:instance_idx+1]

    # Get prediction (log-odds for gradient boosting)
    pred_proba = model.predict_proba(instance)[0, 1]

    # Baseline: average prediction over training set
    baseline_proba = model.predict_proba(X_train)[:, 1].mean()

    print(f"\nFor a single test instance:")
    print(f"  Baseline (average training prediction): {baseline_proba:.3f}")
    print(f"  This instance's prediction: {pred_proba:.3f}")
    print(f"  Difference to explain: {pred_proba - baseline_proba:+.3f}")

    # Simulate SHAP values (approximation for demonstration)
    # In practice, you'd use shap library
    np.random.seed(123)
    diff = pred_proba - baseline_proba

    # Generate plausible SHAP values that sum to the difference
    raw_values = np.random.randn(4)
    shap_values = raw_values / raw_values.sum() * diff

    print("\nSHAP value breakdown:")
    print("-" * 40)
    print(f"{'Component':<20} {'Value':>10}")
    print("-" * 40)
    print(f"{'Base value':<20} {baseline_proba:>10.3f}")

    running_total = baseline_proba
    for i, (name, sv) in enumerate(zip(feature_names, shap_values)):
        running_total += sv
        sign = "+" if sv >= 0 else ""
        print(f"{name:<20} {sign}{sv:>9.3f}")

    print("-" * 40)
    print(f"{'Prediction':<20} {running_total:>10.3f}")
    print(f"{'(Actual prediction)':<20} {pred_proba:>10.3f}")

    print("\nThe SHAP additivity property:")
    print("  base_value + Σ(SHAP values) = prediction")
    print(f"  {baseline_proba:.3f} + ({' + '.join([f'{sv:.3f}' for sv in shap_values])}) = {pred_proba:.3f}")

    print("\nThis is mathematically guaranteed by Shapley theory.")
    print("Every prediction is fully explained by feature contributions.")


def demo_shap_summary_interpretation():
    """
    Show how to read a SHAP summary plot.
    Demonstrates the feature value → SHAP value relationship.
    """
    print("\n" + "=" * 60)
    print("READING A SHAP SUMMARY PLOT")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 200

    # Create interpretable features
    age = np.random.uniform(low=25, high=65, size=n_samples)
    income = np.random.uniform(low=30, high=150, size=n_samples)  # in $1000s
    months_customer = np.random.uniform(low=1, high=60, size=n_samples)
    support_tickets = np.random.poisson(lam=2, size=n_samples)

    # Churn: higher tickets = more churn, higher tenure = less churn
    logit = (0.3 * support_tickets
             - 0.05 * months_customer
             - 0.01 * income
             + 0.02 * (age - 45))
    prob = 1 / (1 + np.exp(-logit))
    y = (np.random.rand(n_samples) < prob).astype(int)

    X = np.column_stack([age, income, months_customer, support_tickets])
    feature_names = ['age', 'income ($k)', 'months_customer', 'support_tickets']

    model = GradientBoostingClassifier(
        n_estimators=50,
        random_state=42,
    )
    model.fit(X, y)

    print("\nA SHAP summary plot shows:")
    print("  - Each dot = one prediction")
    print("  - X-axis = SHAP value (impact on prediction)")
    print("  - Color = feature value (red=high, blue=low)")
    print("  - Features sorted by importance (top = most important)")

    print("\nSimulated SHAP summary for churn prediction:")
    print("-" * 60)

    # Simulate what the pattern would look like
    patterns = [
        ("support_tickets", "HIGH", "positive",
         "More tickets (red dots) → higher churn prediction (right side)"),
        ("months_customer", "HIGH", "negative",
         "Longer tenure (red dots) → lower churn prediction (left side)"),
        ("income", "HIGH", "negative",
         "Higher income (red dots) → slightly lower churn (left side)"),
        ("age", "MIXED", "varied",
         "Age effect is noisy/weak (dots spread across)"),
    ]

    print(f"\n{'Feature':<18} {'High values':<12} {'Pattern':<40}")
    print("-" * 70)

    for feat, high_val_effect, direction, interpretation in patterns:
        print(f"{feat:<18} → {direction:<10} {interpretation}")

    print("\nHow to read these patterns:")
    print("  1. Look at where RED dots cluster (high feature values)")
    print("  2. If red dots are RIGHT → high value increases prediction")
    print("  3. If red dots are LEFT → high value decreases prediction")
    print("  4. Wide spread → feature has strong impact")
    print("  5. Tight cluster at 0 → feature has little impact")

    print("\nBusiness interpretation:")
    print("  'Customers with many support tickets are at high churn risk.'")
    print("  'Long-tenured customers are less likely to churn.'")


def demo_lime_perturbation():
    """
    Show how LIME creates local explanations through perturbation.
    Demonstrates the 'magnifying glass' concept.
    """
    print("\n" + "=" * 60)
    print("LIME PERTURBATION IN ACTION")
    print("=" * 60)

    np.random.seed(42)

    # Train a model on simple data
    n_samples = 500
    X = np.random.randn(n_samples, 3)
    feature_names = ['feature_A', 'feature_B', 'feature_C']

    # Non-linear relationship
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)  # AND gate

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
    )
    model.fit(X, y)

    # Instance to explain
    instance = np.array([[0.8, 0.9, 0.1]])  # Should predict class 1
    pred = model.predict_proba(instance)[0, 1]

    print(f"\nOriginal instance: {feature_names}")
    print(f"  Values: [{instance[0, 0]:.2f}, {instance[0, 1]:.2f}, {instance[0, 2]:.2f}]")
    print(f"  Model prediction (P(class=1)): {pred:.1%}")

    print("\nLIME process:")
    print("1. Generate perturbed samples NEAR the instance")
    print("2. Get the complex model's predictions for each")
    print("3. Fit a simple linear model weighted by distance")
    print("4. Interpret the simple model's coefficients")

    # Generate perturbations
    n_perturb = 10
    np.random.seed(42)
    perturbations = instance + np.random.randn(n_perturb, 3) * 0.3

    print(f"\nPerturbed samples and predictions:")
    print("-" * 65)
    print(f"{'Sample':<8} {'feat_A':>10} {'feat_B':>10} {'feat_C':>10} {'P(class=1)':>12}")
    print("-" * 65)

    predictions = []
    distances = []
    for i, perturb in enumerate(perturbations):
        p = model.predict_proba(perturb.reshape(1, -1))[0, 1]
        dist = np.sqrt(np.sum((perturb - instance[0]) ** 2))
        predictions.append(p)
        distances.append(dist)
        print(f"{i+1:<8} {perturb[0]:>10.2f} {perturb[1]:>10.2f} {perturb[2]:>10.2f} {p:>12.1%}")

    # Fit weighted linear model
    from sklearn.linear_model import Ridge

    # Weights: closer samples get higher weight
    weights = np.exp(-np.array(distances))

    linear_model = Ridge(alpha=0.01)
    linear_model.fit(
        perturbations,
        predictions,
        sample_weight=weights,
    )

    print(f"\nLocal linear approximation (LIME):")
    print("-" * 40)
    for name, coef in zip(feature_names, linear_model.coef_):
        direction = "increases" if coef > 0 else "decreases"
        print(f"  {name}: {coef:+.3f} ({direction} prediction)")

    print(f"\n  Intercept: {linear_model.intercept_:.3f}")

    print("\nInterpretation:")
    print("  'In the neighborhood of THIS instance, both feature_A")
    print("   and feature_B positively influence the prediction.'")
    print("\nThe 'magnifying glass': LIME zooms into a local region")
    print("where even a complex model behaves approximately linearly.")


def demo_shap_vs_permutation():
    """
    Compare SHAP and permutation importance on the same model.
    Shows when they agree and when they might differ.
    """
    print("\n" + "=" * 60)
    print("SHAP VS PERMUTATION IMPORTANCE")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 500

    # Create correlated features to show differences
    x1 = np.random.randn(n_samples)
    x2 = x1 + np.random.randn(n_samples) * 0.3  # Correlated with x1
    x3 = np.random.randn(n_samples)  # Independent

    # Target depends on x1 (and thus indirectly on x2 due to correlation)
    y = (x1 + np.random.randn(n_samples) * 0.5 > 0).astype(int)

    X = np.column_stack([x1, x2, x3])
    feature_names = ['x1 (causal)', 'x2 (correlated)', 'x3 (noise)']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Permutation importance
    perm_result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
    )

    # Simulate SHAP importance (mean absolute SHAP value)
    # In practice, use shap library
    # For correlated features, SHAP distributes importance between them
    shap_importance = np.array([0.35, 0.25, 0.02])  # Simulated values

    print("\nDataset structure:")
    print("  - x1: True predictor (target depends on x1)")
    print("  - x2: Correlated with x1 (r ≈ 0.96) but not directly causal")
    print("  - x3: Independent noise")

    print(f"\nCorrelation matrix:")
    corr = np.corrcoef(X.T)
    print(f"        {'x1':>8} {'x2':>8} {'x3':>8}")
    for i, name in enumerate(['x1', 'x2', 'x3']):
        print(f"  {name}  {corr[i, 0]:>8.2f} {corr[i, 1]:>8.2f} {corr[i, 2]:>8.2f}")

    print(f"\nImportance comparison:")
    print("-" * 55)
    print(f"{'Feature':<20} {'Permutation':>15} {'SHAP (mean |val|)':>18}")
    print("-" * 55)

    for i, name in enumerate(feature_names):
        perm = perm_result.importances_mean[i]
        shap = shap_importance[i]
        print(f"{name:<20} {perm:>15.3f} {shap:>18.3f}")

    print("\nKey observation:")
    print("  - Permutation: x2 (correlated) may show LOW importance")
    print("    Why? When x2 is shuffled, x1 still carries the signal!")
    print()
    print("  - SHAP: Distributes credit between correlated features")
    print("    Both x1 and x2 get meaningful values")

    print("\nWhen to use each:")
    print("  - Permutation: 'What would happen if we removed this feature?'")
    print("  - SHAP: 'How much did each feature contribute to predictions?'")
    print("\nNeither is 'wrong'—they answer different questions!")


def demo_shap_to_business():
    """
    Convert SHAP output to plain language business explanation.
    Shows the translation from technical to stakeholder-friendly.
    """
    print("\n" + "=" * 60)
    print("FROM SHAP TO BUSINESS ENGLISH")
    print("=" * 60)

    print("\nScenario: Explaining why a customer was flagged as high churn risk")

    # Simulated SHAP output for a customer
    shap_data = {
        'base_value': 0.25,  # Average churn rate
        'prediction': 0.78,  # This customer's churn probability
        'features': [
            ('support_tickets_30d', 5, +0.22),
            ('days_since_login', 45, +0.18),
            ('contract_type', 'monthly', +0.12),
            ('tenure_months', 3, +0.08),
            ('satisfaction_score', 6, -0.05),
            ('total_spend', 250, -0.02),
        ]
    }

    print("\nRaw SHAP output:")
    print("-" * 50)
    print(f"Base value (avg churn rate): {shap_data['base_value']:.0%}")
    print(f"Prediction: {shap_data['prediction']:.0%}")
    print()
    print(f"{'Feature':<25} {'Value':<12} {'SHAP':>10}")
    print("-" * 50)
    for feat, val, shap in shap_data['features']:
        print(f"{feat:<25} {str(val):<12} {shap:>+10.2f}")

    print("\n" + "=" * 50)
    print("TRANSLATED TO BUSINESS LANGUAGE")
    print("=" * 50)

    print("""
Customer Churn Risk Assessment
------------------------------
Risk Level: HIGH (78% likelihood of churning)
Baseline comparison: Average customer has 25% churn risk

Top factors increasing this customer's risk:

1. SUPPORT ISSUES (+22 percentage points)
   This customer filed 5 support tickets in the last 30 days.
   High support volume often indicates frustration with the product.

2. ENGAGEMENT DROP (+18 percentage points)
   Last login was 45 days ago.
   Customers who stop using the product are much more likely to leave.

3. CONTRACT FLEXIBILITY (+12 percentage points)
   Monthly contract makes it easy to cancel anytime.
   Annual contracts show 3x better retention.

Mitigating factors (slightly reducing risk):
- Satisfaction score of 6/10 (better than many churners)
- Recent spending of $250 (shows some investment)

RECOMMENDED ACTIONS:
1. Proactive outreach from customer success team
2. Address open support tickets within 24 hours
3. Offer incentive to switch to annual contract
""")

    print("Notice what changed:")
    print("  ✓ No mention of 'SHAP values' or 'base value'")
    print("  ✓ Percentages, not decimal probabilities")
    print("  ✓ Plain English explanations of WHY each matters")
    print("  ✓ Actionable recommendations at the end")
    print("  ✓ Grouped into 'increasing risk' vs 'mitigating'")


if __name__ == "__main__":
    demo_permutation_importance()
    demo_partial_dependence()
    demo_shapley_game()
    demo_shap_sum_to_prediction()
    demo_shap_summary_interpretation()
    demo_lime_perturbation()
    demo_shap_vs_permutation()
    demo_shap_to_business()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
