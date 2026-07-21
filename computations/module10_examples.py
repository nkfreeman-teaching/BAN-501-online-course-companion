"""
Module 10: Ethics, Deployment & Real-World ML - Numerical Examples

This script demonstrates key concepts from Module 10 with concrete calculations.
Run with: pixi run python course-companion-computations/module10_examples.py

References in Course Companion:
- demo_fairness_metrics()            -> Module 10, Section 10.1 (Fairness Metrics)
- demo_impossibility_theorem()       -> Module 10, Section 10.1 (Impossibility Theorem)
- demo_proxy_variable_correlation()  -> Module 10, Section 10.1 (Proxy Variables)
- demo_ab_test_sample_size()         -> Module 10, Section 10.3 (A/B Testing)
- demo_roi_sensitivity_analysis()    -> Module 10, Section 10.4 (ROI Calculation)

Last updated: 2026-02-22
"""

import numpy as np
from scipy import stats


def demo_fairness_metrics():
    """
    Calculate fairness metrics by hand for two demographic groups.
    Shows how same accuracy can hide very different error patterns.
    """
    print("=" * 60)
    print("CALCULATING FAIRNESS METRICS BY HAND")
    print("=" * 60)

    # Scenario: Loan approval model predictions for two groups
    # Group A: 1000 applicants (600 actually qualified, 400 not)
    # Group B: 1000 applicants (400 actually qualified, 600 not)

    print("\nScenario: Loan approval model evaluated on two demographic groups")
    print("Both groups have 1000 applicants.")
    print()

    # Group A confusion matrix
    # True positives: qualified and approved
    # False positives: not qualified but approved
    # True negatives: not qualified and denied
    # False negatives: qualified but denied

    group_a = {
        'name': 'Group A',
        'total': 1000,
        'actual_positive': 600,  # Actually qualified
        'actual_negative': 400,  # Not qualified
        'tp': 510,   # Qualified and approved
        'fp': 60,    # Not qualified but approved
        'tn': 340,   # Not qualified and denied
        'fn': 90,    # Qualified but denied
    }

    group_b = {
        'name': 'Group B',
        'total': 1000,
        'actual_positive': 400,  # Actually qualified
        'actual_negative': 600,  # Not qualified
        'tp': 320,   # Qualified and approved
        'fp': 120,   # Not qualified but approved
        'tn': 480,   # Not qualified and denied
        'fn': 80,    # Qualified but denied
    }

    for group in [group_a, group_b]:
        # Calculate metrics
        accuracy = (group['tp'] + group['tn']) / group['total']
        approval_rate = (group['tp'] + group['fp']) / group['total']
        tpr = group['tp'] / group['actual_positive']  # Recall / True Positive Rate
        fpr = group['fp'] / group['actual_negative']  # False Positive Rate
        precision = group['tp'] / (group['tp'] + group['fp'])

        group['accuracy'] = accuracy
        group['approval_rate'] = approval_rate
        group['tpr'] = tpr
        group['fpr'] = fpr
        group['precision'] = precision

        print(f"{group['name']} (Base rate: {group['actual_positive']/group['total']:.0%} qualified)")
        print(f"  Confusion Matrix: TP={group['tp']}, FP={group['fp']}, TN={group['tn']}, FN={group['fn']}")
        print(f"  Accuracy:      {accuracy:.1%}")
        print(f"  Approval Rate: {approval_rate:.1%}")
        print(f"  TPR (Recall):  {tpr:.1%}")
        print(f"  FPR:           {fpr:.1%}")
        print(f"  Precision:     {precision:.1%}")
        print()

    # Calculate fairness metrics
    print("FAIRNESS METRIC CALCULATIONS:")
    print("-" * 40)

    # Demographic Parity: Equal approval rates
    dp_ratio = group_b['approval_rate'] / group_a['approval_rate']
    print(f"\n1. Demographic Parity (equal approval rates)")
    print(f"   Group A approval rate: {group_a['approval_rate']:.1%}")
    print(f"   Group B approval rate: {group_b['approval_rate']:.1%}")
    print(f"   Ratio (B/A): {dp_ratio:.2f}")
    print(f"   Status: {'PASS (0.8-1.25)' if 0.8 <= dp_ratio <= 1.25 else 'FAIL'}")

    # Equalized Odds: Equal TPR AND equal FPR
    tpr_ratio = group_b['tpr'] / group_a['tpr']
    fpr_ratio = group_b['fpr'] / group_a['fpr']
    print(f"\n2. Equalized Odds (equal TPR and FPR)")
    print(f"   TPR: Group A = {group_a['tpr']:.1%}, Group B = {group_b['tpr']:.1%}, Ratio = {tpr_ratio:.2f}")
    print(f"   FPR: Group A = {group_a['fpr']:.1%}, Group B = {group_b['fpr']:.1%}, Ratio = {fpr_ratio:.2f}")
    tpr_ok = 0.8 <= tpr_ratio <= 1.25
    fpr_ok = 0.8 <= fpr_ratio <= 1.25
    print(f"   Status: TPR {'PASS' if tpr_ok else 'FAIL'}, FPR {'PASS' if fpr_ok else 'FAIL'}")

    # Predictive Parity: Equal precision
    prec_ratio = group_b['precision'] / group_a['precision']
    print(f"\n3. Predictive Parity (equal precision)")
    print(f"   Precision: Group A = {group_a['precision']:.1%}, Group B = {group_b['precision']:.1%}")
    print(f"   Ratio (B/A): {prec_ratio:.2f}")
    print(f"   Status: {'PASS (0.8-1.25)' if 0.8 <= prec_ratio <= 1.25 else 'FAIL'}")

    print(f"\nKey insight: Both groups have ~85% accuracy, but Group B has")
    print(f"a 33% higher false positive rate (20% vs 15%). Same accuracy,")
    print(f"very different error patterns across groups.")


def demo_impossibility_theorem():
    """
    The fairness impossibility theorem (Chouldechova 2017; Kleinberg,
    Mullainathan & Raghavan 2016): when two groups have different base rates, no
    classifier can satisfy equalized odds (equal TPR AND equal FPR) and
    predictive parity (equal PPV / precision) at the same time. Enforcing one
    forces the other to differ; only equal base rates or a perfect classifier
    escape the bind.
    """
    print("\n" + "=" * 60)
    print("FAIRNESS IMPOSSIBILITY THEOREM")
    print("=" * 60)

    # Two groups with DIFFERENT base rates (prevalence of the positive class).
    p_a, p_b = 0.60, 0.30
    print(f"\nGroup A base rate: {p_a:.0%} positive")
    print(f"Group B base rate: {p_b:.0%} positive")

    def ppv(tpr, fpr, prevalence):
        # Precision: P(actually positive | predicted positive), by Bayes' rule.
        tp = tpr * prevalence
        fp = fpr * (1 - prevalence)
        return tp / (tp + fp)

    # --- Enforce EQUALIZED ODDS: identical TPR and FPR in both groups. --------
    tpr, fpr = 0.80, 0.20
    ppv_a = ppv(tpr, fpr, p_a)
    ppv_b = ppv(tpr, fpr, p_b)

    print("\nSuppose we enforce EQUALIZED ODDS (identical TPR and FPR for both):")
    print(f"{'Group':>8} {'Base rate':>11} {'TPR':>7} {'FPR':>7} {'Precision':>12}")
    print("-" * 48)
    print(f"{'A':>8} {p_a:>11.0%} {tpr:>7.0%} {fpr:>7.0%} {ppv_a:>12.1%}")
    print(f"{'B':>8} {p_b:>11.0%} {tpr:>7.0%} {fpr:>7.0%} {ppv_b:>12.1%}")
    print(f"\n  TPR and FPR are equal, yet precision differs: "
          f"{ppv_a:.1%} vs {ppv_b:.1%}")
    print(f"  ({ppv_a - ppv_b:.1%} gap) -- predictive parity is broken purely")
    print("  because the base rates differ.")

    # --- Converse (Chouldechova's identity): hold precision + TPR fixed and the
    #     false-positive rates are forced apart:
    #     FPR = (p / (1 - p)) * ((1 - PPV) / PPV) * (1 - FNR)
    target_ppv = 0.75
    fnr = 0.20                       # so TPR = 1 - FNR = 80% for both groups

    def fpr_needed(prevalence):
        return (prevalence / (1 - prevalence)) * ((1 - target_ppv) / target_ppv) * (1 - fnr)

    fpr_a = fpr_needed(p_a)
    fpr_b = fpr_needed(p_b)
    print(f"\nConversely, forcing equal precision ({target_ppv:.0%}) and equal TPR "
          f"({1 - fnr:.0%})")
    print("forces the false-positive rates apart:")
    print(f"  Group A FPR = {fpr_a:.1%}     Group B FPR = {fpr_b:.1%}")

    print("\nConclusion: with unequal base rates you must give up at least one of")
    print("equal TPR, equal FPR, or equal precision -- they cannot all hold at")
    print("once. This is a mathematical constraint (Chouldechova 2017), not a")
    print("modeling failure. Which fairness criterion to prioritize is an ethical")
    print("choice, not a calculation.")

def demo_proxy_variable_correlation():
    """
    Show how proxy variables can encode protected attributes.
    Demonstrates correlation between ZIP code and income.
    """
    print("\n" + "=" * 60)
    print("DETECTING PROXY VARIABLES")
    print("=" * 60)

    np.random.seed(42)

    # Simulate: ZIP codes that correlate with income and race
    n = 1000

    # Create demographic groups
    group = np.random.choice(['A', 'B'], size=n, p=[0.7, 0.3])

    # Income differs by group (structural inequality)
    income = np.where(
        group == 'A',
        np.random.normal(loc=75000, scale=20000, size=n),
        np.random.normal(loc=55000, scale=18000, size=n)
    )
    income = np.clip(income, 20000, 200000)

    # ZIP code correlates with income (residential segregation)
    # Higher income -> higher ZIP code (simplified model)
    zip_score = income / 1000 + np.random.normal(loc=0, scale=10, size=n)

    # Credit score also correlates with income
    credit_score = 500 + (income / 500) + np.random.normal(loc=0, scale=50, size=n)
    credit_score = np.clip(credit_score, 300, 850)

    print("\nScenario: Building a loan approval model")
    print("Protected attribute: Demographic group (A vs B)")
    print("We removed group from the model. Is it still biased?")
    print()

    # Calculate correlations
    group_numeric = np.where(group == 'A', 0, 1)

    corr_zip_income = np.corrcoef(zip_score, income)[0, 1]
    corr_zip_group = np.corrcoef(zip_score, group_numeric)[0, 1]
    corr_credit_group = np.corrcoef(credit_score, group_numeric)[0, 1]
    corr_income_group = np.corrcoef(income, group_numeric)[0, 1]

    print("CORRELATION ANALYSIS:")
    print("-" * 40)
    print(f"{'Variable Pair':<30} {'Correlation':>12}")
    print("-" * 40)
    print(f"{'ZIP score ↔ Income':<30} {corr_zip_income:>12.3f}")
    print(f"{'ZIP score ↔ Group':<30} {corr_zip_group:>12.3f}")
    print(f"{'Credit score ↔ Group':<30} {corr_credit_group:>12.3f}")
    print(f"{'Income ↔ Group':<30} {corr_income_group:>12.3f}")

    # Show mean differences
    print("\nMEAN VALUES BY GROUP:")
    print("-" * 40)
    print(f"{'Metric':<20} {'Group A':>12} {'Group B':>12} {'Difference':>12}")
    print("-" * 40)

    income_a = income[group == 'A'].mean()
    income_b = income[group == 'B'].mean()
    zip_a = zip_score[group == 'A'].mean()
    zip_b = zip_score[group == 'B'].mean()
    credit_a = credit_score[group == 'A'].mean()
    credit_b = credit_score[group == 'B'].mean()

    print(f"{'Income':<20} ${income_a:>10,.0f} ${income_b:>10,.0f} ${income_a - income_b:>10,.0f}")
    print(f"{'ZIP score':<20} {zip_a:>12.1f} {zip_b:>12.1f} {zip_a - zip_b:>12.1f}")
    print(f"{'Credit score':<20} {credit_a:>12.0f} {credit_b:>12.0f} {credit_a - credit_b:>12.0f}")

    print("\nTHE PROXY PROBLEM:")
    print("-" * 40)
    print("Even without 'group' in the model, ZIP code and credit score")
    print("are correlated with group membership (r = -0.35 and -0.36).")
    print()
    print("A model using these features will produce different outcomes")
    print("by group, even though group is not an explicit input.")
    print()
    print("This is why 'we don't use race in our model' is not sufficient")
    print("to guarantee fairness. Proxy variables carry the same information.")


def demo_roi_sensitivity_analysis():
    """
    Calculate ROI for an ML project with sensitivity to assumptions.
    Shows how varying key parameters affects the business case.
    """
    print("\n" + "=" * 60)
    print("ROI SENSITIVITY ANALYSIS")
    print("=" * 60)

    print("\nScenario: Churn prediction model for subscription business")
    print()

    # Base case assumptions
    base = {
        'annual_churners': 10000,
        'customer_ltv': 500,
        'model_recall': 0.75,  # Catches 75% of churners
        'intervention_success': 0.30,  # 30% of interventions work
        'development_cost': 100000,
        'annual_maintenance': 20000,
        'cost_per_intervention': 50,
    }

    def calculate_roi(params):
        """Calculate ROI given parameters."""
        # Customers saved
        flagged = params['annual_churners'] * params['model_recall']
        saved = flagged * params['intervention_success']
        value_saved = saved * params['customer_ltv']

        # Costs
        intervention_cost = flagged * params['cost_per_intervention']
        total_cost = (params['development_cost'] +
                      params['annual_maintenance'] +
                      intervention_cost)

        # ROI
        net_value = value_saved - total_cost
        roi = net_value / total_cost

        return {
            'flagged': flagged,
            'saved': saved,
            'value_saved': value_saved,
            'total_cost': total_cost,
            'net_value': net_value,
            'roi': roi
        }

    # Base case
    base_result = calculate_roi(base)

    print("BASE CASE:")
    print("-" * 50)
    print(f"Annual churning customers:     {base['annual_churners']:>10,}")
    print(f"Customer lifetime value:       ${base['customer_ltv']:>9,}")
    print(f"Model recall:                  {base['model_recall']:>10.0%}")
    print(f"Intervention success rate:     {base['intervention_success']:>10.0%}")
    print(f"Development cost:              ${base['development_cost']:>9,}")
    print(f"Annual maintenance:            ${base['annual_maintenance']:>9,}")
    print(f"Cost per intervention:         ${base['cost_per_intervention']:>9,}")
    print()
    print("RESULTS:")
    print(f"Customers flagged:             {base_result['flagged']:>10,.0f}")
    print(f"Customers saved:               {base_result['saved']:>10,.0f}")
    print(f"Value saved:                   ${base_result['value_saved']:>9,.0f}")
    print(f"Total costs:                   ${base_result['total_cost']:>9,.0f}")
    print(f"Net value (Year 1):            ${base_result['net_value']:>9,.0f}")
    print(f"ROI:                           {base_result['roi']:>10.0%}")

    # Sensitivity analysis
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)

    print("\nVarying one parameter at a time (+/- from base case):")
    print()

    sensitivities = [
        ('model_recall', [0.60, 0.75, 0.90], 'Model Recall'),
        ('intervention_success', [0.20, 0.30, 0.40], 'Intervention Success'),
        ('customer_ltv', [400, 500, 600], 'Customer LTV'),
        ('development_cost', [75000, 100000, 150000], 'Development Cost'),
    ]

    print(f"{'Parameter':<25} {'Low':>12} {'Base':>12} {'High':>12}")
    print("-" * 65)

    for param_name, values, label in sensitivities:
        rois = []
        for val in values:
            test_params = base.copy()
            test_params[param_name] = val
            result = calculate_roi(test_params)
            rois.append(result['roi'])

        print(f"{label:<25} {rois[0]:>11.0%} {rois[1]:>11.0%} {rois[2]:>11.0%}")

    # Best and worst case
    print("\n" + "=" * 60)
    print("SCENARIO ANALYSIS")
    print("=" * 60)

    pessimistic = base.copy()
    pessimistic['model_recall'] = 0.60
    pessimistic['intervention_success'] = 0.20
    pessimistic['customer_ltv'] = 400
    pessimistic['development_cost'] = 150000

    optimistic = base.copy()
    optimistic['model_recall'] = 0.90
    optimistic['intervention_success'] = 0.40
    optimistic['customer_ltv'] = 600
    optimistic['development_cost'] = 75000

    pess_result = calculate_roi(pessimistic)
    opt_result = calculate_roi(optimistic)

    print(f"\n{'Scenario':<20} {'Net Value':>15} {'ROI':>12}")
    print("-" * 50)
    print(f"{'Pessimistic':<20} ${pess_result['net_value']:>13,.0f} {pess_result['roi']:>11.0%}")
    print(f"{'Base Case':<20} ${base_result['net_value']:>13,.0f} {base_result['roi']:>11.0%}")
    print(f"{'Optimistic':<20} ${opt_result['net_value']:>13,.0f} {opt_result['roi']:>11.0%}")

    print("\nKEY INSIGHT:")
    print("-" * 50)
    print("Even pessimistic scenario shows positive ROI (~2%).")
    print("Model recall and intervention success are the biggest drivers.")
    print("Present stakeholders with the range, not just the base case.")


def demo_ab_test_sample_size():
    """
    Calculate required sample size for A/B testing a model change.
    Shows the relationship between effect size, power, and sample size.
    """
    print("\n" + "=" * 60)
    print("A/B TEST SAMPLE SIZE CALCULATION")
    print("=" * 60)

    print("\nScenario: Testing whether a new model improves conversion rate")
    print("Current conversion rate (control): 5.0%")
    print("Minimum detectable effect: We want to detect a 0.5% absolute improvement")
    print("Target conversion rate (treatment): 5.5%")
    print()

    # Parameters
    p_control = 0.05  # 5% baseline conversion
    p_treatment = 0.055  # 5.5% target (10% relative improvement)
    alpha = 0.05  # Significance level (Type I error rate)
    power = 0.80  # Statistical power (1 - Type II error rate)

    # Sample size formula for two-proportion z-test
    # n = 2 * (z_alpha + z_beta)^2 * p_pooled * (1 - p_pooled) / (p1 - p2)^2

    z_alpha = stats.norm.ppf(1 - alpha / 2)  # Two-tailed test
    z_beta = stats.norm.ppf(power)

    p_pooled = (p_control + p_treatment) / 2
    effect_size = p_treatment - p_control

    n_per_group = (2 * (z_alpha + z_beta) ** 2 * p_pooled * (1 - p_pooled) /
                   effect_size ** 2)
    n_per_group = int(np.ceil(n_per_group))
    n_total = 2 * n_per_group

    print("CALCULATION:")
    print("-" * 50)
    print(f"Baseline conversion (p_control):    {p_control:.1%}")
    print(f"Target conversion (p_treatment):    {p_treatment:.1%}")
    print(f"Absolute effect size:               {effect_size:.1%}")
    print(f"Relative effect size:               {effect_size/p_control:.0%}")
    print(f"Significance level (alpha):         {alpha:.0%}")
    print(f"Statistical power:                  {power:.0%}")
    print()
    print(f"Z-score for alpha:                  {z_alpha:.3f}")
    print(f"Z-score for power:                  {z_beta:.3f}")
    print()
    print(f"REQUIRED SAMPLE SIZE:")
    print(f"  Per group:  {n_per_group:,} users")
    print(f"  Total:      {n_total:,} users")

    # Show how sample size changes with different parameters
    print("\n" + "=" * 60)
    print("SENSITIVITY TO PARAMETERS")
    print("=" * 60)

    print("\nEffect of minimum detectable effect on sample size:")
    print(f"{'Effect Size':<15} {'N per group':>15} {'Total N':>15}")
    print("-" * 50)

    for effect in [0.25, 0.50, 0.75, 1.00]:
        effect_pct = effect / 100
        p_treat = p_control + effect_pct
        n = (2 * (z_alpha + z_beta) ** 2 * ((p_control + p_treat) / 2) *
             (1 - (p_control + p_treat) / 2) / effect_pct ** 2)
        n = int(np.ceil(n))
        print(f"{effect:.2f}% absolute  {n:>15,} {2*n:>15,}")

    print("\nEffect of statistical power on sample size:")
    print(f"{'Power':<15} {'N per group':>15} {'Total N':>15}")
    print("-" * 50)

    for pwr in [0.70, 0.80, 0.90, 0.95]:
        z_b = stats.norm.ppf(pwr)
        n = (2 * (z_alpha + z_b) ** 2 * p_pooled * (1 - p_pooled) /
             effect_size ** 2)
        n = int(np.ceil(n))
        print(f"{pwr:.0%}            {n:>15,} {2*n:>15,}")

    # Test duration estimate
    print("\n" + "=" * 60)
    print("TEST DURATION ESTIMATE")
    print("=" * 60)

    daily_visitors = 10000
    days_needed = np.ceil(n_total / daily_visitors)

    print(f"\nWith {daily_visitors:,} daily visitors:")
    print(f"  Required: {n_total:,} total users")
    print(f"  Duration: {days_needed:.0f} days ({days_needed/7:.1f} weeks)")
    print()
    print("PRACTICAL CONSIDERATIONS:")
    print("-" * 50)
    print("1. Smaller effects need larger samples (quadratic relationship)")
    print("2. Higher power (less risk of missing real effect) needs more data")
    print("3. Plan test duration BEFORE starting - don't peek at results early")
    print("4. Include guardrail metrics to detect harm even if main metric improves")


if __name__ == "__main__":
    demo_fairness_metrics()
    demo_impossibility_theorem()
    demo_proxy_variable_correlation()
    demo_roi_sensitivity_analysis()
    demo_ab_test_sample_size()
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
